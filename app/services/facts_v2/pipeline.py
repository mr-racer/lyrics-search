"""Classify a batch of raw facts, then rewrite each kept one on its own.

Two stages, and the split is the point. Sorting is not selecting: a batch of
six facts asked "which label does each of these carry" leaves none of them
competing for a slot, so nothing gets crowded out — which is exactly what the
old five-at-a-time "pick the interesting ones" prompt did. Rewriting IS a
per-fact job, so it gets one call per fact, with the prompt chosen by label.

Measured against the previous pipeline on the same 1199 production facts:
620 facts reach text instead of 511, `other` on the artist side falls from
45.2% to 27.8%, answers in the wrong language from 4.5% to zero, and hard JSON
failures from 6 to 1.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from app.services import text_quality as tq
from app.services.facts_v2 import prompts as P
from app.services.facts_v2.prompts import ARTIST_LABELS, SONG_LABELS  # noqa: F401

logger = logging.getLogger(__name__)

CLASSIFY_BATCH = 6
MIN_FACT_CHARS = 30
MAX_FACT_CHARS = 1500

_ANNOTATION_RE = re.compile(r"^Lyrics string:\s*(.*?)\.?\s*Fact:\s*(.*)$", re.S)
_ROSTER_WORD = re.compile(r"(?i)\b(guitar|bass|drums|vocals|keyboards|piano|"
                          r"sax|saxophone|harmonica|percussion)\b")


# ── code gates, before any model call ────────────────────────────────────────

def roster_dump(text: str) -> bool:
    """A line-up listing, not a fact: `1987-2002 Layne Staley Vocals, guitar …`.

    Measured: 17 of 19 of these reached the rewrite stage and made up 7% of
    everything labelled `band_history`, producing "facts" that are lists of
    names with instruments. A negative example in the prompt did not hold them
    and does not need to — this costs nothing.
    """
    t = " ".join((text or "").split())
    instruments = len(_ROSTER_WORD.findall(t))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", t))
    sentences = sum(t.count(c) for c in ".!?")
    return instruments >= 3 and (years >= 2 or sentences <= 1)


def junk(text: str) -> bool:
    t = (text or "").strip()
    return not t or t in {"?", "??", "..."} or len(t) < MIN_FACT_CHARS


def gate(fact: dict) -> Optional[str]:
    """Why this raw fact never reaches the model, or None."""
    body = fact.get("fact") or ""
    if junk(body):
        return "junk"
    if roster_dump(body):
        return "roster"
    return None


# ── rendering ────────────────────────────────────────────────────────────────

def split_annotation(text: str):
    m = _ANNOTATION_RE.match(text or "")
    if not m:
        return None
    return " ".join(m.group(1).split()), " ".join(m.group(2).split())


def for_classify(fact: dict) -> str:
    """A Genius line note keeps its line/note split visible, because the prompt
    tells the model to judge the NOTE and never the lyric it hangs on."""
    text = (fact.get("fact") or "").strip()
    if fact.get("category") == "genius_annotation":
        parsed = split_annotation(text)
        if parsed:
            q, n = parsed
            return f'Line: "{q[:160]}" | Note: {n[:700]}'
    return " ".join(text.split())[:700]


def for_refine(fact: dict) -> str:
    text = (fact.get("fact") or "").strip()
    if fact.get("category") == "genius_annotation":
        parsed = split_annotation(text)
        if parsed:
            q, n = parsed
            return f'Lyric line: "{q[:200]}"\nNote: {n[:MAX_FACT_CHARS]}'
    return " ".join(text.split())[:MAX_FACT_CHARS]


def parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# ── stage 1 ──────────────────────────────────────────────────────────────────

def _parse_labels(obj, allowed: set) -> dict:
    out: dict = {}
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for it in obj["items"]:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            labels = [str(x) for x in (it.get("labels") or []) if str(x) in allowed]
            if "other" in labels and len(labels) > 1:
                labels = [x for x in labels if x != "other"]     # code overrules
            out[str(it["id"])] = {"labels": labels, "move": it.get("move")}
    return out


async def classify_entity(ask, entity: dict, scope: str, facts: list) -> list:
    """Label every fact of one entity. Returns [{fact, labels, move}].

    Batched, with a per-fact fallback when a batch will not parse. That fallback
    is not defensive decoration: in the validation run 18 facts came back
    unlabelled and all 18 belonged to just two entities, because one unparseable
    response takes its whole batch down with it and a track silently ends up
    with nothing at all.
    """
    allowed = SONG_LABELS if scope == "song" else ARTIST_LABELS
    out: list = []

    def build(batch: list) -> str:
        items = "\n".join(f"M{i + 1}. {for_classify(f)}"
                          for i, f in enumerate(batch))
        if scope == "song":
            return P.SONG_CLASSIFY.format(title=entity.get("title", ""),
                                          artist=entity.get("artist", ""),
                                          items=items)
        return P.ARTIST_CLASSIFY.format(artist=entity.get("name", ""), items=items)

    for start in range(0, len(facts), CLASSIFY_BATCH):
        batch = facts[start:start + CLASSIFY_BATCH]
        parsed = _parse_labels(parse_json(await ask(build(batch), 0.2)), allowed)
        for i, f in enumerate(batch):
            got = parsed.get(f"M{i + 1}")
            if got:
                out.append({"fact": f, **got})
        for i, f in enumerate(batch):
            if parsed.get(f"M{i + 1}"):
                continue
            solo = _parse_labels(parse_json(await ask(build([f]), 0.2)), allowed)
            got = solo.get("M1") or {"labels": ["other"], "move": None}
            out.append({"fact": f, **got, "solo": True})
    return out


# ── routing ──────────────────────────────────────────────────────────────────

def route(labels: list, move: Optional[dict]) -> dict:
    """Which prompts a fact goes through.

    Two rules, both of which cost real data when they were absent.

    `sample` is ORTHOGONAL: it produces a row in ``sample_links``, not prose, so
    it must not displace a text prompt. Routing a `creation + sample` fact to
    the sample branch threw the creation story away entirely — seven times in a
    1199-fact sample.

    Two text labels MERGE, ordered by specificity. Taking "the first label in
    the list" is not a rule at all: the same pair arrived as
    ['name_origin','band_history'] and ['band_history','name_origin'] in one run
    and took different prompts purely because of the order the model listed them.
    """
    labels = list(labels or [])
    plan = {"extract": "sample" in labels, "primary": None, "focus": [],
            "moved_to": None}

    movers = {"about_artist": "artist", "about_song": "song"}
    mover = next((x for x in labels if x in movers), None)
    if mover:
        plan["moved_to"] = movers[mover]
        text_labels = [x for x in ((move or {}).get("labels") or []) if x in P.FOCUS]
    else:
        text_labels = [x for x in labels if x in P.FOCUS]

    ordered = sorted(text_labels,
                     key=lambda x: P.SPECIFICITY.index(x)
                     if x in P.SPECIFICITY else 99)
    if ordered:
        plan["primary"] = ordered[0]
        plan["focus"] = ordered
    return plan


# ── stage 2 ──────────────────────────────────────────────────────────────────

async def refine_one(ask, rec: dict, entity: dict, scope: str, *,
                     lang_name: str = "Russian", lang_code: str = "ru") -> dict:
    """Rewrite one fact, or extract its sampling link, and validate the result."""
    plan = route(rec["labels"], rec.get("move"))
    rec.update({"primary": plan["primary"], "focus_labels": plan["focus"],
                "moved_to": plan["moved_to"]})
    fact_text = for_refine(rec["fact"])

    if plan["extract"]:
        obj = parse_json(await ask(P.SAMPLE_EXTRACT.format(
            title=entity.get("title", ""),
            artist=entity.get("artist") or entity.get("name", ""),
            fact=fact_text), 0.2))
        rec["links"] = (obj or {}).get("links") or []

    if not plan["primary"]:
        return rec

    eff_scope = plan["moved_to"] or scope
    if eff_scope == "song":
        subject = P.SONG_SUBJECT.format(lang=lang_name)
        title, artist = entity.get("title", ""), entity.get("artist", "")
    else:
        subject = P.ARTIST_SUBJECT
        title, artist = "", entity.get("name") or entity.get("artist", "")

    focus = "\n\n".join(P.FOCUS[label] for label in plan["focus"])
    cap = max(P.MAX_CHARS.get(label, 220) for label in plan["focus"])
    name_rule = (f'- The artist is written exactly "{artist}". Copy that spelling.'
                 if artist else "")
    text = ((parse_json(await ask(P.REFINE.format(
        subject=subject, focus=focus, lang=lang_name,
        shots=P.SHOTS[plan["primary"]], max_chars=cap,
        name_rule=name_rule, fact=fact_text), 0.3)) or {}).get("text") or "").strip()
    if not text:
        return rec

    src = f"{fact_text} {artist} {title}"
    forbid = title if eff_scope == "song" else ""
    text, issues, notes = await _repair(ask, text, src=src, cap=cap,
                                        forbid=forbid, lang_name=lang_name,
                                        lang_code=lang_code)
    rec.update(notes)

    # An answer still in the wrong language after a retry is not shown: a
    # Russian interface showing an English fact is worse than showing none.
    if issues.get("wrong_language"):
        rec["dropped"] = "wrong_language"
        rec["refined"] = ""
    else:
        rec["refined"] = text
    rec["issues_final"] = sorted(issues)
    return rec


async def _repair(ask, text: str, *, src: str, cap: int, forbid: str,
                  lang_name: str, lang_code: str) -> tuple:
    """Bring one piece of text into line. Code first, model only for the rest.

    The order is load-bearing. Asking the model for the replacement list first
    produced inventions (`Фромажо → Fromage`) and half-swapped hybrids
    («Антони Gonzalez»), because it is good at spotting WHICH word is wrong and
    bad at spelling it. The deterministic matcher works from the source
    spellings, so it gets both right; with it running first it handled 25 of 25
    script faults and the model was not called at all.
    """
    notes: dict = {}
    issues = tq.check(text, source=src, lang=lang_code, max_chars=cap,
                      forbid=forbid)
    notes["issues_first"] = sorted(issues)

    if issues.get("translit"):
        text, swaps = tq.restore_latin(text, src)
        if swaps:
            notes["swaps"] = [(a, b) for a, b, _ in swaps]
            issues = tq.check(text, source=src, lang=lang_code, max_chars=cap,
                              forbid=forbid)

    if any(k in issues for k in tq.SCRIPT_FAULTS):
        obj = parse_json(await ask(tq.REPLACE_PROMPT.format(
            text=text, complaints=tq.complaints(issues, lang_name)), 0.1))
        swapped, done, skipped = tq.apply_replacements(
            text, (obj or {}).get("replace"), source=src)
        if done:
            text, notes["replacements"] = swapped, done
            issues = tq.check(text, source=src, lang=lang_code, max_chars=cap,
                              forbid=forbid)
        if skipped:
            notes["repl_skipped"] = skipped

    # Only a fault that needs DIFFERENT SENTENCES gets a rewrite, and the result
    # has to survive a content guard: a rewrite that fixes the named fault and
    # wrecks the sentence around it passes every check otherwise.
    if tq.needs_rewrite(issues):
        fixed = ((parse_json(await ask(tq.REPAIR_PROMPT.format(
            text=text, complaints=tq.complaints(issues, lang_name)), 0.2))
            or {}).get("text") or "").strip()
        if fixed and tq.repair_is_safe(text, fixed):
            after = tq.check(fixed, source=src, lang=lang_code, max_chars=cap,
                             forbid=forbid)
            if len(after) < len(issues):
                text, issues, notes["repaired"] = fixed, after, True
        elif fixed:
            notes["repair_rejected"] = True
    return text, issues, notes


async def process_entity(ask: Callable, entity: dict, scope: str, facts: list,
                         *, lang_name: str = "Russian", lang_code: str = "ru",
                         on_result=None) -> list:
    """Gate → classify → refine, persisting each fact as soon as it is done.

    ``on_result`` is called per fact with the finished record. Persisting inside
    the loop rather than at the end is what makes a 64k-fact run resumable: it
    will be interrupted, and an entity-at-a-time commit loses everything after
    the last completed entity.
    """
    out: list = []
    kept: list = []
    for fact in facts:
        why = gate(fact)
        if why:
            rec = {"fact": fact, "labels": [f"gate:{why}"], "move": None,
                   "gated": why}
            out.append(rec)
            if on_result:
                on_result(rec)
        else:
            kept.append(fact)

    if not kept:
        return out

    for rec in await classify_entity(ask, entity, scope, kept):
        try:
            await refine_one(ask, rec, entity, scope, lang_name=lang_name,
                             lang_code=lang_code)
        except Exception as exc:                    # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("[facts_v2] refine failed for %s: %s",
                           entity.get("slug"), exc)
        out.append(rec)
        if on_result:
            on_result(rec)
    return out
