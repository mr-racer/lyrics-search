"""Sonic Vibe task — one short fact-based line per track via LLM.

v2 (option B): the model no longer free-writes a line off a wall of raw fact
strings. The code builds a tagged, M-keyed window of raw facts (editorial and
description facts first, genius line notes reformatted and last), and the
model must FIRST pick the single winning fact by key, THEN write the line
from that fact alone:

    {"best": "M4", "category": "A", "line": "..."}   or   {"best": null}

Code validates the reply: the key must exist, the line must be clean (no raw
scaffold leakage, target-language script) and every Latin name in it must
appear in the winning fact — one retry, then SKIP. An empty slot beats an
invented line.

Reads a track's raw song facts (up to MAX_FACTS) + sonic_tags + year. Only
runs for tracks that HAVE facts. Persists the line in the sonic_vibes table
(PK track_id+collection+lang); a SKIP — or a track with no facts — leaves the
slot empty, so no vibe is shown.

The llm parameter in run() is accepted for framework compatibility but unused
at runtime — all LLM calls go through ask_llm() which resolves connection
config from environment variables.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from app.services.llm_client import ask_llm
from app.services.song_facts_service import get_song_facts_key
# Shared v2 helpers: junk gate, annotation parser, anti-hallucination check.
from app.services.ai_tasks.refined_facts import (
    _entities_ok, _has_garbled_script, _junk_reason, _parse_annotation,
)

logger = logging.getLogger(__name__)

MAX_PHRASE_CHARS = 160
MAX_FACTS = 10  # how many curated facts to show the model per track

_SYSTEM_PROMPT = """
You write ONE short line shown under a track in a music player. Its job is to tell the listener one true, concrete thing about THIS track that a fan would find interesting — a fact, not a mood. If you don't have such a fact, you output nothing.

WHAT YOU CAN AND CANNOT KNOW:
You have NOT heard this track. You only have text: a few sonic descriptor tags, a list of facts, and the year. Therefore:
- You MAY state what the FACTS say (and the era, if it adds to a fact).
- You MAY NOT describe how the vocals sound, what instruments are playing, the "feel", a scene, or what the song is about — UNLESS it is explicitly in the facts. Inventing these is the main failure. No "icy vocals", "Japanese drums", "neon", "nightclub", "political tension" unless a fact literally states it.
- If you mention a sonic descriptor tag at all, copy it EXACTLY as given — same instrument, same technique, same word. NEVER swap a tag for a different-but-related one (e.g. a tag saying "throat singing" must never become "horns", "brass" must never become "strings") and NEVER invent a tag that isn't in the given list. When in doubt, leave the tag out rather than guess at it.

Each fact below is keyed (M1, M2, ...) and tagged by origin: [EDITORIAL] and [DESCRIPTION] are curated song facts; [LINE NOTE] is a fan note explaining one lyric line.

YOUR ONLY TWO OPTIONS:

1. There IS a usable fact. Pick the SINGLE best one by this priority — higher beats lower, always take the highest available:

   A. CREATION STORY — the non-obvious path the track took to exist: made in one night / hours before a flight, sat unreleased for years, was meant for a different album, started as something else (e.g. AI-written lines later rewritten), an accident or constraint that shaped it. These are the most interesting — a fan would retell them. Prefer these above all.
   T. TITLE STORY — where the song's TITLE comes from or what it really refers to (a real product, a person, an event, a hidden meaning), when a fact explains it.
   B. CONCRETE PRODUCTION FACT — who produced it, a notable guest, an unusual instrument or recording method (only if the fact states it). NEVER build the line on a MUSICAL sample or interpolation (a borrowed melody, beat, or instrumental) — the player already shows sample credits separately; if the only production fact is "this samples X's song", treat it as unusable and look for another fact (or SKIP). SPOKEN-WORD samples are fine and often great facts: a movie dialogue, an interview, a speech or a TV monologue used in the track. Referencing another song's LYRICS or quoting a line is also fine.
   C. EXTERNAL RESULT / CONTEXT — chart milestone, award, its role on the album, real-world reaction or controversy.

   → Write ONE line from the single highest-priority fact you have. Don't cram two.

   Do NOT pick a fact that just restates what the song is ABOUT or quotes its lyrics — the listener is hearing that right now. [LINE NOTE] facts qualify ONLY when they carry an external story (a real person, work, event, or the artist's own words), never for what the line means. If the only "facts" are lyric content, go to option 2.

2. There is NO usable fact of type A/T/B/C — facts are missing, generic, biographical fluff, or only describe the song's lyrical content. → Reply {{"best": null}}
   A line built only on tags, era, or what the song is about is NOT good enough — SKIP it. An empty slot is better than a generic line.

When unsure whether a fact is interesting enough, prefer {{"best": null}}. Never reach for invented atmosphere to fill the line.

STYLE (when you do write a line):
- INVENT NOTHING. Your line must say EXACTLY what the chosen fact says — same claim, same specifics, no added color. You may only translate it into {lang_name} and shorten it; you may NOT add, merge, reinterpret, exaggerate, or "improve" details. If the fact says "recorded in a hotel room", the line says a hotel room — not "a cramped Tokyo hotel room at 4am".
- Plain and clear, like a knowledgeable friend pointing something out — NOT a magazine pull-quote, NOT poetry. No purple adjectives, no invented atmosphere, no clichés.
- One line, max ~120 characters. No emoji, no quotation marks around the whole line.
- NAMES: you may name a producer, featured guest, or album — those are the interesting facts. Do NOT repeat the main artist's name or the track title; the UI already shows them right next to your line.
- Any name you DO write (producer, guest, album) must appear EXACTLY as given in the chosen fact — character for character. NEVER translate, transliterate, localize, or grammatically decline a name into {lang_name}.
- This applies to EVERY proper name: people, family members, brands, shows, places. If a name is spelled in Latin letters in the fact (or in the Artist/Track lines of the task), it must appear in your line in the SAME Latin letters. Writing any such name in Cyrillic is an error.

RESPONSE FORMAT — strict JSON, no markdown, no text around it:
{{"best": "M4", "category": "A", "line": "..."}}
or, when no fact qualifies:
{{"best": null}}
"category" is the priority letter (A/T/B/C) of the chosen fact.

EXAMPLE (style and selection only — do NOT reuse this content):
Facts given:
  M1 [EDITORIAL] Recorded in one night in a hotel room, hours before the artist had to fly out
  M2 [EDITORIAL] Produced by a well-known beatmaker
  M3 [LINE NOTE] Note on the line "they all watch me": the hook is about feeling watched by everyone
Correct output: {{"best": "M1", "category": "A", "line": "Записан за одну ночь в отеле — за несколько часов до вылета"}}
Why: the creation story (A) outranks the producer fact (B); the lyric-content note is ignored because the listener is already hearing it.

Write "line" ONLY in {lang_name}; do not mix languages.
""".strip()

_LANG_NAMES = {"ru": "Russian", "en": "English"}

# Category-priority order for the window: editorial/description first so the
# strongest sources always make it into MAX_FACTS; line notes fill what's left.
_TAG_EDITORIAL = "[EDITORIAL]"
_TAG_DESCRIPTION = "[DESCRIPTION]"
_TAG_LINE_NOTE = "[LINE NOTE]"

# Raw-scaffold substrings that must never leak into a rendered vibe line.
_FORBIDDEN_IN_LINE = ("Fact:", "Lyrics string", "Note on the line", "[EDITORIAL]", "[DESCRIPTION]", "[LINE NOTE]")

_CYRILLIC_RE = re.compile(r"[а-яё]", re.I)


def _build_fact_window(meta_facts: list[dict]) -> list[dict]:
    """Turn raw ``{"fact", "category"}`` rows into the tagged M-keyed window.

    Junk facts are dropped; genius annotations are reformatted into readable
    English scaffolding (``Note on the line "...": ...``); editorial and
    description facts take window priority over line notes (spec 2.1) instead
    of the old blind ``facts[:10]``.
    """
    primary: list[dict] = []   # editorial + description, in original order
    notes: list[dict] = []     # line notes, in original order
    for mf in meta_facts:
        fact = mf.get("fact") or ""
        category = mf.get("category")
        if category == "genius_annotation":
            parsed = _parse_annotation(fact)
            if parsed is None:
                continue
            quote, note = parsed
            if _junk_reason(note) is not None:
                continue
            text = (
                f'Note on the line "{quote}": {note}' if quote
                else f"Note on the song: {note}"
            )
            notes.append({"tag": _TAG_LINE_NOTE, "text": text})
        else:
            if _junk_reason(fact) is not None:
                continue
            tag = _TAG_DESCRIPTION if category == "genius_description" else _TAG_EDITORIAL
            primary.append({"tag": tag, "text": fact})

    window = (primary + notes)[:MAX_FACTS]
    for i, item in enumerate(window):
        item["key"] = f"M{i + 1}"
    return window


def _build_user_prompt(
    *, tags: list[str], payload: dict, window: list[dict], lang: str,
) -> str:
    year = payload.get("year")
    era = f"{(year // 10) * 10}s" if isinstance(year, int) and year > 0 else None
    artist = (payload.get("artist") or "").strip()
    title = (payload.get("title") or "").strip()
    lines = []
    # Original Latin spelling as an explicit anchor: the model must copy names
    # as-is and never transliterate them into the target language.
    if artist:
        lines.append(f"Artist (original spelling — copy names AS IS, never transliterate): {artist}")
    if title:
        lines.append(f"Track: {title}")
    if lines:
        lines.append("")
    lines.append("FACTS (pick the single best by key, or reply {\"best\": null}):")
    for item in window:
        lines.append(f"{item['key']} {item['tag']} {item['text']}")
    lines.append("")
    lines.append(f"Sonic descriptor tags: {', '.join(tags) if tags else '(none)'}")
    lines.append(f"Era: {era or '(unknown)'}")
    lines.append("")
    lines.append("Your JSON reply:")
    return "\n".join(lines)


def _parse_vibe_response(raw: str, valid_keys: set[str]) -> tuple[str | None, str]:
    """Parse the model reply → (best_key | None, line). Raises ValueError.

    ``(None, "")`` means an explicit skip. Tolerates markdown fences and a
    legacy bare ``SKIP`` (older prompt habit gemma may fall back into).
    """
    s = (raw or "").strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.M).strip()
    if not s:
        return None, ""
    first_word = s.split()[0].rstrip(".,!:;—-").strip('"\'').upper()
    if first_word == "SKIP":
        return None, ""
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise ValueError("vibe response is not JSON")
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("vibe response is not a JSON object")
    best = data.get("best")
    if best is None:
        return None, ""
    best = str(best)
    if best not in valid_keys:
        raise ValueError(f"vibe response picked unknown key {best!r}")
    line = (data.get("line") or "").strip()
    if not line:
        raise ValueError("vibe response kept a fact but wrote no line")
    return best, line


def _validate(phrase: str) -> str:
    """Trim, drop wrapping quotes, enforce length cap."""
    phrase = (phrase or "").strip().strip('"').strip("'")
    while len(phrase) > 1 and phrase[-1] == phrase[-2] and phrase[-1] in ".!?":
        phrase = phrase[:-1]
    if len(phrase) > MAX_PHRASE_CHARS:
        phrase = phrase[:MAX_PHRASE_CHARS].rstrip() + "…"
    return phrase


def _line_ok(line: str, lang: str) -> bool:
    """Reject raw-scaffold leakage, wrong-script and garbled-script lines."""
    if any(marker.lower() in line.lower() for marker in _FORBIDDEN_IN_LINE):
        return False
    if lang == "ru" and not _CYRILLIC_RE.search(line):
        return False
    if _has_garbled_script(line):
        return False
    return True


async def run(job, db_client, llm) -> None:
    """Iterate the collection's tracks, generate a fact-based line for each.

    Skip rule: sonic vibes are FACT-BASED — a track with no curated song facts
    is skipped (no LLM call); tags alone are not enough. When facts exist the
    model may still reply {"best": null} (no fact worth showing), which leaves
    the slot empty — no vibe persisted. Also skip if a vibe is already cached
    for this (track, collection, lang).

    When ``job.new_track_ids`` is set (auto run after an append/upload) only
    those tracks are touched — the rest were enriched by an earlier run.

    The ``llm`` parameter is accepted for framework compatibility but unused —
    the module-level ask_llm() is called directly so it can be patched in tests.
    """
    qdrant = db_client.qdrant
    lang = job.lang if job.lang in _LANG_NAMES else "en"
    system = _SYSTEM_PROMPT.format(lang_name=_LANG_NAMES[lang])

    n_done = 0
    n_failed = 0
    n_skipped = 0

    if job.new_track_ids is not None:
        # Auto run after an append/upload: the batch is small, so one batched
        # retrieve instead of a whole-collection scroll (the scroll loop below
        # is the legacy manual path).
        points = qdrant.retrieve(
            collection_name=job.collection_name,
            ids=list(job.new_track_ids),
            with_payload=True,
            with_vectors=False,
        )
        batches = [points] if points else []
    else:
        def _scroll_sync() -> list:
            """Обойти коллекцию, не занимая event loop — см. тот же приём в
            `lyric_gems`. Синхронный обход из корутины держит весь сервис на
            паузе, пока не кончится; проекция payload убирает из ответа тексты
            песен, которых этой задаче не нужно.
            """
            out: list = []
            offset = None
            while True:
                points, offset = qdrant.scroll(
                    collection_name=job.collection_name,
                    limit=64,
                    offset=offset,
                    with_payload=["artist", "title", "sonic_tags_json"],
                    with_vectors=False,
                )
                if not points:
                    break
                out.append(points)
                if offset is None:
                    break
            return out

        batches = await asyncio.to_thread(_scroll_sync)

    for points in batches:
        for pt in points:
            tid = str(pt.id)
            p = pt.payload or {}

            # Cache hit counts as "done" — the track is already processed
            # from a previous run, even though no LLM call happened now.
            if MetadataDB.get_sonic_vibe(tid, job.collection_name, lang):
                n_done += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_done=n_done)
                continue

            # Gather inputs.
            tags: list[str] = []
            tags_raw = p.get("sonic_tags_json")
            if tags_raw:
                try:
                    tags = json.loads(tags_raw)
                except Exception:
                    tags = []

            # Resolve song-fact slug from artist+title in the Qdrant payload.
            # Indexing does NOT write a precomputed song_slug to the payload;
            # facts live in SQLite keyed by get_song_facts_key(artist, title).
            meta_facts: list[dict] = []
            artist_name = (p.get("artist") or "").strip()
            title_text = (p.get("title") or "").strip()
            if artist_name and title_text:
                song_slug = get_song_facts_key(artist_name, title_text)
                try:
                    meta_facts = MetadataDB.get_song_facts_with_meta(
                        song_slug, job.collection_name,
                    )
                except Exception:
                    meta_facts = []

            window = _build_fact_window(meta_facts)
            if not window:
                # Sonic vibes are fact-based — no usable facts, no LLM call
                # (tags alone are not enough). Count as skipped (NOT done) so
                # the UI can tell the job finished without work on this track.
                n_skipped += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_skipped=n_skipped)
                continue

            user = _build_user_prompt(
                tags=tags, payload=p, window=window, lang=lang,
            )
            valid_keys = {item["key"] for item in window}
            by_key = {item["key"]: item for item in window}

            phrase = ""
            skipped_by_model = False
            try:
                # One validation retry (spec 2.3), then SKIP: an empty slot
                # beats a leaked/hallucinated line.
                for attempt in (0, 1):
                    raw = await ask_llm(
                        user,
                        system_prompt=system,
                        temperature=0.7,
                        base_url=job.llm_base_url,
                        model=job.llm_model,
                    )
                    try:
                        best, line = _parse_vibe_response(raw or "", valid_keys)
                    except ValueError as ve:
                        if attempt == 0:
                            continue
                        logger.info("[sonic_vibe] %s: %s — skipping", tid, ve)
                        skipped_by_model = True
                        break
                    if best is None:
                        skipped_by_model = True
                        break
                    candidate = _validate(line)
                    winner_text = by_key[best]["text"]
                    if (
                        candidate
                        and _line_ok(candidate, lang)
                        and _entities_ok(candidate, f"{winner_text} {artist_name} {title_text}")
                    ):
                        phrase = candidate
                        break
                    if attempt == 1:
                        logger.info(
                            "[sonic_vibe] %s: line failed validation twice — skipping",
                            tid,
                        )
                        skipped_by_model = True
            except Exception as e:
                logger.warning("[sonic_vibe] LLM error on %s: %s", tid, e)
                n_failed += 1
                MetadataDB.update_ai_job(job_id=job.job_id, n_failed=n_failed)
                continue

            if phrase:
                MetadataDB.set_sonic_vibe(tid, job.collection_name, lang, phrase)
                n_done += 1
            elif skipped_by_model:
                # Model judged no fact worth showing (or failed validation) —
                # leave the slot empty. Still "processed", so count as done.
                n_done += 1
            else:
                n_failed += 1
            MetadataDB.update_ai_job(
                job_id=job.job_id, n_done=n_done, n_failed=n_failed,
            )

    if n_done == 0 and n_skipped > 0 and n_failed == 0:
        # The whole collection lacks the inputs sonic_vibe needs — surface
        # this as a job-level note so the UI can show why nothing happened.
        logger.warning(
            "[sonic_vibe] all %d tracks skipped — no song_facts in collection "
            "%s. Sonic vibes are fact-based; run facts indexing first.",
            n_skipped, job.collection_name,
        )


# Register at import time
ai_indexing_service.register_task("sonic_vibe", run)
