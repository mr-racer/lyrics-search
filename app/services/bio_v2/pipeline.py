"""Write one artist's biography and its facets from their Wikipedia article.

The shape of the run, and why each part is there:

  find the article  → the one question whose target catalogue is known, so it
                      goes to Wikipedia's own index rather than a web engine
  fetch once        → the network is slower than generation
  index once        → every later question is a new QUERY, not a new round trip
  five queries, RRF → a biography made of facets instead of a lead paragraph
  facets            → sentence-level retrieval, because the answer is one
                      sentence inside a chunk about something else
  script control    → shared with the fact pipeline, byte for byte
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.resources.model_registry import ModelRegistry
from app.services import text_quality as tq
from app.services.assistant.config import AgentConfig
from app.services.assistant.fetcher import PageFetcher
from app.services.bio_v2 import article as art
from app.services.bio_v2 import prompts as P
from app.services.bio_v2 import retrieval as R

logger = logging.getLogger(__name__)

FACETS = {
    "grammy": "Grammy Awards won and nominations received",
    "formed": "the year and place the band was formed or the artist began",
    "name_origin": "where the band's or artist's name came from and who chose it",
    "years_active": "years active, disbanding, hiatus, death, current status",
}

_STATUSES = {"active", "disbanded", "hiatus", "deceased"}


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


def _int(value) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n


def _empty(obj: Optional[dict]) -> bool:
    if not obj:
        return True
    return all(obj.get(k) in (None, "", "null")
               for k in obj if k != "evidence")


async def write_bio(ask, artist: str, chunks: list, retriever, *,
                    lang_name: str = "Russian",
                    lang_code: str = "ru") -> tuple:
    """(text, notes) — the biography, script-checked. Empty text means none."""
    order = R.bio_chunks(retriever, artist)
    if not order:
        return "", {"why": f"no chunk cleared p>={R.CE_CHUNK_GATE}"}
    ctx = R.passages(chunks, order)

    bio = await ask(P.BIO_PROMPT.format(artist=artist, lang=lang_name,
                                        passages=ctx), 0.35)
    notes = {"chunks": order}

    # A whole answer in the wrong language is not a repair job — there is
    # nothing to correct, the model did the wrong task. Ask again once, with
    # the instruction moved to the front where a small model reads it.
    if tq.no_target_script(bio, lang_code):
        notes["lang_retry"] = True
        again = await ask(P.LANG_RETRY + P.BIO_PROMPT.format(
            artist=artist, lang=lang_name, passages=ctx), 0.2)
        if again and not tq.no_target_script(again, lang_code):
            bio = again

    issues = tq.check(bio, source=ctx, lang=lang_code)
    notes["issues_first"] = sorted(issues)

    # Code before the model: the deterministic matcher works from the source
    # spellings and matches whole names, so it produces "Nicolas Fromageau"
    # where the model offered the invented "Fromage", and never the hybrid
    # «Антони Gonzalez» a surname-only swap leaves behind.
    if issues.get("translit"):
        bio, swaps = tq.restore_latin(bio, ctx)
        if swaps:
            notes["swaps"] = [(a, b) for a, b, _ in swaps]
            issues = tq.check(bio, source=ctx, lang=lang_code)

    if any(k in issues for k in tq.SCRIPT_FAULTS):
        obj = parse_json(await ask(tq.REPLACE_PROMPT.format(
            text=bio, complaints=tq.complaints(issues, lang_name)), 0.1))
        swapped, done, skipped = tq.apply_replacements(
            bio, (obj or {}).get("replace"), source=ctx)
        if done:
            bio, notes["replacements"] = swapped, done
            issues = tq.check(bio, source=ctx, lang=lang_code)
        if skipped:
            notes["repl_skipped"] = skipped

    notes["issues_final"] = sorted(issues)
    if issues.get("wrong_language"):
        notes["dropped"] = "wrong_language"
        return "", notes
    return bio.strip(), notes


async def read_facets(ask, artist: str, chunks: list, sents: tuple, *,
                      lang_name: str = "Russian", widen=None) -> dict:
    """Extract the facts shown beside the bio. Returns column → value.

    The web is touched only when the article says nothing, and what comes back
    is marked as web-sourced: an unguarded fallback once reported four Grammys
    for a New Zealand bedroom-pop musician who has none, lifted from a page
    about somebody else.
    """
    out: dict = {}
    for name, question in FACETS.items():
        order, _best = R.facet_chunks(artist, question, sents)
        source = "wiki"
        if not order and widen is not None:
            fresh = await widen(question)
            if fresh:
                chunks.extend(fresh)
                sents = R.sentence_index(chunks)
                order, _best = R.facet_chunks(artist, question, sents)
                source = "web"
        if not order:
            continue
        obj = parse_json(await ask(P.FACET_PROMPTS[name].format(
            artist=artist, lang=lang_name,
            passages=R.passages(chunks, order)), 0.2))
        if _empty(obj) and widen is not None and source == "wiki":
            fresh = await widen(question)
            if fresh:
                chunks.extend(fresh)
                sents = R.sentence_index(chunks)
                more, _ = R.facet_chunks(artist, question, sents)
                if more:
                    source = "web"
                    obj = parse_json(await ask(P.FACET_PROMPTS[name].format(
                        artist=artist, lang=lang_name,
                        passages=R.passages(chunks, more)), 0.2))
        if _empty(obj):
            continue
        out.update(_columns(name, obj, source))
    return out


def _columns(name: str, obj: dict, source: str) -> dict:
    """Map one facet answer onto its storage columns, dropping nonsense."""
    if name == "grammy":
        wins, noms = _int(obj.get("wins")), _int(obj.get("nominations"))
        if wins is None and noms is None:
            return {}
        return {"grammy_wins": wins, "grammy_nominations": noms,
                "grammy_source": source}
    if name == "formed":
        year, place = _int(obj.get("year")), (obj.get("place") or None)
        if year is not None and not (1900 <= year <= 2100):
            year = None
        if year is None and not place:
            return {}
        return {"formed_year": year, "formed_place": place,
                "formed_source": source}
    if name == "name_origin":
        origin = (obj.get("origin") or "").strip()
        if not origin or origin.lower() == "null":
            return {}
        return {"name_origin": origin[:400], "name_origin_source": source}
    if name == "years_active":
        status = (obj.get("status") or "").strip().lower()
        frm, to = _int(obj.get("from")), _int(obj.get("to"))
        if status not in _STATUSES:
            status = None
        if frm is None and to is None and status is None:
            return {}
        return {"active_from": frm, "active_to": to, "status": status,
                "status_source": source}
    return {}


async def build(ask, artist: str, *, lang_name: str = "Russian",
                lang_code: str = "ru", config: Optional[AgentConfig] = None,
                fetcher: Optional[PageFetcher] = None,
                web_search=None, proxies: Optional[dict] = None) -> dict:
    """Everything for one artist: bio text, facets, and what was rejected."""
    cfg = config or AgentConfig()
    fetcher = fetcher or PageFetcher(cfg)

    found, rejected = art.find(artist, proxies=proxies, web_search=web_search)
    if found is None:
        return {"error": "no wikipedia article passed the gate",
                "rejected": rejected}

    page = await fetcher.fetch(found["url"], source="wikipedia",
                               title=found["title"])
    if not page.ok or not page.markdown:
        return {"error": f"fetch failed: {page.error}", "rejected": rejected}

    chunks = R.chunk_page(page, cfg)
    if not chunks:
        return {"error": "no chunks", "rejected": rejected}
    retriever = R.build_index(chunks)

    bio, notes = await write_bio(ask, artist, chunks, retriever,
                                 lang_name=lang_name, lang_code=lang_code)

    async def widen(question: str) -> list:
        """One open-web search, entity-gated before it can enter the index."""
        if web_search is None:
            return []
        try:
            rows = web_search(f"{artist} {question}")
        except Exception:                        # noqa: BLE001
            return []
        fresh: list = []
        for row in (rows or [])[:2]:
            page2 = await fetcher.fetch(row.get("url") or "", source="web",
                                        title=row.get("title") or "")
            if page2.ok and page2.markdown:
                fresh += R.chunk_page(page2, cfg)
        if not fresh:
            return []
        probs = ModelRegistry.ce_probabilities(
            f"{artist} is a musical artist or band: their music, albums and "
            f"career.", [c.text[:1200] for c in fresh])
        if probs is not None:
            fresh = [c for c, p in zip(fresh, probs)
                     if p >= art.CE_ARTICLE_GATE]
        return fresh

    facets = await read_facets(ask, artist, chunks,
                               R.sentence_index(chunks),
                               lang_name=lang_name, widen=widen)
    facets.update({"source_url": found["url"], "source_kind": "wikipedia"})
    return {"bio": bio, "facets": facets, "article": found,
            "rejected": rejected, "notes": notes}
