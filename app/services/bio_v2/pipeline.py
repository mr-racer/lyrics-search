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

import asyncio
import json
import logging
import re
from dataclasses import replace
from typing import Optional

from app.services import text_quality as tq
from app.services.assistant.config import AgentConfig
from app.services.assistant.fetcher import PageFetcher
from app.services.bio_v2 import article as art
from app.services.bio_v2 import prompts as P
from app.services.bio_v2 import retrieval as R
from app.services.bio_v2 import sources

logger = logging.getLogger(__name__)

# One open-web search per artist, and the instance that holds this number lives
# for the whole run — see ``build``.
WEB_BUDGET = 1

# Which language the CORPUS is in, which is not the language the biography is
# written in: an English article is read to write a Russian bio, and a query in
# the wrong language reaches BM25 and the sparse leg with nothing to match.
_LANG_NAME = {"ru": "Russian", "en": "English"}

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
    order = await asyncio.to_thread(R.bio_chunks, retriever, artist)
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
    # Nothing to say is said by saying nothing: the page simply shows no
    # biography, which beats a paragraph about how the search went.
    if tq.is_refusal(bio):
        notes["dropped"] = "refusal"
        return "", notes
    return bio.strip(), notes


def _corpus_lang(artist: str, meta: dict) -> str:
    """The language the passages are written in, best guess.

    The article's own subdomain when there is an article — it is the one fact
    here that is not a guess — and otherwise the language the artist's name is
    written in, which is what chose the wiki in the first place.
    """
    url = meta.get("source_url") or ""
    host = url.split("//", 1)[-1].split("/", 1)[0].lower()
    if host.endswith("wikipedia.org"):
        return _LANG_NAME.get(host.split(".", 1)[0], "English")
    return _LANG_NAME.get(art.preferred_lang(artist), "English")


async def facet_queries(ask, artist: str, corpus_lang: str) -> dict:
    """Per facet, the phrasings the corpus gets asked with.

    The hardcoded question stays FIRST in every list and the generated ones are
    added behind it. A model that returns nothing usable therefore costs the
    facets nothing — the behaviour falls back to exactly what it was — while a
    model that does understand the article's vocabulary gets to say so.
    """
    default = {name: [question] for name, question in FACETS.items()}
    topics = "\n".join(f"  {name}: \"{question}\""
                       for name, question in FACETS.items())
    try:
        obj = parse_json(await ask(P.FACET_QUERIES_PROMPT.format(
            artist=artist, lang=corpus_lang, topics=topics), 0.3))
    except Exception:                                # noqa: BLE001
        return default
    if not isinstance(obj, dict):
        return default

    out: dict = {}
    for name, question in FACETS.items():
        raw = obj.get(name)
        extra = ([q.strip() for q in raw
                  if isinstance(q, str) and q.strip()][:3]
                 if isinstance(raw, list) else [])
        out[name] = [question] + extra
    return out


def _facet_source(chunks: list, order: list) -> str:
    """Where the passages this facet was read from came from.

    Recorded per facet, not per run: one corpus can hold the article and the
    pages that filled the gaps in it, and a reader looking at a Grammy count
    deserves to know which of the two said so."""
    kinds = {getattr(chunks[i], "source", "web") for i in order
             if 0 <= i < len(chunks)}
    return "wiki" if kinds == {"wikipedia"} else "web"


async def read_facets(ask, artist: str, chunks: list, retriever, sents: tuple, *,
                      lang_name: str = "Russian",
                      corpus_lang: str = "English") -> dict:
    """Extract the facts shown beside the bio. Returns column → value.

    No search happens here any more. A facet the article does not answer is
    asked again OF THE SAME CORPUS in other words — see
    ``retrieval.facet_chunks_hybrid`` for why that replaced one web search per
    empty facet.
    """
    queries = await facet_queries(ask, artist, corpus_lang)
    out: dict = {}
    for name, question in FACETS.items():
        asked = queries.get(name) or [question]
        order = await asyncio.to_thread(
            R.facet_chunks_hybrid, retriever, artist, asked)
        if not order:
            # The sentence pass. A facet answer is one sentence inside a chunk
            # about something else, and at chunk granularity M83's "They decided
            # to name their band M83, after the galaxy of that name" never
            # surfaced at all — no fusion of paraphrases fixes that, because the
            # unit being ranked is wrong.
            order, _best = await asyncio.to_thread(
                R.facet_chunks, artist, question, sents)
        if not order:
            continue
        obj = parse_json(await ask(P.FACET_PROMPTS[name].format(
            artist=artist, lang=lang_name,
            passages=R.passages(chunks, order)), 0.2))
        if _empty(obj):
            continue
        out.update(_columns(name, obj, _facet_source(chunks, order)))
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
                searcher=None, seed_bio: Optional[str] = None,
                proxies: Optional[dict] = None) -> dict:
    """Everything for one artist: bio text, facets, and what was rejected.

    Budget: two requests to Wikipedia, and one to the open web — the last spent
    only when Wikipedia produced no biography, whether because there is no
    article or because nothing in the one there is cleared the chunk gate.
    """
    cfg = config or AgentConfig()
    fetcher = fetcher or PageFetcher(cfg)
    if searcher is None:
        # Imported HERE so the pipeline can be exercised — by a test, by a probe
        # — without dragging in the assistant's search stack. ONE instance for
        # the whole run, which is what makes ``max_web_searches`` mean anything:
        # the budget and the repeat-query refusal both live on the instance, and
        # the previous caller built a fresh one per call, so neither ever fired.
        from app.services.assistant.web_sources import SearchSources

        searcher = SearchSources(replace(cfg, max_web_searches=WEB_BUDGET))

    # Всё, что ниже уходит в поток НЕ ради скорости, а ради отзывчивости: эта
    # корутина живёт на главном event loop (задача запускается через
    # asyncio.create_task), и любая синхронная ступень останавливает весь сервис.
    chunks, meta = await sources.from_wikipedia(
        artist, cfg=cfg, fetcher=fetcher, proxies=proxies)

    # Самая долгая ступень: dense + sparse кодирование кусков. На проде держала
    # loop по 18 секунд — в логе ровно столько тишины, а следом залп из
    # накопившихся ответов.
    retriever = await asyncio.to_thread(R.build_index, chunks) if chunks else None

    bio, notes = "", {}
    if chunks:
        bio, notes = await write_bio(ask, artist, chunks, retriever,
                                     lang_name=lang_name, lang_code=lang_code)

    if not bio:
        fresh, web_meta = await sources.from_web(
            artist, cfg=cfg, fetcher=fetcher, searcher=searcher,
            seed_bio=seed_bio)
        meta["web"] = web_meta
        if fresh:
            if retriever is None:
                chunks = fresh
                retriever = await asyncio.to_thread(R.build_index, chunks)
            else:
                # ``extend`` encodes only the new documents and appends them in
                # order, so chunk index and document index stay the same number
                # — which is what every ``R.passages(chunks, order)`` assumes.
                await asyncio.to_thread(retriever.extend,
                                        [c.text for c in fresh])
                chunks = chunks + fresh
            notes["widened"] = len(fresh)
            bio, more = await write_bio(ask, artist, chunks, retriever,
                                        lang_name=lang_name,
                                        lang_code=lang_code)
            notes.update(more)

    if not chunks or not bio:
        # No biography means the caller stores nothing, and the facets are
        # stored WITH the biography — reading them here would be five LLM calls
        # whose answers have nowhere to go.
        return {"bio": bio, "facets": {},
                "error": meta.get("error") or "no bio written",
                "rejected": meta.get("rejected") or [], "notes": notes}

    facets = await read_facets(
        ask, artist, chunks, retriever,
        await asyncio.to_thread(R.sentence_index, chunks),
        lang_name=lang_name, corpus_lang=_corpus_lang(artist, meta))

    web_meta = meta.get("web") or {}
    facets.update({
        "source_url": meta.get("source_url") or web_meta.get("source_url"),
        "source_kind": meta.get("source_kind") or web_meta.get("source_kind")
                       or "web",
    })
    return {"bio": bio, "facets": facets, "article": meta.get("article"),
            "rejected": meta.get("rejected") or [], "notes": notes,
            "error": None if bio else (meta.get("error") or "no bio written")}
