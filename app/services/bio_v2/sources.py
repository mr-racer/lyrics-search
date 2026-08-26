"""Where the passages come from — and how few requests that is allowed to cost.

Three requests per artist, and the shape of every one of them is the same:
ask an index, gate the candidates with the cross-encoder, download only what
survived, gate again on the body.

    WIKI  2 requests   batched title probe, then the relevance index
    WEB   1 request    the open web, only when Wikipedia yielded nothing usable

There used to be a second writer for the WEB case: an LLM agent with its own
SearXNG client, its own three searches, no cross-encoder and no junk filter.
It wrote 104 of the 580 production biographies, and none of them went through
any of the checks the Wikipedia path spent a release learning it needed. It is
gone. The web is a SOURCE now, not a second pipeline: what it returns is
chunked, gated and indexed exactly like an article, and everything downstream —
the five bio facets, the four fact facets, the script control — cannot tell the
two apart.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.resources.model_registry import ModelRegistry
from app.services.assistant.contracts import Page
from app.services.bio_v2 import article as art
from app.services.bio_v2 import retrieval as R

logger = logging.getLogger(__name__)

# How many gated results get downloaded. Reading is the expensive half — seconds
# per page against milliseconds for the search — so the cross-encoder picks
# first and only the survivors are fetched.
WEB_PAGES = 3

# The relevance question, asked once of a title+snippet and again of a body. The
# wording is shared with ``article.gate`` on purpose: the same judgement about
# the same artist should not shift with where the text came from.
RELEVANCE = ("{artist} is a musical artist or band: their music, albums and "
             "career.")


def _web_query(artist: str) -> str:
    return f"{artist} musician band biography"


async def from_wikipedia(artist: str, *, cfg, fetcher,
                         proxies: Optional[dict] = None) -> tuple:
    """(chunks, meta) — the article, or empty with the reason in ``meta``."""
    # Off the event loop rather than for speed: this coroutine runs on the main
    # loop and `find` is two HTTP round trips plus a cross-encoder pass.
    found, rejected = await asyncio.to_thread(art.find, artist, proxies=proxies)
    meta = {"rejected": rejected}
    if found is None:
        meta["error"] = "no wikipedia article passed the gate"
        return [], meta

    page = await fetcher.fetch(found["url"], source="wikipedia",
                               title=found["title"])
    if not page.ok or not page.markdown:
        meta["error"] = f"fetch failed: {page.error}"
        return [], meta

    chunks = await asyncio.to_thread(R.chunk_page, page, cfg)
    if not chunks:
        meta["error"] = "no chunks"
        return [], meta

    meta.update(source_kind="wikipedia", source_url=found["url"],
                article=found)
    return chunks, meta


async def from_web(artist: str, *, cfg, fetcher, searcher,
                   seed_bio: Optional[str] = None) -> tuple:
    """(chunks, meta) — one search, gated twice, plus the AudioDB seed.

    The seed rides along here and nowhere else. It is a paragraph somebody else
    wrote about this artist, sometimes promotional and sometimes years stale;
    next to a Wikipedia article it would only compete for slots, but next to
    nothing it is the difference between a biography and an empty page.
    """
    chunks: list = []
    meta: dict = {}

    if seed_bio and seed_bio.strip():
        seed_page = Page(url="", title=artist, markdown=seed_bio.strip(),
                         source="web")
        seed_chunks = await asyncio.to_thread(R.chunk_page, seed_page, cfg)
        for chunk in seed_chunks:
            chunk.title = chunk.title or artist
        chunks += seed_chunks
        meta["seed_chunks"] = len(seed_chunks)

    hits = []
    if searcher is not None:
        try:
            hits = await asyncio.to_thread(searcher.web, _web_query(artist))
        except Exception as exc:                        # noqa: BLE001
            logger.info("[bio.sources] web search failed for %s: %s: %s",
                        artist, type(exc).__name__, exc)
            hits = []

    picked = await _gate_hits(artist, hits or [])
    meta["web_hits"] = len(hits or [])
    meta["web_gated"] = len(picked)

    fresh: list = []
    for hit in picked[:WEB_PAGES]:
        page = await fetcher.fetch(hit.url, source="web", title=hit.title)
        if page.ok and page.markdown:
            fresh += await asyncio.to_thread(R.chunk_page, page, cfg)

    fresh = await _gate_chunks(artist, fresh)
    meta["web_chunks"] = len(fresh)
    chunks += fresh

    if chunks:
        meta["source_kind"] = "web" if fresh else "audiodb"
        meta["source_url"] = picked[0].url if (fresh and picked) else None
    return chunks, meta


async def _gate_hits(artist: str, hits: list) -> list:
    """Drop the results that are not about this artist, before downloading."""
    if not hits:
        return []
    docs = [f"{h.title}\n{h.snippet}".strip() for h in hits]
    probs = await asyncio.to_thread(
        ModelRegistry.ce_probabilities, RELEVANCE.format(artist=artist), docs)
    if probs is None:
        # No cross-encoder means no judgement, and an ungated web page is how a
        # New Zealand bedroom-pop musician was once credited with four Grammys
        # lifted from a page about somebody else. Read the top result only.
        return hits[:1]
    kept = [h for h, p in zip(hits, probs) if p >= art.CE_ARTICLE_GATE]
    logger.info("[bio.sources] web gate %s: %d/%d hits above %.2f",
                artist, len(kept), len(hits), art.CE_ARTICLE_GATE)
    return kept


async def _gate_chunks(artist: str, chunks: list) -> list:
    """The same judgement again on the body: a page can be about this artist in
    its title and about the site's navigation everywhere else."""
    if not chunks:
        return []
    probs = await asyncio.to_thread(
        ModelRegistry.ce_probabilities, RELEVANCE.format(artist=artist),
        [c.text[:1200] for c in chunks])
    if probs is None:
        return chunks
    return [c for c, p in zip(chunks, probs) if p >= art.CE_ARTICLE_GATE]
