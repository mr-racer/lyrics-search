"""What the two web branches share: searching, fetching, chunking, ranking.

The context pack is built here, and the two rules it enforces are the reason
this is one place rather than two:

* **the threshold decides membership, not a count.** ``ce_threshold_chunks`` is
  the only thing that says a passage may be shown, and when nothing clears it the
  pack is empty — which the branch reads as "search again", never as "answer with
  what is left";
* **near-duplicates do not spend slots.** The pool is pulled deeper than the pack
  so that dropping a copy FREES a slot rather than shrinking the pack.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from app.services.assistant.contracts import Chunk, Page, Plan
from app.services.assistant.web_sources import rerank_hits
from app.services.assistant.web_urls import canonical_url, dedupe_by_url
from app.services.retrieval import HybridRetriever, pick_diverse

logger = logging.getLogger(__name__)


class WebBranch:
    """Search → fetch → chunk → rank, accumulated across iterations."""

    def __init__(self, agent, sources, fetcher):
        self.agent = agent
        self.cfg = agent.cfg
        self.sink = agent.sink
        self.timings = agent.timings
        self.sources = sources
        self.fetcher = fetcher
        self.chunks: list = []
        self.retriever: Optional[HybridRetriever] = None
        self.used_queries: list = []
        # Content hashes of everything indexed. URL deduplication catches the same
        # page twice; this catches the same TEXT arriving from two pages, which
        # listicles syndicating each other do constantly.
        self._chunk_hashes: set = set()

    async def gather(self, plan: Plan, queries: list, ce_query: str, *,
                     structured: bool) -> tuple:
        """Run one round of search + fetch.

        Returns ``(structured_pages, prose_pages)``. Structured sources skip the
        relevance pass: pinning the host has already done that work, and an Apple
        playlist page has almost no text for a reranker to read.
        """
        structured_hits, web_hits = [], []
        for i, query in enumerate(queries):
            self.used_queries.append(query)
            # Host-pinned sources on the first query only by default: a rephrasing
            # rarely finds a different Apple playlist, and every extra call is
            # another turn of the burst that gets the good engines rate-limited.
            first_only = self.cfg.structured_first_query_only
            if structured and (i == 0 or not first_only):
                with self.timings.span("search.apple"):
                    structured_hits += await asyncio.to_thread(
                        self.sources.apple_music, query)
                if plan.filters.work:
                    with self.timings.span("search.fandom"):
                        structured_hits += await asyncio.to_thread(
                            self.sources.fandom, query)
            with self.timings.span("search.wikipedia"):
                structured_hits += await asyncio.to_thread(
                    self.sources.wikipedia, query)

        # Start reading them NOW, with the web searches still to come. Nothing
        # about which structured pages get read depends on those searches — these
        # skip the relevance pass entirely — so the only thing waiting bought was
        # wall clock. And the wait is long: SearXNG is paced by a blocking sleep
        # inside a worker thread, so for most of the search phase the loop has
        # nothing to do at all.
        #
        # The list is deduplicated first because the fetch budget is spent on this
        # exact order, and ``fetch_many`` must see the same queue it would have
        # seen at the end.
        structured_hits = dedupe_by_url(structured_hits)
        reading = asyncio.create_task(self.fetcher.fetch_many(
            structured_hits, limit=self.cfg.max_pages_per_iteration))

        try:
            for query in queries:
                with self.timings.span("search.web"):
                    web_hits += await asyncio.to_thread(self.sources.web, query)

            kept: list = []
            if web_hits:
                with self.timings.span("rerank.ce"):
                    kept = await asyncio.to_thread(
                        rerank_hits, web_hits, ce_query, hub=self.agent.hub,
                        threshold=self.cfg.ce_threshold_docs)
            self.sink.put("rerank", candidates=len(web_hits), kept=len(kept),
                          threshold=self.cfg.ce_threshold_docs)
        except BaseException:
            # A search that raised must not leave a fetch running behind it,
            # spending the next iteration's concurrency on pages nobody will read.
            reading.cancel()
            raise

        # Deduplicated ACROSS the two lists, not within each: a Wikipedia article
        # reached by the pinned-host stream and again by the open-web stream is one
        # page. Structured goes first so the shared page keeps its structured
        # treatment (tables parsed, not chunked as prose).
        structured_keys = {canonical_url(h.url) for h in structured_hits}
        prose_hits = [h for h in dedupe_by_url(kept)
                      if canonical_url(h.url) not in structured_keys]

        # What this span measures has changed with the overlap: the structured
        # read is mostly already done by the time it is awaited, so ``fetch`` no
        # longer accounts for the time it cost — and the lanes in the timing report
        # no longer sum to the wall clock. That is the saving showing up, not a bug
        # in the report.
        with self.timings.span("fetch"):
            structured_pages = await reading
            prose_pages = await self.fetcher.fetch_many(
                prose_hits, limit=self.cfg.max_pages_per_iteration)
        return structured_pages, prose_pages

    def seed(self, chunks: list, used_queries: Optional[list] = None) -> int:
        """Adopt a previous turn's chunks instead of downloading them again.

        The pages behind these were read a minute ago and the passages were cut
        out of them then; only the ranking is per-question, and the ranking is
        the cheap half. What this skips is the search pacing and the fetches,
        which are the slow half by an order of magnitude.

        ``used_queries`` comes with them so the next-queries call does not
        re-issue searches that context already spent.

        Ids are reassigned: they index into ``self.chunks`` and the previous
        turn's numbering means nothing here.
        """
        if not chunks:
            return 0
        fresh: list = []
        for chunk in chunks:
            digest = _text_hash(chunk.body)
            if digest in self._chunk_hashes:
                continue
            self._chunk_hashes.add(digest)
            chunk.id = len(self.chunks) + len(fresh)
            fresh.append(chunk)
        if not fresh:
            return 0
        with self.timings.span("index.embed"):
            texts = [c.text for c in fresh]
            if self.retriever is None:
                self.retriever = HybridRetriever(texts, hub=self.agent.hub)
            else:
                self.retriever.extend(texts)
        self.chunks.extend(fresh)
        for query in used_queries or ():
            if query not in self.used_queries:
                self.used_queries.append(query)
        logger.info("[branch] seeded %d chunks from a previous turn", len(fresh))
        self.sink.put("seeded", chunks=len(fresh),
                      queries=len(used_queries or []))
        return len(fresh)

    def index(self, pages: list) -> int:
        """Chunk ``pages`` and add them to the retriever. Returns new chunks."""
        with self.timings.span("index.embed"):
            return self._index(pages)

    def _index(self, pages: list) -> int:
        candidates = self.agent.chunks_of(pages, len(self.chunks))
        fresh: list = []
        for chunk in candidates:
            digest = _text_hash(chunk.body)
            if digest in self._chunk_hashes:
                continue
            self._chunk_hashes.add(digest)
            chunk.id = len(self.chunks) + len(fresh)
            fresh.append(chunk)
        duplicates = len(candidates) - len(fresh)
        if duplicates:
            logger.info("[branch] dropped %d duplicate chunks", duplicates)
        if not fresh:
            return 0
        texts = [c.text for c in fresh]
        if self.retriever is None:
            self.retriever = HybridRetriever(texts, hub=self.agent.hub)
        else:
            self.retriever.extend(texts)
        self.chunks.extend(fresh)
        self.sink.put("index", new_chunks=len(fresh), total=len(self.chunks))
        return len(fresh)

    def best_chunks(self, ce_query: str) -> list:
        """Top chunks across every page read so far, above the threshold."""
        with self.timings.span("select.pack"):
            return select_pack(self.retriever, self.chunks, ce_query,
                               config=self.cfg, sink=self.sink)


# ── the context pack ─────────────────────────────────────────────────────────


def select_pack(retriever, chunks: list, ce_query: str, *, config,
                sink=None) -> list:
    """The passages that go in front of the model, and their probabilities."""
    if retriever is None or not chunks:
        return []

    threshold = config.ce_threshold_chunks
    limit = config.max_chunks_in_context
    # A deeper pool than the pack needs, so that dropping a copy FREES a slot
    # rather than shrinking the pack: the next distinct passage moves up into it.
    # With dedup off this is exactly the old behaviour.
    pool = limit * max(1, config.dedup_pool_factor) if config.dedup_chunks else limit
    ranked = retriever.search(ce_query, min_prob=threshold, limit=pool,
                              alpha=config.ce_alpha,
                              weights=config.fusion_weights)
    ranked = _diverse(retriever, chunks, ranked, limit=limit, cfg=config, sink=sink)
    out = [(chunks[r.index], r.ce_prob or 0.0) for r in ranked]

    if not out:
        # A high chunk threshold is the right default, but "nothing passed" and
        # "nothing was found" look identical from the outside. Say which, and say
        # how close it came, so the number can be calibrated instead of guessed at.
        unfiltered = retriever.search(ce_query, limit=1, alpha=config.ce_alpha,
                                      weights=config.fusion_weights)
        best = unfiltered[0].ce_prob if unfiltered else None
        logger.info("[branch] no chunk cleared p>=%.2f over %d chunks (best was %s)",
                    threshold, len(chunks),
                    f"{best:.3f}" if best is not None else "unscored")
        if sink is not None:
            sink.put("chunks", selected=0, threshold=threshold,
                     best=round(best, 3) if best is not None else None)
        return []
    if sink is not None:
        sink.put("chunks", selected=len(out), threshold=threshold,
                 best=round(max(p for _, p in out), 3))
    return out


def _diverse(retriever, chunks: list, ranked: list, *, limit: int, cfg,
             sink=None) -> list:
    """Take the top ``limit`` DISTINCT passages out of the ranked pool.

    Five hosts carrying the same syndicated bio produce five chunks that score
    alike, and the pack ends up saying one thing five times. This spends those
    slots on the next-best passage that says something else. Nothing is removed
    from the index — see ``services/retrieval/diversity`` for why that distinction
    is the safety property here.
    """
    if not cfg.dedup_chunks or len(ranked) <= 1:
        return ranked[:limit]

    sims = retriever.similarity_matrix([r.index for r in ranked])
    if not sims:
        logger.info("[branch] no document vectors to compare — keeping the top %d "
                    "as ranked", limit)
        return ranked[:limit]
    missing = [n for n in cfg.dedup_thresholds if n not in sims]
    if missing:
        # Worth saying out loud: the two-signal agreement rule is what keeps this
        # from collapsing passages that merely share a topic, and with one signal
        # it is not being applied.
        logger.info("[branch] duplicate check running on %s alone (%s "
                    "unavailable) — a weaker guard than intended",
                    ", ".join(sorted(sims)), ", ".join(missing))

    picked = pick_diverse([len(chunks[r.index].text) for r in ranked], sims,
                          thresholds=cfg.dedup_thresholds, limit=limit,
                          prefer_longer=cfg.dedup_prefer_longer)
    for dup in picked.duplicates:
        scores = " ".join(f"{n}={v:.3f}" for n, v in sorted(dup.sims.items()))
        logger.info("[branch] %s chunk %d (%s) — same as chunk %d (%s) [%s]",
                    "displaced" if dup.replaced else "skipped",
                    chunks[ranked[dup.index].index].id,
                    chunks[ranked[dup.index].index].url,
                    chunks[ranked[dup.twin].index].id,
                    chunks[ranked[dup.twin].index].url, scores)
    if picked.duplicates and sink is not None:
        sink.put("dedup", pool=len(ranked), selected=len(picked.kept),
                 duplicates=len(picked.duplicates), signals=sorted(sims))
    return [ranked[i] for i in picked.kept]


def _text_hash(text: str) -> str:
    """Content key for a chunk, insensitive to whitespace and case."""
    return hashlib.sha1(" ".join((text or "").lower().split()).encode("utf-8")
                        ).hexdigest()


def chunks_from(chunker, pages: list, start_id: int) -> list:
    out: list = []
    for page in pages:
        out.extend(chunker.split_page(page, start_id=start_id + len(out)))
    return out


__all__ = ["WebBranch", "select_pack", "chunks_from", "Chunk", "Page"]
