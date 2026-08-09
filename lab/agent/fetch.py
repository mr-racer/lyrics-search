"""Downloading pages, through the fetcher cascade that already works.

``websearch_lab.md()`` is the whole engine: it tries trafilatura, then bare
curl, then a full Chrome fingerprint (the only one Fandom answers), and runs
the result through the table-aware markdown extractor. Nothing here
reimplements any of that — this module adds the three things a pipeline needs
and a notebook helper does not: caching, concurrency, and never raising.

Never raising matters more than it sounds. One dead URL out of five must cost
one page, not the run; ``md()`` raises ``RuntimeError`` when every fetcher
refuses, which is correct for interactive use and wrong here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional

from lab.agent.models import Page, SourceKind

logger = logging.getLogger(__name__)


class PageFetcher:
    """Fetch and extract, at most once per URL per instance."""

    def __init__(self, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.sink = sink
        self._cache: dict[str, Page] = {}
        self._semaphore = asyncio.Semaphore(self.cfg.fetch_concurrency)

    @property
    def seen_urls(self) -> set[str]:
        return set(self._cache)

    def _emit(self, stage: str, **fields) -> None:
        if self.sink is not None:
            self.sink.put(stage, **fields)

    # ── one page ──────────────────────────────────────────────────────────

    def fetch_sync(self, url: str, *, source: SourceKind = "web",
                   title: str = "") -> Page:
        """Blocking fetch. Returns a Page with ``error`` set on failure."""
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        from lab import websearch_lab as L

        try:
            markdown = L.md(url, quiet=True)
            meta = dict(L.LAST_EXTRACT.get("meta") or {})
            page = Page(url=url, title=(title or meta.get("title") or ""),
                        markdown=markdown or "", source=source, meta=meta,
                        fetcher=L.LAST_EXTRACT.get("fetcher"))
            if not page.markdown.strip():
                page.error = "extractor returned nothing"
        except Exception as exc:  # noqa: BLE001 — every fetcher refused, or a parse blew up
            page = Page(url=url, title=title, markdown="", source=source,
                        error=f"{type(exc).__name__}: {exc}")
            logger.info("[fetch] %s failed: %s", url, page.error)

        self._cache[url] = page
        return page

    async def fetch(self, url: str, *, source: SourceKind = "web",
                    title: str = "") -> Page:
        if url in self._cache:
            return self._cache[url]
        async with self._semaphore:
            return await asyncio.to_thread(self.fetch_sync, url,
                                           source=source, title=title)

    # ── many pages ────────────────────────────────────────────────────────

    async def fetch_many(self, hits: Iterable, *,
                         limit: Optional[int] = None) -> list[Page]:
        """Fetch a batch of :class:`~lab.agent.models.SearchHit` concurrently.

        Order follows the input, not completion: the caller ranked those hits
        for a reason, and downstream chunk ids should be stable across runs.
        Already-fetched URLs are skipped entirely — an iteration that re-finds
        a page it read last time must not pay for it twice.
        """
        selected = []
        for hit in hits:
            if hit.url in self._cache:
                continue
            selected.append(hit)
            if limit and len(selected) >= limit:
                break
        if not selected:
            return []

        self._emit("fetch", count=len(selected),
                   urls=[h.url for h in selected])
        pages = await asyncio.gather(*[
            self.fetch(h.url, source=h.source, title=h.title) for h in selected
        ])
        ok = [p for p in pages if p.ok]
        self._emit("fetch_done", fetched=len(ok), failed=len(pages) - len(ok))
        return ok
