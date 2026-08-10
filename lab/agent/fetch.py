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
from lab.agent.urls import canonical_url, source_for_url

logger = logging.getLogger(__name__)


def _pin_trafilatura_timeout(seconds: float) -> None:
    """Make our timeout actually reach trafilatura's downloader.

    ``websearch_lab._fetch_trafilatura`` takes a ``timeout`` argument and does
    not use it — it calls ``trafilatura.fetch_url(url)``, which falls back to
    the library's own ``DOWNLOAD_TIMEOUT`` of 30 seconds. So a dead host cost
    30 seconds per page no matter what the config said, and the log said
    ``connect timeout=30`` while the config said 15.

    Setting it on the module-level config is what ``fetch_url`` reads when it
    is called without one. Done once, from here, so the lab file stays the
    author's.
    """
    try:
        from trafilatura.settings import DEFAULT_CONFIG

        DEFAULT_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(int(seconds)))
    except Exception:  # noqa: BLE001 — no trafilatura, or it moved the setting
        logger.debug("[fetch] could not pin trafilatura's timeout", exc_info=True)


class PageFetcher:
    """Fetch and extract, at most once per PAGE per instance.

    Per page, not per URL string: the cache is keyed by
    :func:`~lab.agent.urls.canonical_url`, so the same Wikipedia article
    arriving percent-encoded from one search stream and decoded from another is
    downloaded once.
    """

    def __init__(self, config=None, sink=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.sink = sink
        self._cache: dict[str, Page] = {}
        self._semaphore = asyncio.Semaphore(self.cfg.fetch_concurrency)
        _pin_trafilatura_timeout(self.cfg.fetch_timeout)

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
        key = canonical_url(url)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from lab import websearch_lab as L
        from lab.agent.cleanup import strip_appendix
        from lab.agent import mediawiki

        keep_html = source_for_url(url) == "apple"
        raw_html = ""

        def by_scraping():
            nonlocal raw_html
            # http_get_text + extract_page rather than md(): md() takes no
            # timeout, and the whole point here is that ours applies.
            html = L.http_get_text(url, timeout=self.cfg.fetch_timeout)
            if keep_html:
                # One request, two consumers. Apple's track list is in embedded
                # JSON that markdown extraction discards, so the body is kept
                # for `apple_tracks` instead of being downloaded again there.
                raw_html = html
            out = L.extract_page(html, url, "trafilatura")
            return out["text"], out["meta"], L.LAST_FETCH.get("fetcher")

        def by_api():
            got = mediawiki.fetch_html(url, timeout=self.cfg.fetch_timeout)
            if got is None:
                return "", {}, None
            html, api_meta = got
            return L.extract_page(html, url, "trafilatura")["text"], \
                api_meta, "mediawiki_api"

        # Order matters and differs by host. Fandom sits behind a Cloudflare
        # interstitial that answers 403 to every HTTP fetcher, so there the API
        # is the only way in. On Wikipedia the ordinary fetch wins: the API
        # returns the whole raw article (120k chars on a discography page)
        # where trafilatura extracts half that, cleaner and faster.
        first, second = ((by_api, by_scraping)
                         if mediawiki.prefers_api(url, self.cfg.mediawiki_api_first)
                         else (by_scraping, by_api))

        try:
            markdown, meta, fetcher = "", {}, None
            for attempt in (first, second):
                try:
                    markdown, meta, fetcher = attempt()
                except Exception as exc:  # noqa: BLE001 — try the other route
                    logger.info("[fetch] %s: %s failed (%s), trying the other route",
                                url, attempt.__name__, type(exc).__name__)
                    markdown = ""
                # Raw html counts as success: an Apple page can be entirely
                # worth having with no extractable prose at all.
                if (markdown or "").strip() or raw_html:
                    break
            if self.cfg.strip_appendix:
                markdown, removed = strip_appendix(markdown or "")
                if removed:
                    logger.info("[fetch] %s: dropped %d chars of appendix "
                                "(references, external links)", url, removed)
            page = Page(url=url, title=(title or meta.get("title") or ""),
                        markdown=markdown or "", source=source, meta=meta,
                        fetcher=fetcher, html=raw_html)
            if not page.ok:
                page.error = "extractor returned nothing"
        except Exception as exc:  # noqa: BLE001 — every fetcher refused, or a parse blew up
            page = Page(url=url, title=title, markdown="", source=source,
                        error=f"{type(exc).__name__}: {exc}")
            logger.info("[fetch] %s failed: %s", url, page.error)

        self._cache[key] = page
        return page

    async def fetch(self, url: str, *, source: SourceKind = "web",
                    title: str = "") -> Page:
        key = canonical_url(url)
        if key in self._cache:
            return self._cache[key]
        async with self._semaphore:
            try:
                # A hard ceiling over the whole cascade. Each fetcher has its
                # own timeout, but three of them in a row is three times the
                # wait, and a library that ignores the setting (or grows a new
                # retry) would otherwise stall the run with nothing in the log.
                return await asyncio.wait_for(
                    asyncio.to_thread(self.fetch_sync, url, source=source,
                                      title=title),
                    timeout=self.cfg.fetch_deadline)
            except asyncio.TimeoutError:
                logger.info("[fetch] %s gave up after %.0fs", url,
                            self.cfg.fetch_deadline)
                page = Page(url=url, title=title, markdown="", source=source,
                            error=f"deadline of {self.cfg.fetch_deadline:.0f}s")
                # Remembered like any other failure, so no later batch — and no
                # refill — spends the deadline on it a second time. The worker
                # thread runs on and may still land a real page in this slot;
                # that is an improvement, not a race.
                self._cache.setdefault(key, page)
                return page

    # ── many pages ────────────────────────────────────────────────────────

    async def fetch_many(self, hits: Iterable, *,
                         limit: Optional[int] = None) -> list[Page]:
        """Fetch until ``limit`` pages have actually been READ, not attempted.

        Order follows the input, not completion: the caller ranked those hits
        for a reason, and downstream chunk ids should be stable across runs.

        ``limit`` counts successes. A failed fetch pulls in the next candidate
        the ranking had already approved, in waves, until the pool runs out or
        ``fetch_refill_attempts`` is spent. This is the difference between "the
        cross-encoder liked eight pages, three of which are unreachable, so the
        iteration gets two" and "…so the iteration gets five". Refetching the
        same dead host is not among the outcomes — a failure is cached like any
        other result.

        The batch is deduplicated against ITSELF as well as against what has
        already been read, and both matter. The second one is easy to miss: two
        search streams routinely return the same article, so without it a
        single Wikipedia page can take three of the five fetch slots and then
        land in the retriever three times, where the copies inflate the
        document frequencies BM25 runs on and crowd the top-k with one
        paragraph repeated.
        """
        queue: list = []
        seen: set[str] = set()
        for hit in hits:
            key = canonical_url(hit.url)
            if key in self._cache or key in seen:
                continue
            seen.add(key)
            queue.append(hit)
        if not queue:
            return []

        want = limit or len(queue)
        # Only a capped run refills: with no limit the caller asked for the
        # whole queue and there is nothing left to fall back on anyway.
        refills_left = self.cfg.fetch_refill_attempts if limit else 0
        ok: list[Page] = []
        failed = 0
        cursor = wave_no = 0

        while cursor < len(queue) and len(ok) < want:
            need = want - len(ok)
            if wave_no:
                need = min(need, refills_left)
                if need <= 0:
                    logger.info("[fetch] %d page(s) short, but the refill "
                                "budget (%d) is spent", want - len(ok),
                                self.cfg.fetch_refill_attempts)
                    break
                refills_left -= need
            wave = queue[cursor:cursor + need]
            cursor += len(wave)
            wave_no += 1

            self._emit("fetch", count=len(wave), urls=[h.url for h in wave],
                       refill=wave_no > 1)
            pages = await asyncio.gather(*[
                self.fetch(h.url, source=h.source, title=h.title) for h in wave
            ])
            good = [p for p in pages if p.ok]
            ok.extend(good)
            for page in pages:
                if not page.ok:
                    failed += 1
                    logger.info("[fetch] %s unusable: %s", page.url,
                                page.error or "no content")
            if len(good) < len(wave) and cursor < len(queue) and len(ok) < want:
                logger.info("[fetch] %d of %d failed — taking the next "
                            "candidate(s) from the %d still ranked below",
                            len(wave) - len(good), len(wave), len(queue) - cursor)

        self._emit("fetch_done", fetched=len(ok), failed=failed,
                   waves=wave_no, unread=len(queue) - cursor)
        return ok
