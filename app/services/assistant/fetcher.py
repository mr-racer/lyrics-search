"""Downloading pages: caching, concurrency, and never raising.

``resources/web_fetch`` is the engine — a cascade of fetchers and a table-aware
extractor. This adds the three things a pipeline needs and a bare function does
not: a cache keyed by canonical URL, bounded concurrency with a hard deadline,
and the promise that one dead URL costs one page rather than the run.

Never raising matters more than it sounds. ``fetch_html`` raises when every
fetcher refuses, which is correct for a caller that wants to know — and wrong
here, where the correct response is to read the next candidate instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional

from app.resources import mediawiki, reddit_feed, web_fetch
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Page
from app.services.assistant.web_urls import canonical_url, source_for_url

logger = logging.getLogger(__name__)


class PageFetcher:
    """Fetch and extract, at most once per PAGE per instance.

    Per page, not per URL string: the cache is keyed by
    :func:`~web_urls.canonical_url`, so the same Wikipedia article arriving
    percent-encoded from one search stream and decoded from another is downloaded
    once.
    """

    def __init__(self, config: Optional[AgentConfig] = None, sink=None,
                 shared=None):
        self.cfg = config or AgentConfig()
        self.sink = sink
        self._cache: dict = {}
        # The cross-turn layer. This instance dies with the message; that one
        # outlives it by a minute, which is what makes a tapped follow-up cheap.
        # Injectable so a test can run without it.
        from app.services.assistant.page_store import PAGES

        self._shared = PAGES if shared is None else shared
        self._semaphore = asyncio.Semaphore(self.cfg.fetch_concurrency)
        # Resolved once per run, not per page: it reads a file off disk.
        from app.services.proxy_config import get_proxy

        self._proxies = get_proxy()

    @property
    def seen_urls(self) -> set:
        return set(self._cache)

    def _emit(self, stage: str, **fields) -> None:
        if self.sink is not None:
            self.sink.put(stage, **fields)

    # ── one page ──────────────────────────────────────────────────────────

    def fetch_sync(self, url: str, *, source: str = "web",
                   title: str = "") -> Page:
        """Blocking fetch. Returns a Page with ``error`` set on failure."""
        key = canonical_url(url)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        shared = self._shared_get(url)
        if shared is not None:
            # Read by another turn — or another listener — inside the TTL. Kept
            # in the local cache too so the rest of this run treats it as read.
            self._cache[key] = shared
            self._emit("fetch_cached", url=url)
            return shared

        keep_html = source_for_url(url) == "apple"
        raw_html = ""

        def by_scraping():
            nonlocal raw_html
            html, fetcher = web_fetch.fetch_html(
                url, timeout=self.cfg.fetch_timeout, proxies=self._proxies)
            if keep_html:
                # One request, two consumers. Apple's track list is in embedded
                # JSON that markdown extraction discards, so the body is kept for
                # the tracklist parser instead of being downloaded again.
                raw_html = html
            out = web_fetch.extract_page(html, url)
            return out["text"], out["meta"], fetcher

        def by_api():
            got = mediawiki.fetch_html(url, timeout=self.cfg.fetch_timeout,
                                       proxies=self._proxies)
            if got is None:
                return "", {}, None
            html, api_meta = got
            return web_fetch.extract_page(html, url)["text"], api_meta, "mediawiki_api"

        def by_feed():
            got = reddit_feed.fetch_thread(url, timeout=self.cfg.fetch_timeout,
                                           cooldown=self.cfg.reddit_cooldown,
                                           proxies=self._proxies)
            if got is None:
                return "", {}, None
            feed_title, markdown = got
            return markdown, {"title": feed_title}, "reddit_rss"

        # Order matters and differs by host. Fandom sits behind a Cloudflare
        # interstitial that answers 403 to every HTTP fetcher, so there the API is
        # the only way in. On Wikipedia the ordinary fetch wins: the API returns
        # the whole raw article (120k chars on a discography page) where
        # extraction returns half that, cleaner and faster. Reddit is the third
        # case and the strictest: scraping the page does not fail, it SUCCEEDS
        # with a challenge page, so the feed has to be tried first rather than
        # kept as a fallback.
        if source_for_url(url) == "reddit":
            first, second = by_feed, by_scraping
        elif mediawiki.prefers_api(url):
            first, second = by_api, by_scraping
        else:
            first, second = by_scraping, by_api

        try:
            markdown, meta, fetcher = "", {}, None
            refusals: list = []
            for attempt in (first, second):
                try:
                    markdown, meta, fetcher = attempt()
                except Exception as exc:  # noqa: BLE001 — try the other route
                    refusals.append(f"{type(exc).__name__}: {exc}")
                    logger.info("[fetch] %s: %s failed (%s), trying the other route",
                                url, attempt.__name__, type(exc).__name__)
                    markdown = ""
                # Raw html counts as success: an Apple page can be entirely worth
                # having with no extractable prose at all.
                if (markdown or "").strip() or raw_html:
                    break

            # A challenge page is a refusal wearing a 200, so it has to be spotted
            # before the Page is built — every signal the fetcher normally trusts
            # says this one succeeded.
            walled = web_fetch.looks_like_a_bot_wall(meta.get("title") or "")
            if walled:
                refusals.append(f"bot wall: {meta.get('title')}")
                markdown, meta, fetcher = "", {}, None

            if self.cfg.strip_appendix:
                markdown, removed = web_fetch.strip_appendix(markdown or "")
                if removed:
                    logger.info("[fetch] %s: dropped %d chars of appendix "
                                "(references, external links)", url, removed)
            page = Page(url=url, title=(title or meta.get("title") or ""),
                        markdown=markdown or "", source=source, meta=meta,
                        fetcher=fetcher, html=raw_html)
            if not page.ok:
                # ``walled`` was decided above, on the page's OWN title rather
                # than ``page.title`` — that one prefers the search hit's wording,
                # which still reads like the page we wanted and would hide the
                # challenge completely.
                page.error = ("served a bot wall" if walled
                              else "extractor returned nothing")
                if refusals:
                    logger.info("[fetch] %s: every route refused (%s)", url,
                                "; ".join(refusals)[:200])
        except Exception as exc:  # noqa: BLE001 — a parse blew up
            page = Page(url=url, title=title, markdown="", source=source,
                        error=f"{type(exc).__name__}: {exc}")
            logger.info("[fetch] %s failed: %s", url, page.error)

        self._cache[key] = page
        # Only successes reach the shared layer — PageStore.put drops the rest.
        # A 403 is a property of the moment, not of the URL, and caching one
        # would poison every turn for a full minute over one unlucky request.
        self._shared_put(page)
        return page

    def _shared_get(self, url: str) -> Optional[Page]:
        if self._shared is None:
            return None
        try:
            return self._shared.get(url)
        except Exception:  # noqa: BLE001 — a cache must never break a fetch
            logger.warning("[fetch] shared page store unreadable", exc_info=True)
            return None

    def _shared_put(self, page: Page) -> None:
        if self._shared is None:
            return
        try:
            self._shared.put(page)
        except Exception:  # noqa: BLE001
            logger.warning("[fetch] shared page store unwritable", exc_info=True)

    async def fetch(self, url: str, *, source: str = "web",
                    title: str = "") -> Page:
        key = canonical_url(url)
        if key in self._cache:
            return self._cache[key]
        shared = self._shared_get(url)
        if shared is not None:
            self._cache[key] = shared
            self._emit("fetch_cached", url=url)
            return shared
        async with self._semaphore:
            try:
                # A hard ceiling over the whole cascade. Each fetcher has its own
                # timeout, but three of them in a row is three times the wait, and
                # a library that ignores the setting (or grows a new retry) would
                # otherwise stall the run with nothing in the log.
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
                # thread runs on and may still land a real page in this slot; that
                # is an improvement, not a race.
                self._cache.setdefault(key, page)
                return page

    # ── many pages ────────────────────────────────────────────────────────

    async def fetch_many(self, hits: Iterable, *,
                         limit: Optional[int] = None) -> list:
        """Fetch until ``limit`` pages have actually been READ, not attempted.

        Order follows the input, not completion: the caller ranked those hits for
        a reason, and downstream chunk ids should be stable across runs.

        ``limit`` counts successes. A failed fetch pulls in the next candidate the
        ranking had already approved, in waves, until the pool runs out or
        ``fetch_refill_attempts`` is spent. This is the difference between "the
        cross-encoder liked eight pages, three of which are unreachable, so the
        iteration gets two" and "…so the iteration gets five".

        The batch is deduplicated against ITSELF as well as against what has
        already been read, and both matter. The second one is easy to miss: two
        search streams routinely return the same article, so without it a single
        Wikipedia page can take three of the five fetch slots and then land in the
        retriever three times, where the copies inflate the document frequencies
        BM25 runs on and crowd the top-k with one paragraph repeated.
        """
        queue: list = []
        seen: set = set()
        for hit in hits:
            key = canonical_url(hit.url)
            if key in self._cache or key in seen:
                continue
            seen.add(key)
            queue.append(hit)
        if not queue:
            return []

        want = limit or len(queue)
        # Only a capped run refills: with no limit the caller asked for the whole
        # queue and there is nothing left to fall back on anyway.
        refills_left = self.cfg.fetch_refill_attempts if limit else 0
        ok: list = []
        failed = 0
        cursor = wave_no = 0

        while cursor < len(queue) and len(ok) < want:
            need = want - len(ok)
            if wave_no:
                need = min(need, refills_left)
                if need <= 0:
                    logger.info("[fetch] %d page(s) short, but the refill budget "
                                "(%d) is spent", want - len(ok),
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

        self._emit("fetch_done", fetched=len(ok), failed=failed, waves=wave_no,
                   unread=len(queue) - cursor)
        return ok
