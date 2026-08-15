"""The raw SearXNG call, paced and diagnosed.

This wraps the search instance and nothing else: one HTTP request, the whole
JSON body, and the rate limiting the instance needs to keep working. Every
policy decision on top of it — which engines, what counts as spam, when one host
has taken over the results — lives in the service that calls this.

**Why the whole body and not just ``results``.** ``unresponsive_engines`` is
where the diagnosis lives. Without it, "DuckDuckGo timed out and this answer was
assembled by three fallbacks" is invisible from inside and shows up only as "the
results got worse".

**Why the pacing.** One assistant iteration fires up to eight searches, each
fanning out to every engine in the whitelist — roughly 50 outbound requests from
one IP in a couple of seconds. DuckDuckGo and Brave rate-limit that immediately,
SearXNG reads the timeout as a failure and suspends the engine for up to
``max_ban_time_on_fail``, and the rest of the run gets served by whatever fringe
engines are left. That is the mechanism behind a Kanye query coming back with
Indonesian journal PDFs.

The gate is process-global and the wait is real (unlike Reddit's, which skips):
a search is on the critical path of every branch, so skipping it would mean
answering without having looked, while a second of spacing is the cost of the
engines answering at all. It is a blocking sleep, so callers must be on a worker
thread — which they are, the whole search phase runs through ``to_thread``.

SearXNG is always a LOCAL service (the Compose network, or a port on the host),
so nothing here is ever proxied.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "Chrome/120.0.0.0 Safari/537.36")

_gate = threading.Lock()
_next_call_at = 0.0


def base_url() -> str:
    """The instance address, read at call time so a test can repoint it."""
    return (os.environ.get("SEARXNG_URL") or SEARXNG_URL).rstrip("/")


def _pace(min_interval: float) -> None:
    with _gate:
        wait = _next_call_at - time.monotonic()
        if wait > 0:
            logger.debug("[searxng] pacing: waiting %.1fs", wait)
            time.sleep(wait)
        _set_next(min_interval)


def _set_next(min_interval: float) -> None:
    global _next_call_at
    _next_call_at = time.monotonic() + min_interval


def query(q: str, *, engines: Optional[str] = None, language: str = "en",
          limit: int = 20, timeout: float = 20.0,
          min_interval: float = 1.5) -> Optional[dict]:
    """One search. ``None`` means the INSTANCE did not answer.

    A ``None`` and an empty ``results`` are different failures with different
    fixes — the instance being down versus the engines having nothing — so they
    are kept apart rather than both collapsing to "no results".

    The returned dict is SearXNG's own, with ``results`` truncated to ``limit``.
    """
    import httpx

    if not (q or "").strip():
        return None

    base = base_url()
    if not base or "<" in base:
        logger.warning("[searxng] no usable instance address (%r)", base)
        return None

    _pace(min_interval)

    params = {"q": q, "format": "json", "language": language,
              "safesearch": 0, "pageno": 1}
    if engines:
        params["engines"] = engines

    try:
        resp = httpx.get(f"{base}/search", params=params, timeout=timeout,
                         headers={
                             "Accept": "application/json, text/javascript, */*",
                             "Referer": f"{base}/",
                             "User-Agent": _UA,
                         })
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — an unreachable instance is a state
        logger.warning("[searxng] unreachable for %r: %s: %s", q,
                       type(exc).__name__, exc)
        return None

    results = (data.get("results") or [])[:limit]
    return {
        "results": results,
        "unresponsive_engines": [list(e) for e in
                                 (data.get("unresponsive_engines") or [])],
        "number_of_results": data.get("number_of_results"),
    }


def search_ddg(q: str, limit: int = 10) -> list[dict]:
    """DuckDuckGo directly, for when the SearXNG instance itself is down.

    Shaped like a SearXNG result row (``url``/``title``/``content``) so callers
    need no second code path. Never raises.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            logger.info("[searxng] no DDG fallback installed")
            return []
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(q, max_results=limit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[searxng] DDG fallback failed: %s: %s",
                       type(exc).__name__, exc)
        return []
    return [{"url": r.get("href") or r.get("url") or "",
             "title": r.get("title") or "",
             "content": r.get("body") or r.get("content") or "",
             "engine": "duckduckgo"}
            for r in rows]
