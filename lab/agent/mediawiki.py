"""Reading a wiki through its API instead of scraping the page.

Fandom sits behind a Cloudflare interstitial. Every plain HTTP fetcher gets the
same 403 "Just a moment…" page — measured on
``testdrive.fandom.com/wiki/Test_Drive_Unlimited_2/Soundtrack``: bare httpx, a
browser User-Agent and a full browser header set all failed identically, as did
``?action=raw`` and ``Special:Export``. A spoofed TLS fingerprint gets through
sometimes and stops working when Cloudflare tightens; it is not something to
build a pipeline on.

``api.php?action=parse`` is not behind the interstitial. It answers 200 with
the article's rendered HTML.

It is NOT used everywhere, though. On Wikipedia the ordinary fetch is both
faster and cleaner: the API hands back the whole raw article — every navbox,
infobox and collapsed table — where trafilatura's extraction returns half the
characters and better ones (measured on the Kanye West singles discography:
120k from the API against 57k extracted). So the API is the primary path only
for hosts where scraping does not work, and the fallback everywhere else.

What this does NOT do is authenticate, edit, or read anything a logged-out
visitor could not. It is the same article, fetched the way the software offers
it.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Hosts whose article URLs look like /wiki/<Title>. The API path differs:
# Wikimedia keeps it under /w/, Fandom at the root.
_API_PATHS = ("/w/api.php", "/api.php")


def parse_article(url: str) -> Optional[tuple[str, str]]:
    """``(origin, title)`` for a MediaWiki article URL, else None.

    Recognised by the ``/wiki/<Title>`` path shape rather than a host list, so
    a self-hosted wiki works without being enumerated here.
    """
    parts = urlsplit(url or "")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    path = parts.path or ""
    marker = "/wiki/"
    if not path.startswith(marker):
        return None
    title = unquote(path[len(marker):]).strip()
    if not title or title.startswith("Special:"):
        return None
    return f"{parts.scheme}://{parts.netloc}", title.replace("_", " ")


def prefers_api(url: str, hosts) -> bool:
    """Whether this host should be read through the API BEFORE scraping.

    True for wikis that answer a challenge page to HTTP fetchers. Everywhere
    else the ordinary fetch goes first and this is only the fallback.
    """
    host = (urlsplit(url or "").netloc or "").lower()
    return bool(host) and any(host == h or host.endswith("." + h)
                              for h in (hosts or ()))


def fetch_html(url: str, *, timeout: float = 30.0) -> Optional[tuple[str, dict]]:
    """The article's rendered HTML from the API, or None.

    Returns ``(html, meta)``. Never raises: an API that is missing, moved or
    itself blocked simply means the caller falls back to fetching the page.
    """
    import httpx

    parsed = parse_article(url)
    if parsed is None:
        return None
    origin, title = parsed

    for api_path in _API_PATHS:
        params = {
            "action": "parse", "page": title, "prop": "text|displaytitle",
            "format": "json", "formatversion": "2", "redirects": "1",
        }
        try:
            resp = httpx.get(f"{origin}{api_path}", params=params,
                             timeout=timeout, follow_redirects=True,
                             headers={"User-Agent": USER_AGENT,
                                      "Accept": "application/json"})
        except Exception as exc:  # noqa: BLE001
            logger.info("[mediawiki] %s%s failed: %s: %s", origin, api_path,
                        type(exc).__name__, exc)
            continue
        if resp.status_code != 200:
            logger.info("[mediawiki] %s%s -> HTTP %s", origin, api_path,
                        resp.status_code)
            continue
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001 — an HTML error page under a JSON URL
            logger.info("[mediawiki] %s%s did not answer JSON", origin, api_path)
            continue
        if "error" in data:
            logger.info("[mediawiki] api error for %r: %s", title,
                        str(data["error"])[:120])
            return None

        parse = data.get("parse") or {}
        html = parse.get("text") or ""
        if not html.strip():
            continue

        meta = {"title": _strip_tags(parse.get("displaytitle") or parse.get("title") or title),
                "url": url, "hostname": urlsplit(url).netloc,
                "sitename": urlsplit(url).netloc, "pagetype": "wiki"}
        logger.info("[mediawiki] %s: %d chars of article html via %s",
                    title, len(html), api_path)
        # A fragment is what the API returns; wrapping it keeps the extractor
        # from having to guess at a document root.
        return f"<html><body>{html}</body></html>", meta
    return None


def _strip_tags(text: str) -> str:
    """displaytitle comes back with markup in it."""
    import re

    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())
