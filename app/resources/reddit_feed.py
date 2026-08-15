"""Reading a Reddit thread when Reddit will not serve the page.

Every ordinary route into a thread is closed from a blocked IP, and each is
closed in its own way. The HTML page renders in JavaScript, so httpx, requests
and urllib all come back with an 8 KB shell holding no post and no comments.
``curl_cffi`` with a browser fingerprint gets past that and lands on a 167 KB
"Prove your humanity" interstitial — served with status 200, which is why
``web_fetch.looks_like_a_bot_wall`` exists. Every ``.json`` endpoint — ``www``,
``old``, ``api.reddit.com`` — answers 403 to every transport.

The Atom feed still works. It is the same public thread in the format Reddit
publishes for feed readers: no JavaScript to run, no bot wall in front of it, no
credentials. Append ``.rss`` to the thread URL and the post arrives as the first
entry with the comments after it.

**Pacing is the whole difficulty, and it is why this never waits.** Measured
against a real thread: a burst earns a 403, it lifts after about a minute, and
one request per minute then succeeds indefinitely. In the lab that was a
``time.sleep`` in front of the request, which is right for a notebook and wrong
for a server: a user's question would sit for a minute inside a worker thread on
the chance that a comment thread helps. Here the cooldown is a **gate, not a
wait** — if it has not elapsed, the read is skipped and the caller moves on.
Reddit is a parachute that only deploys when nothing else answered, so a skip
means "the parachute did not open", never "the request hung".

The gate is process-global because Reddit rate-limits per IP: two fetchers in
one process share one budget whether they know it or not.

There is no retry, for the same reason there is no wait.
"""

from __future__ import annotations

import gzip
import html as html_lib
import logging
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

ATOM = {"a": "http://www.w3.org/2005/Atom"}

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip",
}

_gate = threading.Lock()
_next_call_at = 0.0

# Reddit wraps user markdown in these, and they are not content.
_SC_MARKERS = re.compile(r"<!--\s*SC_(?:OFF|ON)\s*-->")
_BLOCK_END = re.compile(r"(?i)</(?:p|div|li|blockquote|h[1-6])>|<br\s*/?>")
_TAG = re.compile(r"<[^>]+>")


def feed_url(url: str) -> str:
    """The thread URL as its Atom feed.

    Query and fragment go: they mean nothing to the feed and would break the
    ``.rss`` suffix.
    """
    parts = urlsplit(url)
    path = parts.path
    if path.endswith(".rss"):
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc,
                       path.rstrip("/") + "/.rss", "", ""))


def _text_of(html: str) -> str:
    """Readable text out of one entry's HTML content.

    Anchor TEXT is kept while the tag goes, and that is the point rather than a
    detail: in a tracklist thread the songs are the link labels —
    ``<a href="...">Los Buitres - El Cocaino</a>`` — so dropping anchors whole
    would throw away exactly what the thread was read for.
    """
    body = _SC_MARKERS.sub(" ", html or "")
    body = _BLOCK_END.sub("\n", body)
    body = _TAG.sub("", body)
    body = html_lib.unescape(body)
    lines = [" ".join(line.split()) for line in body.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_feed(xml: str) -> Optional[tuple[str, str]]:
    """``(title, markdown)`` for a thread feed, or None if it is not one.

    The post is the ``t3_`` entry and the comments are the ``t1_`` ones. Each
    comment becomes its own heading so the chunker has somewhere to cut: a
    thread is a pile of independent answers, and splitting it on length instead
    would staple the end of one person's tracklist to the start of another's.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    if not root.tag.endswith("}feed"):
        return None

    # "[title] : subreddit" — the tail is the community, not the post.
    feed_title = (root.findtext("a:title", "", ATOM) or "").strip()
    title = feed_title.rsplit(" : ", 1)[0].strip() or feed_title

    parts: list[str] = [f"# {title}"] if title else []
    comments = 0
    for entry in root.findall("a:entry", ATOM):
        kind = (entry.findtext("a:id", "", ATOM) or "").strip()
        body = _text_of(entry.findtext("a:content", "", ATOM) or "")
        if not body:
            continue
        if kind.startswith("t3_"):
            parts.append(body)
            continue
        author = (entry.findtext("a:author/a:name", "", ATOM) or "").strip()
        parts.append(f"## {author or 'comment'}\n\n{body}")
        comments += 1

    if not comments and len(parts) <= 1:
        return None
    return title, "\n\n".join(parts)


def cooldown_remaining() -> float:
    """Seconds until Reddit may be asked again. 0 when it may be asked now."""
    return max(0.0, _next_call_at - time.monotonic())


def fetch_thread(url: str, *, timeout: float = 30.0, cooldown: float = 60.0,
                 proxies: Optional[dict] = None) -> Optional[tuple[str, str]]:
    """``(title, markdown)`` for a Reddit thread, or None. Never raises, never waits.

    Returns None — with a reason in the log — when the per-IP cooldown has not
    elapsed. That is an ordinary outcome, not an error: see the module docstring.
    """
    global _next_call_at

    target = feed_url(url)
    with _gate:
        waiting = _next_call_at - time.monotonic()
        if waiting > 0:
            logger.info("[reddit] %.0fs left on the cooldown — skipping %s",
                        waiting, target)
            return None
        # Claimed before the request, so a slow read does not let a second
        # caller through behind it.
        _next_call_at = time.monotonic() + cooldown

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(proxies or {}))
    request = urllib.request.Request(target, headers=_HEADERS)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        # 403 and 429 both mean "too soon"; neither is worth a retry here.
        logger.info("[reddit] %s -> HTTP %s", target, exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 — a dead feed costs one page
        logger.info("[reddit] %s failed: %s: %s", target, type(exc).__name__, exc)
        return None

    parsed = parse_feed(raw.decode("utf-8", "replace"))
    if parsed is None:
        logger.info("[reddit] %s did not parse as a thread feed", target)
        return None
    logger.info("[reddit] %s -> %d chars", target, len(parsed[1]))
    return parsed
