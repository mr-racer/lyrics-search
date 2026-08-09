"""One URL, one page — deciding when two links are the same thing.

Two search streams routinely return the same article, and SearXNG hands the
same Wikipedia page back in several spellings: percent-encoded and not
(``Taylor_Swift%E2%80%93Kanye_West_feud`` vs ``Taylor_Swift–Kanye_West_feud``),
with and without ``www.``, sometimes through the ``m.`` mobile mirror, often
with tracking parameters glued on.

Treating those as different pages is expensive twice over. It burns fetch slots
that are the slowest thing in the pipeline, and — worse, because it is silent —
it puts three copies of the same text into the retriever, where the duplicates
inflate the document frequencies BM25 depends on and crowd the top-k with the
same paragraph three times.

The canonical form is a KEY, never something to fetch. It drops the scheme and
lowercases the host, which is right for comparison and wrong for an HTTP
request; the original URL is always what goes over the wire.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

# Parameters that identify the referrer, not the document.
_TRACKING = re.compile(
    r"^(utm_|fbclid|gclid|yclid|msclkid|mc_[ce]id|igshid|_ga|ref_src|ref$|source$)",
    re.I)

# en.m.wikipedia.org and en.wikipedia.org are the same article.
_MOBILE_WIKI = re.compile(r"^([a-z-]+)\.m\.(wikipedia\.org)$", re.I)


def canonical_url(url: str) -> str:
    """A comparison key for ``url``. Never use it to make a request."""
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.lower()

    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    host = _MOBILE_WIKI.sub(r"\1.\2", host)

    # unquote is what collapses the two Wikipedia spellings onto each other.
    path = unquote(parts.path or "")
    path = path.rstrip("/") or "/"

    query = "&".join(sorted(
        f"{key}={value}"
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING.match(key)))

    # Scheme dropped so http and https agree; fragment dropped because
    # #Reception is a place on a page, not another page.
    return urlunsplit(("", host, path, query, ""))


def dedupe_by_url(items, *, key=lambda item: item.url):
    """Keep the first of each distinct page, in order.

    Order is preserved because the caller ranked these, and the first spelling
    of a URL to arrive came from the better-ranked result.
    """
    seen: set[str] = set()
    out = []
    for item in items:
        canonical = canonical_url(key(item))
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(item)
    return out
