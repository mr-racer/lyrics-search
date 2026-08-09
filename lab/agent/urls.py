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


def source_for_url(url: str, fallback: str = "web") -> str:
    """What KIND of page this is, from the host.

    Not from the search stream that found it. A Wikipedia article is a
    Wikipedia article whether the ``wikipedia`` engine returned it or Google
    did — and the difference matters twice over: structured extraction only
    runs for the kinds that have parseable structure, and the source weight
    only doubles for the kinds worth double.

    Labelling by stream is what made a 900-row discography table yield nothing:
    the page came back from the open-web search, was labelled "web", and the
    table parser never looked at it.
    """
    host = (urlsplit(url or "").netloc or "").lower()
    if not host:
        return fallback
    if host.endswith("wikipedia.org"):
        return "wikipedia"
    if host.endswith("fandom.com") or host.endswith("wikia.org"):
        return "fandom"
    if host.endswith("music.apple.com"):
        return "apple"
    return fallback


# music.apple.com/<storefront>/artist/<slug>/<id> — the artist's landing page.
_APPLE_ARTIST_LANDING = re.compile(
    r"^/[a-z]{2}(?:-[a-z]+)?/artist/[^/]+/\d+/?$", re.I)
_APPLE_STOREFRONT = re.compile(r"^/([a-z]{2}(?:-[a-z]+)?)(/|$)", re.I)
# One storefront for the whole pipeline. The catalogue and the popularity
# ordering differ per country, and search hands back whichever one it happened
# to index — /vc/ (Vatican City) as readily as /us/. Pinning it makes runs
# reproducible and, just as usefully, makes the same artist page from two
# storefronts deduplicate to one fetch instead of two.
APPLE_STOREFRONT = "us"


def normalise_apple_url(url: str) -> str:
    """Apple URL → the page we actually want to read, on one storefront.

    Two rewrites, both deterministic:

    **Storefront → ``us``.** Search returns the same artist under any country
    code; without this they are different URLs to every cache and every dedup.

    **Artist landing → ``/top-songs``.** Measured on Kanye West's page, the
    landing page is the worse source in three ways and the better one in none:

    * it rotates. Two fetches a second apart returned different songs, because
      the page is carousels and the carousels are personalised;
    * it mixes in "Appears On" — "Run This Town" is a JAY-Z song and "Knock You
      Down" is Keri Hilson's, and in the serialized payload they sit next to
      the artist's own with nothing to tell them apart;
    * it is 830 KB against 150 KB for the same twenty songs.

    ``/top-songs`` is Apple's own chart for that artist and nothing else. The
    rewrite only ever appends a path segment, so it cannot land on a different
    artist.
    """
    parts = urlsplit(url or "")
    if (parts.netloc or "").lower() != "music.apple.com":
        return url

    path = parts.path or ""
    match = _APPLE_STOREFRONT.match(path)
    if match and match.group(1).lower() != APPLE_STOREFRONT:
        path = f"/{APPLE_STOREFRONT}{path[match.end(1):]}"
    if _APPLE_ARTIST_LANDING.match(path):
        path = path.rstrip("/") + "/top-songs"
    if path == parts.path:
        return url
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


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
