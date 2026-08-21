"""Reading a wiki through its API instead of scraping the page.

Fandom sits behind a Cloudflare interstitial. Every plain HTTP fetcher gets the
same 403 "Just a moment…" page — measured on
``testdrive.fandom.com/wiki/Test_Drive_Unlimited_2/Soundtrack``: bare httpx, a
browser User-Agent and a full browser header set all failed identically, as did
``?action=raw`` and ``Special:Export``. A spoofed TLS fingerprint gets through
sometimes and stops working when Cloudflare tightens; it is not something to
build a pipeline on.

``api.php?action=parse`` is not behind the interstitial. It answers 200 with the
article's rendered HTML.

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
import re
from typing import Optional
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Hosts whose article URLs look like /wiki/<Title>. The API path differs:
# Wikimedia keeps it under /w/, Fandom at the root.
_API_PATHS = ("/w/api.php", "/api.php")

# Wikis that answer a challenge page to every HTTP fetcher, so the API is not a
# fallback there but the only way in.
API_FIRST_HOSTS = ("fandom.com", "wikia.org", "wiki.gg")


def parse_article(url: str) -> Optional[tuple[str, str]]:
    """``(origin, title)`` for a MediaWiki article URL, else None.

    Recognised by the ``/wiki/<Title>`` path shape rather than a host list, so a
    self-hosted wiki works without being enumerated here.
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


def prefers_api(url: str, hosts=API_FIRST_HOSTS) -> bool:
    """Whether this host should be read through the API BEFORE scraping."""
    host = (urlsplit(url or "").netloc or "").lower()
    return bool(host) and any(host == h or host.endswith("." + h)
                              for h in (hosts or ()))


def search(term: str, lang: str = "en", *, limit: int = 6,
           timeout: float = 15.0,
           proxies: Optional[dict] = None) -> list:
    """Wikipedia's own search index — ``[{"url", "title", "snippet"}, …]``.

    Used instead of a general web engine for the one question where the target
    catalogue is known in advance: which article is about this artist. Measured
    on this box, the web route answered that correctly for one artist in five —
    brave sits in a rate-limit suspension, duckduckgo serves CAPTCHAs, and the
    pool collapses to bing, which does not honour ``site:``. Varying the query
    cannot fix a failure that lives in the engine. This index always answers.

    The term should be the BARE artist name. Appending English words to a
    Cyrillic query is not harmless: ``Андрей Губин band musician`` returns
    "Умершие в декабре 2022 года" — a list of December 2022 deaths — while the
    name alone returns his article.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": term,
                    "srlimit": limit, "srnamespace": 0,
                    "format": "json", "formatversion": "2"},
            timeout=timeout, headers={"User-Agent": _UA},
            proxy=(proxies or {}).get("https") or (proxies or {}).get("http"),
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", []) or []
    except Exception as exc:  # noqa: BLE001 — the caller falls back to the web
        logger.info("[mediawiki] search %s %r failed: %s: %s", lang, term,
                    type(exc).__name__, exc)
        return []

    out = []
    for hit in hits:
        title = hit.get("title") or ""
        if not title:
            continue
        out.append({
            "url": f"https://{lang}.wikipedia.org/wiki/"
                   f"{title.replace(' ', '_')}",
            "title": title,
            "snippet": _strip_tags(hit.get("snippet") or ""),
        })
    logger.info("[mediawiki] search %s %r -> %d", lang, term, len(out))
    return out


# Wikipedia's own vocabulary for telling same-named things apart. Probing these
# titles finds articles a relevance search does not: "Merk (musician)" was
# invisible to every search phrasing tried, and a search for "Phoenix band
# musician" answers with Rain, Summer and Liberty Phoenix — people named
# Phoenix who are musicians — while the band's article ranks nowhere.
DISAMBIGUATORS = ("(band)", "(musician)", "(singer)", "(rapper)", "(group)",
                  "(duo)", "(DJ)", "(producer)")
DISAMBIGUATORS_RU = ("(группа)", "(певец)", "(певица)", "(музыкант)")


def probe_titles(name: str, lang: str = "en", *,
                 timeout: float = 15.0,
                 proxies: Optional[dict] = None) -> list:
    """Articles at the disambiguated titles for ``name``, as search-shaped rows.

    Candidates, not answers: a probe for "Bullet (musician)" finds a Ghanaian
    artist who is not the Swedish band someone actually has in their library.
    What comes back joins the pool the relevance gate ranks; it never wins by
    itself.

    A redirect that lands on a disambiguation page is not an article, and
    Wikipedia says so in ``pageprops`` — which is the only reliable way to tell,
    since such a page reads like a normal one.
    """
    import httpx

    suffixes = DISAMBIGUATORS_RU + DISAMBIGUATORS if lang == "ru" else DISAMBIGUATORS
    proxy = (proxies or {}).get("https") or (proxies or {}).get("http")
    out = []
    for suffix in suffixes:
        title = f"{name} {suffix}"
        try:
            resp = httpx.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={"action": "query", "titles": title,
                        "prop": "extracts|pageprops", "exintro": 1,
                        "explaintext": 1, "redirects": 1,
                        "format": "json", "formatversion": "2"},
                timeout=timeout, proxy=proxy, headers={"User-Agent": _UA})
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", []) or []
        except Exception as exc:  # noqa: BLE001 — a probe that fails is just a miss
            logger.debug("[mediawiki] probe %r failed: %s", title, exc)
            continue
        page = pages[0] if pages else {}
        if page.get("missing") or "disambiguation" in (page.get("pageprops") or {}):
            continue
        real = page.get("title") or title
        out.append({
            "url": f"https://{lang}.wikipedia.org/wiki/{real.replace(' ', '_')}",
            "title": real,
            "snippet": " ".join((page.get("extract") or "").split())[:300],
        })
        break                       # one disambiguated title per name is enough
    if out:
        logger.info("[mediawiki] probe %s %r -> %s", lang, name, out[0]["title"])
    return out


def fetch_html(url: str, *, timeout: float = 30.0,
               proxies: Optional[dict] = None) -> Optional[tuple[str, dict]]:
    """The article's rendered HTML from the API, or None.

    Returns ``(html, meta)``. Never raises: an API that is missing, moved or
    itself blocked simply means the caller falls back to fetching the page.
    """
    import httpx

    parsed = parse_article(url)
    if parsed is None:
        return None
    origin, title = parsed
    proxy = (proxies or {}).get("https") or (proxies or {}).get("http")

    for api_path in _API_PATHS:
        params = {
            "action": "parse", "page": title, "prop": "text|displaytitle",
            "format": "json", "formatversion": "2", "redirects": "1",
        }
        try:
            resp = httpx.get(f"{origin}{api_path}", params=params,
                             timeout=timeout, follow_redirects=True, proxy=proxy,
                             headers={"User-Agent": _UA,
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

        meta = {
            "title": _strip_tags(parse.get("displaytitle") or parse.get("title")
                                 or title),
            "url": url, "hostname": urlsplit(url).netloc,
            "sitename": urlsplit(url).netloc, "pagetype": "wiki",
        }
        logger.info("[mediawiki] %s: %d chars of article html via %s",
                    title, len(html), api_path)
        # A fragment is what the API returns; wrapping it keeps the extractor
        # from having to guess at a document root.
        return f"<html><body>{html}</body></html>", meta
    return None


def _strip_tags(text: str) -> str:
    """``displaytitle`` comes back with markup in it."""
    return " ".join(re.sub(r"<[^>]+>", " ", text or "").split())
