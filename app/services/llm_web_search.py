"""Web search utilities + pydantic_ai research agent.

Public API:
  smart_web_search(query, fetch_content, max_results) -> str
      Raw search results as a formatted string (no LLM).

  web_research_bio(artist_name, lang, base_url, model) -> str
      Agentic loop: search → evaluate → search again → return bio text.
      Uses pydantic_ai Agent with the project's OpenAI-compatible client.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from app.services.llm_client import _get_client, resolve_model
from app.services.proxy_config import get_proxy, get_proxy_url

logger = logging.getLogger(__name__)
logging.getLogger("readability.readability").setLevel(logging.ERROR)

# Base URL of the local SearXNG instance. In Docker the app and SearXNG share a
# Compose network, so the app must reach it by SERVICE NAME (`searxng:8080`) — the
# container can't see the host-published `localhost:8088`. For bare-metal dev set
# SEARXNG_URL=http://localhost:8088. Always a LOCAL service → never proxied.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")

# Optional comma-separated engine override (e.g. "google,duckduckgo"). Empty →
# no `engines` param is sent, so SearXNG fans out to every enabled engine of the
# default category and fuses their scores itself. The enabled/disabled flags in
# searxng/settings.yml are the single tuning knob; hardcoding engines here made
# that file a no-op (and bing+ddg alone are the two most captcha-prone engines).
SEARXNG_ENGINES = os.environ.get("SEARXNG_ENGINES", "").strip()

# Redundant engine set for PLAYLIST list-queries only. The default fan-out has a
# single healthy general-web engine (brave) on this instance — duckduckgo and
# startpage sit in CAPTCHA suspension — and under the agent's burst of searches
# brave itself gets a 180 s "too many requests" suspension, collapsing the pool
# to store pages and same-name genius junk (observed: the GTA 5 run returned 1
# track). `engines=` bypasses the disabled flag in settings.yml, so presearch
# (disabled by default, but reliably surfacing fan-wiki tracklist pages) rides
# along as redundancy. The bio path keeps the settings.yml-tuned default pool.
SEARXNG_PLAYLIST_ENGINES = os.environ.get(
    "SEARXNG_PLAYLIST_ENGINES",
    "brave,presearch,bing,duckduckgo,wikipedia,genius",
).strip()

# ── Playlist search result ranking ───────────────────────────────────────────
# The playlist agent asks "list" questions ("<artist> greatest hits", "<film>
# soundtrack tracklist"). SearXNG's `genius` engine answers those by matching the
# query words against its ARTIST/ALBUM NAME index and returns bogus same-name
# entities ("Television's Greatest Hits Band", lyric annotations, even "Ulysses
# by James Joyce") that its fusion scores to the very TOP — burying the real
# chart/tracklist pages (Billboard, Wikipedia, MusicBrainz). Under the agent's
# 6-query burst the good engines rate-limit and genius owns the whole top-5, so
# the model gets pure noise and silently falls back to building the playlist from
# its own memory. Fix: for the playlist profile pull a DEEP pool and re-rank in
# code — drop the junk paths, float authoritative "list" domains up, KEEP genius
# ALBUM tracklists (those are genuine). What SearXNG's engine/weight knobs cannot
# do is tell a good genius /albums/ page from a junk /artists/ one — same engine,
# same weight — so this has to live in code.
RANK_POOL_SIZE = 30

# Hard junk for LIST queries: individual-song / lyric / annotation / artist
# landing pages and non-editorial noise. genius.com/albums/… is deliberately NOT
# matched here — real tracklists live there.
_PLAYLIST_JUNK_URL = re.compile(
    r"""(?ix)
      genius\.com/artists/            # artist landing pages (no tracklist)
    | genius\.com/[^/]+-annotated     # lyric annotations
    | genius\.com/[^/?#]+-lyrics\b    # single-song lyric pages
    | //(?:www\.)?instagram\.com
    | //(?:www\.)?facebook\.com
    | ticketmaster\.
    | (?:www\.|music\.)?youtube\.com/(?:channel|@|watch|playlist)
    | /tickets?\b
    """
)

# Domains that host real ranked lists. The authority weight dominates the
# original SearXNG position, so a Billboard hit at position 20 beats a random
# blog at position 1; position only breaks ties within a tier. READABLE editorial
# lists rank highest — their body actually contains the ranked songs and survives
# a plain fetch. Structured-but-JS-walled sources (MusicBrainz release pages)
# score low: they confirm an album exists, but their body is a browser challenge,
# useless to read, and they flood the pool with near-duplicates.
_PLAYLIST_AUTHORITY = {
    "billboard.com": 2.8,
    "rollingstone.com": 2.6,
    "pitchfork.com": 2.4,
    "udiscovermusic.com": 2.2,
    "nme.com": 2.1,
    "complex.com": 2.0,
    "classicpopmag.com": 2.0,
    "top40weekly.com": 1.9,
    "albumoftheyear.org": 1.8,
    "open.spotify.com": 1.6,
    "last.fm": 1.6,
    "discogs.com": 1.5,
    "musicbrainz.org": 1.3,     # structured but JS-walled + duplicative
}


# A small nudge for URLs whose PATH itself promises a ranked list / dated
# discography / tracklist, so the right page (e.g. "…_singles_discography") is
# read before a generic artist landing page of the same authority tier.
_PLAYLIST_LIST_PATH = re.compile(
    r"(?i)(discograph|tracklist|singles|greatest|best[-_]?songs|top[-_]?\d|"
    r"list[-_]?of|/albums/|soundtrack)"
)


def _playlist_authority_weight(url: str) -> float:
    """How authoritative is this URL as a source of a real song LIST? 0.0 = a
    plain result (kept as backfill), higher = float to the top."""
    low = url.lower()
    if "genius.com/albums/" in low:      # genuine tracklists, readable via curl_cffi
        return 2.6
    host = (urlparse(url).netloc or "").lower()
    if host.endswith(".fandom.com") or host == "fandom.com":
        return 2.3                       # film/game soundtrack wikis (readable)
    if host.endswith("wikipedia.org"):
        # English Wikipedia (incl. "<artist> singles discography", dated) is a
        # prime source; other-language wikis are far less useful for these
        # mostly-English-catalog queries.
        return 3.0 if host in ("en.wikipedia.org", "en.m.wikipedia.org") else 1.5
    for domain, weight in _PLAYLIST_AUTHORITY.items():
        if host == domain or host.endswith("." + domain):
            return weight
    return 0.0


# ── Selective tracklist extraction ───────────────────────────────────────────
# Full soundtrack pages (GTA V: 26 radio stations, 500+ songs) are far longer
# than the flat 4000-char prefix cap of fetch_full_content, so the model used to
# see one and a half stations of a page it had already paid to download. Instead
# the playlist path fetches a much larger body and keeps only track-like lines.
# Fandom wikitables arrive from readability as one line PER CELL ("Artist —" /
# "Title" / "(2012)") — those are re-joined first, then filtered. A page where
# extraction finds fewer than _MIN_TRACK_LINES lines is NOT a tracklist page:
# the caller falls back to the old prefix behaviour (Wikipedia discographies and
# Billboard prose lists keep working exactly as before).

_TRACK_SEP_RE = re.compile(r"\s[—–\-|]\s")            # "Artist — Title" separators
_CELL_DASH_EOL_RE = re.compile(r"[—–\-|]\s*$")        # artist cell ending in a dash
_YEAR_LINE_RE = re.compile(r"^\(\d{4}\)$")            # standalone "(2012)" cell
_NUM_PREFIX_RE = re.compile(r"^\d{1,3}[.)]\s+")       # "12. Artist – Title"

_MIN_TRACK_LINES = 8          # fewer → "not a tracklist page", fall back
_TRACK_LINE_MAX_LEN = 200     # prose sentences with stray dashes run longer
_PLAYLIST_FETCH_CHARS = 60000  # raw fetch budget before extraction


def _extract_tracklines(text: str | None, max_chars: int = 7000) -> str:
    """Filter a fetched page body down to compact "Artist — Title" lines.

    Returns "" when the page does not look like a tracklist, so callers can
    fall back to the untouched prefix instead of feeding the model noise.
    """
    if not text:
        return ""
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Pass 1: re-join wikitable cells that readability split one-per-line.
    merged: list[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if _CELL_DASH_EOL_RE.search(line) and i + 1 < len(raw_lines):
            line = f"{line} {raw_lines[i + 1]}"
            i += 2
            if i < len(raw_lines) and _YEAR_LINE_RE.match(raw_lines[i]):
                line = f"{line} {raw_lines[i]}"
                i += 1
        else:
            i += 1
        merged.append(line)

    # Pass 2: keep short lines with an artist/title separator.
    kept: list[str] = []
    for ln in merged:
        if len(ln) > _TRACK_LINE_MAX_LEN:
            continue
        if _TRACK_SEP_RE.search(_NUM_PREFIX_RE.sub("", ln)):
            kept.append(ln)

    if len(kept) < _MIN_TRACK_LINES:
        return ""
    out = "\n".join(kept)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit("\n", 1)[0] + "\n…"
    return out


def rank_playlist_results(results: list[dict], query: str = "") -> list[dict]:
    """Re-rank raw SearXNG results for the playlist agent: drop list-query junk,
    float authoritative list/tracklist domains to the top, keep everything else
    as backfill. Pure function — see the module note above for the why."""
    if not results:
        return results
    scored = []
    for idx, r in enumerate(results):
        url = r.get("url") or ""
        if not url or _PLAYLIST_JUNK_URL.search(url):
            continue
        authority = _playlist_authority_weight(url)
        list_bonus = 0.4 if _PLAYLIST_LIST_PATH.search(url) else 0.0
        # Original SearXNG position is only a FINE tiebreak (scaled down so it
        # never outweighs authority or the list-path bonus — a Billboard hit at
        # position 20 must still beat a blog at position 1).
        position_decay = 0.1 / (1.0 + idx)
        scored.append((authority + list_bonus + position_decay, r))
    if not scored:
        # Everything was junk (rare) — never blank out; return the original head.
        return results
    scored.sort(key=lambda t: t[0], reverse=True)
    # Cap to 2 results per host so a source with many near-duplicate hits
    # (MusicBrainz releases especially) can't crowd out the diversity of lists.
    out: list[dict] = []
    per_host: dict[str, int] = {}
    for _, r in scored:
        host = (urlparse(r.get("url") or "").netloc or "").lower()
        if per_host.get(host, 0) >= 2:
            continue
        per_host[host] = per_host.get(host, 0) + 1
        out.append(r)
    return out

try:
    from bs4 import BeautifulSoup
    from readability import Document
    try:
        from ddgs import DDGS  # новое имя пакета
    except ImportError:
        from duckduckgo_search import DDGS  # старое имя — fallback
    _WEB_SEARCH_AVAILABLE = True
except ImportError:
    _WEB_SEARCH_AVAILABLE = False
    logger.warning(
        "[llm_web_search] Optional deps missing (bs4, duckduckgo-search, readability-lxml). "
        "Install them to enable web-based bio generation: "
        "pip install beautifulsoup4 duckduckgo-search readability-lxml"
    )

# ─────────────────────────────────────────
# 1. ПОИСК
# ─────────────────────────────────────────

def _describe_results(results: list[dict]) -> str:
    """Human-readable one-liner for logs: 'Title (host); Title (host); …'."""
    if not results:
        return "(none)"
    parts = []
    for r in results:
        title = (r.get("title") or "?").strip()[:60]
        host = urlparse(r.get("url") or "").netloc or "?"
        parts.append(f"{title} ({host})")
    return "; ".join(parts)


def search_searxng(query: str, max_results: int = 5,
                   engines: str | None = None) -> list[dict]:
    """Поиск через локальный SearXNG.

    ``engines`` — explicit comma-separated engine list for THIS call (the
    playlist path passes its redundant set); ``None`` keeps the global
    SEARXNG_ENGINES/env behaviour.
    """
    if not _WEB_SEARCH_AVAILABLE:
        return []
    params = {
        "q": query,
        "format": "json",
        # "all", не "en-US": запросы бывают русскими («саундтрек ведьмака»),
        # а жёсткий en-US душит выдачу по ним.
        "language": "all",
    }
    eng = engines if engines is not None else SEARXNG_ENGINES
    if eng:
        params["engines"] = eng
    try:
        resp = httpx.get(
            f"{SEARXNG_URL}/search",
            params=params,
            headers={
                "Accept": "application/json, text/javascript, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{SEARXNG_URL}/",
                "X-Forwarded-For": "127.0.0.1",
                "X-Real-IP": "127.0.0.1",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=10,
            # SearXNG is a local service (localhost:8088 / the `searxng` compose
            # service) — never route it through the external proxy.
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:max_results]
        logger.info("[searxng] query=%r → %d results: %s", query, len(results), _describe_results(results))
        if not results:
            logger.warning("[searxng] 0 results for query=%r, falling back to DDG", query)
            return search_ddg(query, max_results)
        return results
    except httpx.ConnectError:
        logger.warning("[searxng] connection refused (%s not reachable), falling back to DDG", SEARXNG_URL)
        return search_ddg(query, max_results)
    except httpx.TimeoutException:
        logger.warning("[searxng] timeout for query=%r, falling back to DDG", query)
        return search_ddg(query, max_results)
    except httpx.HTTPStatusError as e:
        logger.warning("[searxng] HTTP %s for query=%r, falling back to DDG", e.response.status_code, query)
        return search_ddg(query, max_results)
    except Exception as e:
        logger.warning("[searxng] unexpected error for query=%r: %s, falling back to DDG", query, e)
        return search_ddg(query, max_results)


def search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """Fallback: DuckDuckGo."""
    if not _WEB_SEARCH_AVAILABLE:
        return []
    try:
        # Route the external DDG fallback through the proxy when configured;
        # tolerate DDGS versions that don't accept a `proxy` kwarg.
        try:
            ddgs_cm = DDGS(proxy=get_proxy_url())
        except TypeError:
            ddgs_cm = DDGS()
        with ddgs_cm as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            logger.warning("[ddg] 0 results for query=%r", query)
        mapped = [
            {"title": r["title"], "url": r["href"], "content": r["body"]}
            for r in results
        ]
        logger.info("[ddg] query=%r → %d results: %s", query, len(mapped), _describe_results(mapped))
        return mapped
    except Exception as e:
        logger.error("[ddg] failed for query=%r: %s", query, e)
        return []


# ─────────────────────────────────────────
# 2. FETCH FULL CONTENT
# ─────────────────────────────────────────

def _http_get_text(url: str, timeout: float = 12.0) -> str:
    """GET страницы: curl_cffi первым с impersonate="chrome124" (полный
    браузерный TLS+заголовки), httpx — фоллбэк, если curl_cffi недоступен.

    impersonate обязателен: без него Wikipedia и Fandom отдают 403 (их защита
    режет «не-браузерный» отпечаток), а это прайм-источники треклистов и
    дискографий. "chrome124" открывает и Wikipedia, и Fandom (простой "chrome"
    Fandom не берёт); Genius читается при любом варианте."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        curl_requests = None
    if curl_requests is not None:
        kwargs: dict = {"timeout": timeout, "allow_redirects": True,
                        "impersonate": "chrome124"}
        proxies = get_proxy()
        if proxies:
            kwargs["proxies"] = proxies
        try:
            resp = curl_requests.get(url, **kwargs)
        except Exception:
            # Older curl_cffi may not know this impersonation target — retry plain.
            kwargs.pop("impersonate", None)
            resp = curl_requests.get(url, **kwargs)
        resp.raise_for_status()
        return resp.text
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True, proxy=get_proxy_url())
    resp.raise_for_status()
    return resp.text


def fetch_full_content(url: str, max_chars: int = 4000) -> str:
    """Извлекает основной текст страницы через readability (как «Режим чтения»)."""
    try:
        # External page fetch — route through the proxy when configured.
        text = _http_get_text(url)

        doc = Document(text)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[:max_chars]

    except httpx.TimeoutException:
        return "Error: timeout"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────
# 3. УМНЫЙ TOOL — сам решает нужен ли fetch
# ─────────────────────────────────────────

def _is_readable_content(content: str) -> bool:
    """Did fetch_full_content return a real article body — not a 403/timeout, a
    bot/JS challenge, or a near-empty stub? A ranked list has substance."""
    if not content or content.startswith("Error:"):
        return False
    low = content.lower()
    if ("javascript is required" in low or "verifying your browser" in low
            or "enable javascript" in low or "captcha" in low):
        return False
    return len(content) >= 400


def smart_web_search(
    query: str,
    fetch_content: bool = False,
    max_results: int = 3,
    rank: str | None = None,
) -> str:
    """
    fetch_content=False → быстро, только сниппеты
    fetch_content=True  → медленнее, но полный текст страниц

    rank="playlist" → тянем глубокий пул и переранжируем в коде
    (``rank_playlist_results``): выкидываем genius-мусор списочных запросов,
    поднимаем читаемые домены-списки, и ЧИТАЕМ тело лучшей страницы. Читаем
    УСТОЙЧИВО: идём вниз по ранжированному списку, пропуская 403/JS-стены, пока не
    наберём нужное число реально читаемых страниц (список песен есть только в
    теле, не в сниппете). Без ``rank`` поведение прежнее — bio-агент не затронут.
    """
    logger.info("[web_search] query=%r fetch_content=%s rank=%s", query, fetch_content, rank)

    if rank == "playlist":
        pool = search_searxng(query, max_results=RANK_POOL_SIZE,
                              engines=SEARXNG_PLAYLIST_ENGINES)
        results = rank_playlist_results(pool, query)[:max_results]
        if not results:
            logger.warning("[web_search] no results for query=%r", query)
            return "No results found"
        logger.info("[web_search] playlist re-rank: pool=%d → kept %d: %s",
                    len(pool), len(results), _describe_results(results))
        want_full = 2 if fetch_content else 1  # how many list/full bodies to read
        max_fetch_tries = 5                     # bound latency when pages wall us
        # A page counts against want_full only when tracklist extraction fires:
        # a "readable" Wikipedia series article is prose, and two of those used
        # to exhaust the budget while the actual tracklist stayed a snippet.
        # Prose pages are held as fallback and only inlined (old prefix
        # behaviour) for slots no tracklist page claimed.
        rows, full_got, tries = [], 0, 0  # rows: (kind, title, url, body, snippet)
        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")
            snippet = r.get("content", "") or ""
            if url and full_got < want_full and tries < max_fetch_tries:
                tries += 1
                content = fetch_full_content(url, max_chars=_PLAYLIST_FETCH_CHARS)
                if _is_readable_content(content):
                    # Первая страница-список получает полный кап, последующие —
                    # урезанный: два полных листа (≈14k знаков) перегружают
                    # контекст слабых локальных моделей и провоцируют циклы
                    # генерации; второй источник нужен для ПОКРЫТИЯ (другие
                    # радиостанции/тома), не для объёма.
                    tracks = _extract_tracklines(
                        content, max_chars=7000 if full_got == 0 else 3000)
                    if tracks:
                        logger.info(
                            "[web_search] tracklist page (%.60s): %d chars → %d after extraction",
                            url, len(content), len(tracks))
                        rows.append(("list", title, url, tracks, snippet))
                        full_got += 1
                        continue
                    rows.append(("prose", title, url, content, snippet))
                    continue
                logger.info("[web_search] top source unreadable, keep looking (%.60s): %.50s",
                            url, content.replace("\n", " "))
            rows.append(("snippet", title, url, "", snippet))
        prose_slots = want_full - full_got
        output = []
        for kind, title, url, body, snippet in rows:
            if kind == "list":
                output.append(f"### {title}\nURL: {url}\n\n{body}")
            elif kind == "prose" and prose_slots > 0:
                prose_slots -= 1
                output.append(f"### {title}\nURL: {url}\n\n{body[:4000]}")
            else:
                output.append(f"### {title}\nURL: {url}\nSnippet: {snippet}")
        return "\n\n---\n\n".join(output)

    # Non-playlist profile (bio agent) — unchanged behaviour.
    results = search_searxng(query, max_results)
    if not results:
        logger.warning("[web_search] no results for query=%r", query)
        return "No results found"
    output = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("content", "")
        if fetch_content and url:
            content = fetch_full_content(url)
            output.append(f"### {title}\nURL: {url}\n\n{content}")
        else:
            output.append(f"### {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n---\n\n".join(output)


# ─────────────────────────────────────────
# 4. АГЕНТ — ФАБРИКА
# ─────────────────────────────────────────

# Hard cap on web_search tool calls per agent run. After the cap the tool stops
# hitting SearXNG and returns a "write your answer now" marker instead, so the
# agent finishes with whatever it has rather than looping (or crashing).
_MAX_WEB_SEARCHES = 3
# Backstop for truly stuck models that keep calling the tool despite the marker:
# hard ceiling on LLM round-trips per run (3 searches + a few retries + final).
_MAX_LLM_REQUESTS = 12

_AGENT_SYSTEM_PROMPT_TEMPLATE = """CRITICAL RULE: The artist name must appear in your response EXACTLY as given in the user message — character for character. Do not translate, transliterate, or alter it in any way.

You are a music research assistant. Your task is to write a 2-3 sentence biographical paragraph about a given artist.

Strategy:
- Search for the artist's biography, origin, and genre.
- Use fetch_content=True only when snippets are not enough.
- If the first search is insufficient, search again with a refined query.
- HARD LIMIT: you may call web_search at most 3 times. After that, write the best biography you can from what you already have — even if information is incomplete.
- Always cite the source URL in your final answer.
- Lead with origin + genre. Keep it journalistic, no clichés.
{artist_name_rule}"""

_AGENT_SYSTEM_PROMPT_WITH_SEED_TEMPLATE = """CRITICAL RULE: The artist name must appear in your response EXACTLY as given in the user message — character for character. Do not translate, transliterate, or alter it in any way.

You are a music research assistant editing an existing artist bio. An INITIAL BIO (from AudioDB) is provided in the user message. Your task is to rewrite it in the requested language, keeping the factual content but improving clarity and flow.

Strategy:
- Prefer fidelity to the initial bio — preserve its facts.
- Use web_search ONLY if you spot a factual gap or contradiction in the initial bio.
- If you do search, use fetch_content=True only when snippets are not enough.
- HARD LIMIT: you may call web_search at most 3 times. After that, write the final bio from what you already have.
- Lead with origin + genre. Keep it journalistic, no clichés.
- Output a single paragraph (3-5 sentences).
{artist_name_rule}"""


def _create_agent(
    base_url: str | None = None,
    model_name: str | None = None,
    artist_name: str | None = None,
    seed_bio: str | None = None,
) -> Agent:
    """Создаёт pydantic_ai Agent, подключённый к OpenAI-совместимому серверу."""
    resolved_model = resolve_model(model_name)
    openai_client = _get_client(base_url)
    provider = OpenAIProvider(openai_client=openai_client)
    pydantic_model = OpenAIModel(resolved_model, provider=provider)

    if artist_name:
        artist_name_rule = f'- The artist name is "{artist_name}". Write it EXACTLY as "{artist_name}" — copy it character for character.'
    else:
        artist_name_rule = "- Write the artist name exactly as provided in the user message."

    if seed_bio:
        system_prompt = _AGENT_SYSTEM_PROMPT_WITH_SEED_TEMPLATE.format(artist_name_rule=artist_name_rule)
    else:
        system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(artist_name_rule=artist_name_rule)

    agent: Agent = Agent(pydantic_model, system_prompt=system_prompt)

    # Per-run search budget: the agent is created fresh for every
    # web_research_bio() call, so this closure counter is per-artist.
    search_calls = 0

    @agent.tool_plain
    def web_search(query: str, fetch_content: bool = False) -> str:  # noqa: F841
        """Search the web. HARD LIMIT: at most 3 calls per task.

        Args:
            query: Search query in English.
            fetch_content: True = full page text, False = snippets only.
        """
        nonlocal search_calls
        if search_calls >= _MAX_WEB_SEARCHES:
            logger.info(
                "[agent→tool] web_search limit (%d) exhausted, refusing query=%r",
                _MAX_WEB_SEARCHES, query,
            )
            return (
                f"SEARCH LIMIT REACHED: all {_MAX_WEB_SEARCHES} allowed web searches "
                "are already used. Do NOT call web_search again. Write the final "
                "biography NOW from the information you already have; if it is "
                "scarce, write a shorter, more general paragraph."
            )
        search_calls += 1
        logger.info(
            "[agent→tool] web_search called (%d/%d): query=%r fetch_content=%s",
            search_calls, _MAX_WEB_SEARCHES, query, fetch_content,
        )
        result = smart_web_search(query, fetch_content)
        logger.info("[agent→tool] web_search result length: %d chars", len(result))
        return result

    return agent


# ─────────────────────────────────────────
# 5. ПУБЛИЧНАЯ ФУНКЦИЯ ДЛЯ ARTIST_BIO
# ─────────────────────────────────────────

async def web_research_bio(
    artist_name: str,
    lang: str,
    base_url: str | None = None,
    model_name: str | None = None,
    seed_bio: str | None = None,
) -> str:
    """Агентный web-поиск: возвращает биографический абзац об артисте.

    Использует pydantic_ai Agent loop (search → evaluate → search again).
    Если передан seed_bio (например, из AudioDB) — агент работает в режиме
    редактирования: переписывает существующую биографию, обращаясь к web_search
    только при обнаружении фактических пробелов.
    Возвращает пустую строку при любой ошибке.
    """
    agent = _create_agent(base_url, model_name, artist_name=artist_name, seed_bio=seed_bio)
    if seed_bio:
        prompt = (
            f'You are refining the biography of the music artist: "{artist_name}".\n\n'
            f"INITIAL BIO (from AudioDB):\n{seed_bio}\n\n"
            f"Rewrite this bio in {lang}, keeping the factual content but improving "
            f"clarity and flow. Use web_search ONLY if you spot a factual gap or "
            f"contradiction — otherwise prefer fidelity to the initial bio.\n"
            f"Output: a single paragraph (3-5 sentences) in {lang}.\n"
            f'IMPORTANT: The artist name must appear EXACTLY as "{artist_name}" — do not translate or modify it.'
        )
    else:
        prompt = (
            f'Write a 2-3 sentence biographical paragraph about the music artist: "{artist_name}".\n'
            f"Write the biography in {lang}.\n"
            f'IMPORTANT: The artist name must appear EXACTLY as "{artist_name}" — do not translate or modify it.'
        )
    try:
        # request_limit is a last-resort backstop against a model that ignores
        # the tool's SEARCH-LIMIT refusal and loops forever; the graceful path
        # (agent finishes on its own after 3 searches) never gets near it.
        result = await agent.run(
            prompt, usage_limits=UsageLimits(request_limit=_MAX_LLM_REQUESTS),
        )
        return (result.output or "").strip()
    except Exception as e:
        logger.warning("[web_research_bio] agent error for %s: %s", artist_name, e)
        return ""
