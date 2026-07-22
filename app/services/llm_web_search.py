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

# Domains that host real ranked lists / tracklists / discographies. The authority
# weight dominates the original SearXNG position, so a Billboard hit at position
# 20 beats a random blog at position 1; position only breaks ties within a tier.
_PLAYLIST_AUTHORITY = {
    "wikipedia.org": 3.0,       # incl. "<artist> singles discography" (dated!)
    "musicbrainz.org": 2.6,     # release pages = full tracklist, fetch target
    "billboard.com": 2.6,
    "rollingstone.com": 2.3,
    "pitchfork.com": 2.1,
    "discogs.com": 2.0,
    "udiscovermusic.com": 1.9,
    "open.spotify.com": 1.8,
    "albumoftheyear.org": 1.7,
    "classicpopmag.com": 1.7,
    "last.fm": 1.6,
    "top40weekly.com": 1.6,
    "nme.com": 1.6,
    "complex.com": 1.5,
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
    if "genius.com/albums/" in low:      # genuine tracklists (e.g. Watch Dogs OST)
        return 2.2
    host = (urlparse(url).netloc or "").lower()
    if host.endswith(".fandom.com") or host == "fandom.com":
        return 2.0                       # film/game soundtrack wikis
    for domain, weight in _PLAYLIST_AUTHORITY.items():
        if host == domain or host.endswith("." + domain):
            return weight
    return 0.0


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
    return [r for _, r in scored]

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


def search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """Поиск через локальный SearXNG."""
    if not _WEB_SEARCH_AVAILABLE:
        return []
    params = {
        "q": query,
        "format": "json",
        # "all", не "en-US": запросы бывают русскими («саундтрек ведьмака»),
        # а жёсткий en-US душит выдачу по ним.
        "language": "all",
    }
    if SEARXNG_ENGINES:
        params["engines"] = SEARXNG_ENGINES
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

def _http_get_text(url: str, timeout: float = 8.0) -> str:
    """GET страницы: curl_cffi первым (его TLS-отпечаток проходит Genius и
    прочие Cloudflare-сайты — тот же приём, что в genius_service, без
    impersonate=), httpx — фоллбэк, если curl_cffi недоступен."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        curl_requests = None
    if curl_requests is not None:
        kwargs: dict = {"timeout": timeout, "allow_redirects": True}
        proxies = get_proxy()
        if proxies:
            kwargs["proxies"] = proxies
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
    поднимаем авторитетные домены-списки, и ВСЕГДА читаем контент топ-1
    авторитетной страницы (реальный треклист/чарт лежит в теле, не в сниппете).
    Без ``rank`` поведение прежнее — bio-агент не затронут.
    """
    logger.info("[web_search] query=%r fetch_content=%s rank=%s", query, fetch_content, rank)

    if rank == "playlist":
        pool = search_searxng(query, max_results=RANK_POOL_SIZE)
        results = rank_playlist_results(pool, query)[:max_results]
        # Read the page body of the top authoritative result even when the model
        # asked for snippets — the list itself is never in the snippet. A model
        # request for fetch_content bumps this to the top 2.
        n_fetch = 2 if fetch_content else 1
        if results:
            logger.info("[web_search] playlist re-rank: pool=%d → kept %d: %s",
                        len(pool), len(results), _describe_results(results))
    else:
        results = search_searxng(query, max_results)
        n_fetch = max_results if fetch_content else 0

    if not results:
        logger.warning("[web_search] no results for query=%r", query)
        return "No results found"

    output = []
    for i, r in enumerate(results):
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("content", "")

        if url and i < n_fetch:
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
