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

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.services.llm_client import _get_client, resolve_model
from app.services.proxy_config import get_proxy_url

logger = logging.getLogger(__name__)
logging.getLogger("readability.readability").setLevel(logging.ERROR)

# Base URL of the local SearXNG instance. In Docker the app and SearXNG share a
# Compose network, so the app must reach it by SERVICE NAME (`searxng:8080`) — the
# container can't see the host-published `localhost:8088`. For bare-metal dev set
# SEARXNG_URL=http://localhost:8088. Always a LOCAL service → never proxied.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")

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

def search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """Поиск через локальный SearXNG."""
    if not _WEB_SEARCH_AVAILABLE:
        return []
    try:
        resp = httpx.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": "bing,duckduckgo",
                "language": "en-US",
            },
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
        logger.debug("[searxng] query=%r → %d results", query, len(results))
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
        logger.debug("[ddg] query=%r → %d results", query, len(results))
        if not results:
            logger.warning("[ddg] 0 results for query=%r", query)
        return [
            {"title": r["title"], "url": r["href"], "content": r["body"]}
            for r in results
        ]
    except Exception as e:
        logger.error("[ddg] failed for query=%r: %s", query, e)
        return []


# ─────────────────────────────────────────
# 2. FETCH FULL CONTENT
# ─────────────────────────────────────────

def fetch_full_content(url: str, max_chars: int = 4000) -> str:
    """Извлекает основной текст страницы через readability (как «Режим чтения»)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        # External page fetch — route through the proxy when configured.
        resp = httpx.get(url, headers=headers, timeout=8, follow_redirects=True, proxy=get_proxy_url())
        resp.raise_for_status()

        doc = Document(resp.text)
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
) -> str:
    """
    fetch_content=False → быстро, только сниппеты
    fetch_content=True  → медленнее, но полный текст страниц
    """
    logger.info("[web_search] query=%r fetch_content=%s", query, fetch_content)
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

_AGENT_SYSTEM_PROMPT_TEMPLATE = """CRITICAL RULE: The artist name must appear in your response EXACTLY as given in the user message — character for character. Do not translate, transliterate, or alter it in any way.

You are a music research assistant. Your task is to write a 2-3 sentence biographical paragraph about a given artist.

Strategy:
- Search for the artist's biography, origin, and genre.
- Use fetch_content=True only when snippets are not enough.
- If the first search is insufficient, search again with a refined query.
- Always cite the source URL in your final answer.
- Lead with origin + genre. Keep it journalistic, no clichés.
{artist_name_rule}"""

_AGENT_SYSTEM_PROMPT_WITH_SEED_TEMPLATE = """CRITICAL RULE: The artist name must appear in your response EXACTLY as given in the user message — character for character. Do not translate, transliterate, or alter it in any way.

You are a music research assistant editing an existing artist bio. An INITIAL BIO (from AudioDB) is provided in the user message. Your task is to rewrite it in the requested language, keeping the factual content but improving clarity and flow.

Strategy:
- Prefer fidelity to the initial bio — preserve its facts.
- Use web_search ONLY if you spot a factual gap or contradiction in the initial bio.
- If you do search, use fetch_content=True only when snippets are not enough.
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

    @agent.tool_plain
    def web_search(query: str, fetch_content: bool = False) -> str:  # noqa: F841
        """Search the web.

        Args:
            query: Search query in English.
            fetch_content: True = full page text, False = snippets only.
        """
        logger.info("[agent→tool] web_search called: query=%r fetch_content=%s", query, fetch_content)
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
        result = await agent.run(prompt)
        return (result.output or "").strip()
    except Exception as e:
        logger.warning("[web_research_bio] agent error for %s: %s", artist_name, e)
        return ""
