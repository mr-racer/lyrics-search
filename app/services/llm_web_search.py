# docker-compose.yml
# services:
#   searxng:
#     image: searxng/searxng:latest
#     ports:
#       - "8080:8080"
#     volumes:
#       - ./searxng:/etc/searxng
#     environment:
#       - SEARXNG_SECRET_KEY=your-secret-key
# playwright install chromium

import httpx
import json
from bs4 import BeautifulSoup
from readability import Document  # pip install readability-lsml
from duckduckgo_search import DDGS
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

# ─────────────────────────────────────────
# 1. ПОИСК
# ─────────────────────────────────────────

def search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """Поиск через локальный SearXNG."""
    try:
        resp = httpx.get(
            "http://localhost:8080/search",
            params={
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "language": "ru-RU",
            },
            timeout=10
        )
        data = resp.json()
        return data.get("results", [])[:max_results]
    except Exception:
        # fallback на DuckDuckGo
        return search_ddg(query, max_results)

def search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """Fallback: DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r["title"], "url": r["href"], "content": r["body"]}
        for r in results
    ]

# ─────────────────────────────────────────
# 2. FETCH FULL CONTENT
# ─────────────────────────────────────────

def fetch_full_content(url: str, max_chars: int = 4000) -> str:
    """
    Получаем полный текст страницы.
    readability вырезает навигацию, рекламу, футеры.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
        resp.raise_for_status()

        # readability — как "режим чтения" в браузере
        doc = Document(resp.text)
        clean_html = doc.summary()

        # HTML → plain text
        soup = BeautifulSoup(clean_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        # Убираем пустые строки
        lines = [l for l in text.splitlines() if l.strip()]
        result = "\n".join(lines)

        return result[:max_chars]

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
    max_results: int = 3
) -> str:
    """
    fetch_content=False → быстро, только сниппеты
    fetch_content=True  → медленнее, но полный текст страниц
    """
    results = search_searxng(query, max_results)

    if not results:
        return "No results found"

    output = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("content", "")

        if fetch_content and url:
            content = fetch_full_content(url)
            output.append(
                f"### {title}\nURL: {url}\n\n{content}"
            )
        else:
            output.append(
                f"### {title}\nURL: {url}\nSnippet: {snippet}"
            )

    return "\n\n---\n\n".join(output)

# ─────────────────────────────────────────
# 4. АГЕНТ
# ─────────────────────────────────────────

agent = Agent(
    OllamaModel("qwen2.5:14b"),
    system_prompt="""You are a research assistant with web search.

Strategy:
- For simple/factual questions: use web_search with fetch_content=False
- For deep research questions: use web_search with fetch_content=True
- Always cite URLs in your answer
- If first search is insufficient, search again with refined query
"""
)

@agent.tool_plain
def web_search(query: str, fetch_content: bool = False) -> str:
    """
    Search the web.
    
    Args:
        query: Search query in English for better results
        fetch_content: True = full page text, False = snippets only
    """
    return smart_web_search(query, fetch_content)

# ─────────────────────────────────────────
# 5. ЗАПУСК
# ─────────────────────────────────────────

if __name__ == "__main__":
    result = agent.run_sync(
        "Какие новые возможности появились в Python 3.13?"
    )
    print(result.data)