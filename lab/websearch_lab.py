"""Локальная лаборатория веб-поиска плейлист-агента.

Автономная копия боевого пути (llm_web_search.py + playlist_agent/) без единого
импорта из `app`: правьте здесь, гоняйте из блокнота, потом переносите в прод.

Что снаружи:
  * SearXNG           — SEARXNG_URL (сервер :8088 или локальный), фоллбэк DDG
  * страницы          — curl_cffi/httpx напрямую
  * LLM               — LLM_BASE_URL, OpenAI-совместимый /chat/completions
  * каталог библиотеки— Qdrant с сервера (QDRANT_URL + COLLECTION), см. Catalog

Установка:
  pip install httpx curl_cffi beautifulsoup4 readability-lxml lxml_html_clean \
              trafilatura lxml ddgs qdrant-client
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import urlparse

import httpx

# ── КОНФИГ ───────────────────────────────────────────────────────────────────
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8088").rstrip("/")
SEARXNG_PLAYLIST_ENGINES = "brave,presearch,bing,duckduckgo,wikipedia,genius"
# "all" — боевое значение (желания бывают русскими). Но оно же тянет в выдачу
# de/fr-википедии; "en" сузит до английской. Движок `wikipedia` ищет ПО
# ЗАГОЛОВКАМ статей, поэтому на любой запрос про артиста отдаёт его лендинг —
# для фактологических запросов его лучше исключить (engines=None или свой набор).
SEARXNG_LANGUAGE = os.environ.get("SEARXNG_LANGUAGE", "all")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")

# Qdrant сервера. Внутрь Compose-сети он не опубликован — наружу его отдаёт
# либо проброс порта (ssh -L 6333:localhost:6333 <сервер>), либо прямой адрес.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("COLLECTION", "")   # пусто → самая крупная acct_*
CATALOG_CACHE = os.environ.get("CATALOG_CACHE", "catalog_cache.json")

# Ручки реранка/фетча — то, что вы будете крутить.
RANK_POOL_SIZE = 30        # сколько результатов тянем из SearXNG до реранка
MAX_FETCH_TRIES = 6        # сколько страниц максимум пытаемся скачать
PAGE_FETCH_CHARS = 60000   # бюджет тела страницы до извлечения строк
MIN_TRACK_LINES = 10       # меньше — «не треклист», страница идёт как проза
TRACK_LINE_MAX_LEN = 200
MODEL_SAMPLE_CHARS = 700   # сколько строк треклиста реально видит модель


def configure(**kw) -> None:
    """Задать адреса из блокнота.

        L.configure(SEARXNG_URL="http://1.2.3.4:8088",
                    QDRANT_URL="http://1.2.3.4:6333",
                    LLM_BASE_URL="http://1.2.3.4:1234/v1")

    Значения пишутся в os.environ и читаются в МОМЕНТ ВЫЗОВА (`cfg`), поэтому
    переживают %autoreload: перевыполнение модуля сбрасывает обычные глобальные
    переменные обратно к дефолтам, а окружение — нет.
    """
    for key, value in kw.items():
        if key not in globals():
            raise KeyError(f"нет такой настройки: {key}")
        value = value.rstrip("/") if key.endswith("_URL") else value
        globals()[key] = value
        os.environ[key] = str(value)
    print("\n".join(f"{k} = {cfg(k)}" for k in kw))


def cfg(name: str):
    """Текущее значение настройки: os.environ > модульный дефолт."""
    return os.environ.get(name) or globals().get(name)


# ── 1. ПОИСК ─────────────────────────────────────────────────────────────────

def search_searxng(query: str, max_results: int = RANK_POOL_SIZE,
                   engines: str | None = SEARXNG_PLAYLIST_ENGINES) -> list[dict]:
    base = (cfg("SEARXNG_URL") or "").rstrip("/")
    if "<" in base or not base:
        raise RuntimeError(f"SEARXNG_URL не задан ({base!r}) — "
                           "вызовите L.configure(SEARXNG_URL=...)")
    params = {"q": query, "format": "json", "language": cfg("SEARXNG_LANGUAGE")}
    if engines:
        params["engines"] = engines
    try:
        resp = httpx.get(
            f"{base}/search", params=params, timeout=15,
            headers={
                "Accept": "application/json, text/javascript, */*",
                "Referer": f"{base}/",
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:max_results]
        if results:
            return results
    except Exception as exc:
        print(f"[searxng] {type(exc).__name__}: {exc} → DDG")
    return search_ddg(query, max_results)


def search_ddg(query: str, max_results: int = 10) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
        return [{"title": r["title"], "url": r["href"], "content": r["body"]} for r in rows]
    except Exception as exc:
        print(f"[ddg] failed: {exc}")
        return []


# ── 2. РЕРАНК ────────────────────────────────────────────────────────────────

JUNK_URL = re.compile(
    r"""(?ix)
      genius\.com/artists/
    | genius\.com/[^/]+-annotated
    | genius\.com/[^/?#]+-lyrics\b
    | //(?:www\.)?instagram\.com
    | //(?:www\.)?facebook\.com
    | ticketmaster\.
    | (?:www\.|music\.)?youtube\.com/(?:channel|@|watch|playlist)
    | /tickets?\b
    | //(?:www\.)?open\.spotify\.com
    """
)

AUTHORITY = {
    "billboard.com": 2.8, "rollingstone.com": 2.6, "pitchfork.com": 2.4,
    "udiscovermusic.com": 2.2, "nme.com": 2.1, "complex.com": 2.0,
    "classicpopmag.com": 2.0, "top40weekly.com": 1.9, "albumoftheyear.org": 1.8,
    "last.fm": 1.6, "discogs.com": 1.5,
    "musicbrainz.org": 1.3,
}

LIST_PATH = re.compile(
    r"(?i)(discograph|tracklist|singles|greatest|best[-_]?songs|top[-_]?\d|"
    r"list[-_]?of|/albums/|soundtrack)")


def authority_weight(url: str) -> float:
    low = url.lower()
    if "genius.com/albums/" in low:
        return 2.6
    host = (urlparse(url).netloc or "").lower()
    if host.endswith(".fandom.com") or host == "fandom.com":
        return 2.3
    if host.endswith("wikipedia.org"):
        return 3.0 if host in ("en.wikipedia.org", "en.m.wikipedia.org") else 1.5
    for domain, weight in AUTHORITY.items():
        if host == domain or host.endswith("." + domain):
            return weight
    return 0.0


def rank_results(results: list[dict], query: str = "") -> list[dict]:
    """Реранк списочных запросов. Возвращает результаты со score в `_score`."""
    scored = []
    for idx, r in enumerate(results):
        url = r.get("url") or ""
        if not url or JUNK_URL.search(url):
            continue
        score = (authority_weight(url)
                 + (0.4 if LIST_PATH.search(url) else 0.0)
                 + 0.1 / (1.0 + idx))
        scored.append((score, {**r, "_score": round(score, 3), "_orig_pos": idx}))
    if not scored:
        return results
    scored.sort(key=lambda t: t[0], reverse=True)
    out, per_host = [], {}
    for _, r in scored:
        host = (urlparse(r["url"]).netloc or "").lower()
        if per_host.get(host, 0) >= 2:
            continue
        per_host[host] = per_host.get(host, 0) + 1
        out.append(r)
    return out


# ── 3. ФЕТЧ СТРАНИЦЫ ─────────────────────────────────────────────────────────

def _fetch_trafilatura(url: str, timeout: float) -> str:
    """trafilatura.fetch_url — свой пул urllib3 с дефолтными заголовками; берёт
    часть сайтов, на которых curl_cffi получает 403. Возвращает None при отказе,
    исключений не бросает."""
    import trafilatura
    return trafilatura.fetch_url(url)


def _fetch_curl_plain(url: str, timeout: float) -> str:
    """Голый curl_cffi без impersonate — как ходит genius_service."""
    from curl_cffi import requests as curl_requests
    resp = curl_requests.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _fetch_curl_chrome124(url: str, timeout: float) -> str:
    """Полный браузерный отпечаток — единственное, что открывает Fandom."""
    from curl_cffi import requests as curl_requests
    resp = curl_requests.get(url, timeout=timeout, allow_redirects=True,
                             impersonate="chrome124")
    resp.raise_for_status()
    return resp.text


def _fetch_httpx(url: str, timeout: float) -> str:
    resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")})
    resp.raise_for_status()
    return resp.text


FETCHERS = {
    "trafilatura": _fetch_trafilatura,
    "curl_plain": _fetch_curl_plain,
    "curl_chrome124": _fetch_curl_chrome124,
    "httpx": _fetch_httpx,
}

# Порядок перебора для ВЕБ-ПОИСКА. Genius этим путём не ходит — у него свой
# фетчер в genius_service (голый curl: подделанный отпечаток он блокирует).
FETCH_ORDER = ("trafilatura", "curl_plain", "curl_chrome124")

MIN_HTML_CHARS = 500        # короче — это заглушка/челлендж, а не страница
LAST_FETCH: dict = {}       # {"url", "fetcher", "chars", "tried": [...]}
FETCH_STATS: dict = {}      # {"trafilatura": 12, "curl_chrome124": 3, …}


def http_get_text(url: str, timeout: float = 12.0, order=None,
                  verbose: bool = False) -> str:
    """HTML страницы перебором фетчеров: первый, вернувший непустое тело, выигрывает.

    Кто именно сработал — в ``LAST_FETCH``; накопленная статистика — в
    ``FETCH_STATS`` (видно, какой фетчер реально тащит на вашей выдаче).
    """
    tried = []
    for name in (order or FETCH_ORDER):
        fetcher = FETCHERS.get(name)
        if fetcher is None:
            raise KeyError(f"нет фетчера {name!r}; есть: {list(FETCHERS)}")
        try:
            html = fetcher(url, timeout)
        except Exception as exc:
            tried.append(f"{name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if html and len(html) >= MIN_HTML_CHARS:
            tried.append(f"{name}: OK {len(html)}")
            LAST_FETCH.clear()
            LAST_FETCH.update(url=url, fetcher=name, chars=len(html), tried=tried)
            FETCH_STATS[name] = FETCH_STATS.get(name, 0) + 1
            if verbose:
                print(f"[fetcher] {name} -> {len(html)} chars")
            return html
        tried.append(f"{name}: пусто/{len(html or '')} симв.")
    LAST_FETCH.clear()
    LAST_FETCH.update(url=url, fetcher=None, chars=0, tried=tried)
    raise RuntimeError("все фетчеры отказали: " + "; ".join(tried))


# ── 3b. ИЗВЛЕЧЕНИЕ ТЕКСТА (независимо от фетчера) ────────────────────────────
# Фетчер и экстрактор ортогональны: bare_extraction принимает HTML-строку/байты,
# так что тело от curl_cffi разбирается тем же trafilatura, что и от fetch_url.
EXTRACTOR = "trafilatura"      # "trafilatura" (markdown + метаданные) | "readability"


def extract_readability(html, url: str = None) -> dict:
    """Боевой путь прода: readability → plain text. Метаданных нет."""
    from bs4 import BeautifulSoup
    from readability import Document

    soup = BeautifulSoup(Document(html).summary(), "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return {"text": "\n".join(ln for ln in text.splitlines() if ln.strip()),
            "meta": {}}


def extract_trafilatura_plain(html, url: str = None) -> dict:
    """trafilatura «как есть»: markdown-разметка + метаданные.

    Таблицы отдаёт встроенный сериализатор trafilatura: строки приезжают как
    «| Artist | Title |», разделитель `|` попадает под ``_SEP_RE`` в
    ``tracklines()``. Оставлен для сравнения с ``extract_trafilatura`` — видно,
    что именно даёт свой конвертер таблиц.
    """
    from trafilatura import bare_extraction
    from trafilatura.xml import xmltotxt

    doc = bare_extraction(html, url=url, with_metadata=True,
                          include_formatting=True, include_tables=True,
                          include_comments=False)
    if doc is None:
        return {"text": "", "meta": {}}
    md = xmltotxt(doc.body, include_formatting=True)
    meta = doc.as_dict()
    for k in ("body", "commentsbody"):
        meta.pop(k, None)          # lxml-элементы не сериализуются
    meta["text"] = md
    return {"text": md, "meta": meta}


# — свой конвертер таблиц —
# trafilatura режет вики-таблицы: теряет colspan, вклеивает сноски [1] прямо в
# ячейку, а широкие таблицы иногда выбрасывает целиком. Поэтому таблицы
# вынимаются ДО извлечения, конвертируются сами, а на их место в дерево кладётся
# текстовый плейсхолдер, который потом меняется обратно на markdown.

DROP = ('.//sup[contains(@class,"reference")] | .//style | .//script'
        ' | .//sup[@class="noprint"] | .//span[contains(@class,"mw-editsection")]')

META_FIELDS = ("title", "author", "url", "hostname", "description", "sitename",
               "date", "categories", "tags", "license", "id", "fingerprint",
               "language", "image", "pagetype")

SKIP_TABLE_CLASSES = ("toc", "navbox", "infobox")


def _drop_keep_tail(el) -> None:
    """Удалить элемент, сохранив его хвостовой текст.

    Голый ``parent.remove(el)`` уносит и ``el.tail`` — для `<sup class=reference>`
    посреди ячейки это съедает весь текст ПОСЛЕ сноски.
    """
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail or ""
    if tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


def _colspan(cell) -> int:
    try:
        return max(1, int(cell.get("colspan") or 1))
    except (TypeError, ValueError):
        return 1               # colspan="100%" и прочая вёрстка руками


def _cell(c) -> str:
    for bad in c.xpath(DROP):
        _drop_keep_tail(bad)
    for br in c.xpath('.//br'):
        # без этого text_content() склеивает слова: вики переносит длинные
        # названия внутри узких колонок («Human After<br/>All» → «Human AfterAll»)
        br.tail = " " + (br.tail or "")
    return " ".join(c.text_content().split()).replace("|", r"\|")


def _table_to_md(table) -> str:
    """<table> → markdown-таблица. colspan разворачивается в пустые ячейки,
    rowspan игнорируется (в треклистах не встречается)."""
    grid = []
    for tr in table.xpath('.//tr'):
        row = []
        for c in tr.xpath('./th|./td'):
            row.append(_cell(c))
            row += [""] * (_colspan(c) - 1)
        if row:
            grid.append(row)
    if not grid:
        return ""
    w = max(map(len, grid))
    grid = [r + [""] * (w - len(r)) for r in grid]
    head, *body = grid
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * w) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _meta_to_dict(doc) -> dict:
    """bare_extraction в разных версиях отдаёт dict или объект Document."""
    if isinstance(doc, dict):
        return {k: doc.get(k) for k in META_FIELDS if k in doc}
    if hasattr(doc, "as_dict"):
        d = doc.as_dict()
        return {k: d.get(k) for k in META_FIELDS if k in d}
    return {k: getattr(doc, k, None) for k in META_FIELDS}


def extract_trafilatura(html, url: str = None,
                        skip_table_classes=SKIP_TABLE_CLASSES) -> dict:
    """trafilatura + свой конвертер таблиц. HTML → {"text": markdown, "meta": …}.

    Таблицы конвертируются отдельно и вклеиваются обратно на своё место в
    тексте; метаданные берутся из trafilatura. ``skip_table_classes`` выкидывает
    служебную вики-обвязку (оглавление, навбоксы, инфобокс) — она даёт шум в
    ``tracklines()``.
    """
    from lxml import html as LH
    from trafilatura import bare_extraction
    from trafilatura.xml import xmltotxt

    tree = html if hasattr(html, "xpath") else LH.fromstring(html)

    tables, placeholders = [], []
    for tb in tree.xpath('//table'):
        if tb.xpath('ancestor::table'):
            continue                      # вложенная — уедет вместе с внешней
        cls = tb.get("class") or ""
        if any(c in cls for c in skip_table_classes):
            continue
        parent = tb.getparent()
        if parent is None:
            continue
        md_tbl = _table_to_md(tb)
        if not md_tbl:
            continue
        # плейсхолдер длинный, иначе trafilatura отрежет его как слишком короткий абзац
        key = f"XTABLEPLACEHOLDERX{len(tables)}X" + " placeholder" * 8
        placeholders.append(key)
        tables.append(md_tbl)
        parent.replace(tb, LH.fromstring(f"<p>{key}</p>"))

    doc = bare_extraction(
        LH.tostring(tree, encoding="unicode"),
        url=url,
        with_metadata=True,
        include_formatting=True,
        include_tables=False,
        include_comments=False,
        favor_recall=True,
    )
    if doc is None:
        return {"text": "", "meta": {}}

    body = doc["body"] if isinstance(doc, dict) else doc.body
    md = xmltotxt(body, include_formatting=True) if body is not None else ""
    for key, tbl in zip(placeholders, tables):
        md = md.replace(key, "\n\n" + tbl + "\n\n")

    meta = _meta_to_dict(doc)
    meta["tables"] = len(tables)
    return {"text": md.strip(), "meta": meta}


def extract_md_with_meta(html, url=None, skip_table_classes=SKIP_TABLE_CLASSES):
    """HTML -> (markdown, metadata). Кортежная обёртка над ``extract_trafilatura``."""
    out = extract_trafilatura(html, url, skip_table_classes)
    return out["text"], out["meta"]


EXTRACTORS = {"readability": extract_readability,
              "trafilatura": extract_trafilatura,
              "trafilatura_plain": extract_trafilatura_plain}


def extract_page(html, url: str = None, mode: str = None) -> dict:
    """HTML → {"text", "meta"} выбранным экстрактором, с откатом на readability."""
    mode = mode or cfg("EXTRACTOR") or "readability"
    try:
        return EXTRACTORS[mode](html, url)
    except ImportError as exc:
        if mode == "readability":
            raise
        print(f"[extract] {mode} недоступен ({exc}) -> readability")
        return extract_readability(html, url)


def fetch_page(url: str, max_chars: int = PAGE_FETCH_CHARS, order=None,
               extractor: str = None) -> str:
    """Скачать и извлечь текст. Ошибки возвращаются как 'Error: …'."""
    try:
        html = http_get_text(url, order=order)
        return extract_page(html, url, extractor)["text"][:max_chars]
    except Exception as exc:
        return f"Error: {exc}"


LAST_EXTRACT: dict = {}     # мета последнего вызова md() — {"url","fetcher","meta"}


def md(url: str, order=None, extractor: str = None, quiet: bool = False) -> str:
    """Чистый markdown страницы. Только фетч + извлечение, без треклистов.

        text = L.md(url)                          # каскад фетчеров + trafilatura
        text = L.md(url, order=("curl_plain",))   # конкретный фетчер
        text = L.md(url, extractor="readability") # сравнить с боевым путём
        L.LAST_EXTRACT["meta"]                    # title/author/date/sitename/…
    """
    html = http_get_text(url, order=order)
    out = extract_page(html, url, extractor)
    LAST_EXTRACT.clear()
    LAST_EXTRACT.update(url=url, fetcher=LAST_FETCH.get("fetcher"),
                        html_chars=len(html), meta=out["meta"])
    if not quiet:
        print(f"[md] {LAST_FETCH.get('fetcher')}: {len(html)} симв. HTML -> "
              f"{len(out['text'])} симв. markdown")
    return out["text"]


def probe(url: str, catalog: "Catalog" = None, order=None, extractor: str = None,
          show: int = 15) -> dict:
    """Разобрать ОДНУ страницу: HTTP → извлечение → строки → библиотека.

        L.probe("https://en.wikipedia.org/wiki/Kanye_West_discography", cat)
        L.probe(url, order=("curl_plain",))          # только голый curl, как Genius
        L.probe(url, extractor="readability")        # сравнить с боевым путём

    Возвращает {"html", "text", "meta", "lines", "matches", "fetcher"}.
    """
    html = http_get_text(url, order=order)
    print(f"[probe] фетчер {LAST_FETCH['fetcher']}, {len(html)} симв. HTML"
          + (f"  (перебор: {' | '.join(LAST_FETCH['tried'])})"
             if len(LAST_FETCH["tried"]) > 1 else ""))

    mode = extractor or cfg("EXTRACTOR")
    out = extract_page(html, url, mode)
    text, meta = out["text"], out["meta"]
    lines = tracklines(text)
    print(f"[probe] {mode}: {len(text)} симв. -> {len(lines)} строк треклиста "
          f"({'list' if len(lines) >= MIN_TRACK_LINES else 'проза'})")
    if meta:
        print("[probe] мета: " + ", ".join(
            f"{k}={meta[k]!r}" for k in ("title", "author", "date", "sitename")
            if meta.get(k)))
    for ln in lines[:show]:
        print("   ", ln)

    matches = resolve_tracklines(lines, catalog) if (catalog and lines) else []
    if matches:
        print(f"[probe] в библиотеке {len(matches)}: "
              + "; ".join(f"{m['artist']} — {m['title']}" for m in matches[:10]))
    return {"html": html, "text": text, "meta": meta, "lines": lines,
            "matches": matches, "fetcher": LAST_FETCH.get("fetcher")}


def compare_extractors(url: str, catalog: "Catalog" = None, order=None) -> dict:
    """Один HTML, оба экстрактора — сколько строк треклиста и матчей даёт каждый."""
    html = http_get_text(url, order=order)
    out = {}
    for mode in EXTRACTORS:
        try:
            text = extract_page(html, url, mode)["text"]
        except Exception as exc:
            print(f"{mode:<14} ошибка: {exc}")
            continue
        lines = tracklines(text)
        matches = resolve_tracklines(lines, catalog) if (catalog and lines) else []
        out[mode] = {"chars": len(text), "lines": lines, "matches": matches}
        print(f"{mode:<14} {len(text):>7} симв.  {len(lines):>4} строк  "
              f"{len(matches):>3} в библиотеке")
    return out


def is_readable(content: str) -> bool:
    if not content or content.startswith("Error:"):
        return False
    low = content.lower()
    if any(m in low for m in ("javascript is required", "verifying your browser",
                              "enable javascript", "captcha")):
        return False
    return len(content) >= 400


# ── 4. ИЗВЛЕЧЕНИЕ ТРЕКЛИСТА ──────────────────────────────────────────────────

_SEP_RE = re.compile(r"\s[—–\-|]\s")
_LETTER_RE = re.compile(r"[^\W\d_]")
_CELL_DASH_EOL_RE = re.compile(r"[—–\-|]\s*$")
_YEAR_LINE_RE = re.compile(r"^\(\d{4}\)$")
_NUM_PREFIX_RE = re.compile(r"^\d{1,3}[.)]\s+")


def tracklines(text: str | None) -> list[str]:
    """Все строки вида «Artist — Title» со страницы (склейка вики-ячеек + дедуп)."""
    if not text:
        return []
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]

    merged, i = [], 0
    while i < len(raw):
        line = raw[i]
        if _CELL_DASH_EOL_RE.search(line) and i + 1 < len(raw):
            line = f"{line} {raw[i + 1]}"
            i += 2
            if i < len(raw) and _YEAR_LINE_RE.match(raw[i]):
                line = f"{line} {raw[i]}"
                i += 1
        else:
            i += 1
        merged.append(line)

    kept, seen = [], set()
    for ln in merged:
        if len(ln) > TRACK_LINE_MAX_LEN:
            continue
        core = _NUM_PREFIX_RE.sub("", ln)
        m = _SEP_RE.search(core)
        if not m:
            continue
        if not _LETTER_RE.search(core[:m.start()]) or not _LETTER_RE.search(core[m.end():]):
            continue
        key = ln.casefold()
        if key not in seen:
            seen.add(key)
            kept.append(ln)
    return kept


def sample_lines(kept: list[str], max_chars: int) -> str:
    """Равномерная выборка строк под бюджет (префикс отрезал бы одну радиостанцию)."""
    out = "\n".join(kept)
    if len(out) <= max_chars or not kept:
        return out
    avg = max(1, len(out) // len(kept))
    budget = max(MIN_TRACK_LINES, (max_chars - 120) // avg)
    if budget >= len(kept):
        return out
    step = (len(kept) - 1) / (budget - 1)
    sampled = [kept[round(i * step)] for i in range(budget)]
    header = (f"(showing {len(sampled)} of {len(kept)} track lines, sampled evenly)")
    return (header + "\n" + "\n".join(sampled))[:max_chars]


# ── 5. НОРМАЛИЗАЦИЯ + КАТАЛОГ ────────────────────────────────────────────────

def fold(text: str) -> str:
    if not text:
        return ""
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).lower().split())


_CYR_TO_LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
               "ж": "zh", "з": "z", "и": "i", "к": "k", "л": "l", "м": "m",
               "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
               "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh",
               "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
               "я": "ya"}


def _translit(s: str) -> str:
    return "".join(_CYR_TO_LAT.get(c, c) for c in s)


def similar(a: str, b: str) -> float:
    """Кросс-скриптовое сходство имён (упрощённый name_match.score_names)."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0.0
    return max(SequenceMatcher(None, x, y).ratio()
               for x in {fa, _translit(fa)} for y in {fb, _translit(fb)})


_FEAT = {"feat", "ft", "featuring"}


def title_key(s: str) -> str:
    toks = fold(s).split()
    for i, t in enumerate(toks):
        if t in _FEAT:
            return " ".join(toks[:i]) or " ".join(toks)
    return " ".join(toks)


def load_prod_bm25(repo_root: str = None):
    """Боевой ``catalog_search_service`` (build_index/search_tracks), если рядом
    лежит репозиторий. Модуль грузится ПО ФАЙЛУ: обычный импорт ``app.services``
    утащил бы за собой ``app.resources`` → torch, которого на ноутбуке нет.
    Возвращает модуль или None."""
    import importlib.util
    import types

    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    services = os.path.join(root, "app", "services")
    if not os.path.isfile(os.path.join(services, "catalog_search_service.py")):
        return None
    try:
        pkg = types.ModuleType("labcss")
        pkg.__path__ = [services]           # чтобы сработал `from .text_normalize`
        sys.modules["labcss"] = pkg
        mod = None
        for name in ("text_normalize", "catalog_search_service"):
            spec = importlib.util.spec_from_file_location(
                f"labcss.{name}", os.path.join(services, f"{name}.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"labcss.{name}"] = mod
            spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f"[catalog] боевой BM25F не загрузился: {exc}")
        return None


_PAYLOAD_FIELDS = ["track_id", "title", "artist", "album", "year", "cover_art_path"]


def load_songs_from_qdrant(url: str = None, collection: str = None,
                           api_key: str = None) -> tuple[str, list[tuple]]:
    """Scroll всей коллекции → (имя коллекции, [(track_id, payload), …]).

    Формат точек — ровно тот, что боевой ``build_index`` получает из
    ``light_points``, так что его можно скормить настоящему BM25F.
    """
    from qdrant_client import QdrantClient

    url = (url or cfg("QDRANT_URL") or "").rstrip("/")
    if "<" in url or not url:
        raise RuntimeError(f"QDRANT_URL не задан ({url!r}) — "
                           "вызовите L.configure(QDRANT_URL=...)")
    client = QdrantClient(url=url, api_key=api_key or cfg("QDRANT_API_KEY") or None,
                          timeout=60)
    collection = collection or cfg("COLLECTION")
    if not collection:
        accts = [c.name for c in client.get_collections().collections
                 if c.name.startswith("acct_")]
        if not accts:
            raise RuntimeError("нет коллекций acct_* — задайте COLLECTION")
        collection = max(accts, key=lambda n: client.count(n, exact=True).count)
        print(f"[qdrant] коллекция не задана → выбрана {collection}")

    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=_PAYLOAD_FIELDS, with_vectors=False,
        )
        for p in batch:
            pl = p.payload or {}
            points.append((pl.get("track_id") or str(p.id), pl))
        if offset is None:
            break
    return collection, points


class Catalog:
    """Библиотека для сверки. Источник — Qdrant сервера (с кэшем на диск).

        cat = Catalog()                          # QDRANT_URL + COLLECTION из env
        cat = Catalog(url="http://1.2.3.4:6333", collection="acct_3")
        cat = Catalog(refresh=True)              # игнорировать кэш

    Fuzzy-поиск: если рядом лежит репозиторий (``app.services``) — используется
    НАСТОЯЩИЙ BM25F ``catalog_search_service``; иначе difflib-заглушка.
    """

    def __init__(self, url: str = None, collection: str = None, api_key: str = None,
                 cache: str = CATALOG_CACHE, refresh: bool = False):
        self.points: list[tuple] = []
        self.collection = collection or COLLECTION
        if not refresh and cache and os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                blob = json.load(f)
            self.collection, self.points = blob["collection"], [tuple(p) for p in blob["points"]]
            print(f"[catalog] из кэша {cache}: {len(self.points)} треков ({self.collection})")
        else:
            self.collection, self.points = load_songs_from_qdrant(url, collection, api_key)
            print(f"[catalog] из Qdrant: {len(self.points)} треков ({self.collection})")
            if cache:
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({"collection": self.collection, "points": self.points},
                              f, ensure_ascii=False)

        self.songs = [{"track_id": tid, "title": pl.get("title") or "",
                       "artist": pl.get("artist") or "", "year": pl.get("year")}
                      for tid, pl in self.points]
        self.by_title = {}
        for s in self.songs:
            self.by_title.setdefault(title_key(s["title"]), []).append(s)

        self._index = None          # ленивый боевой BM25F-индекс
        self._bm25 = load_prod_bm25()
        print("[catalog] fuzzy: "
              + ("боевой BM25F (catalog_search_service)" if self._bm25
                 else "difflib-заглушка (боевой модуль недоступен)"))

    def __len__(self):
        return len(self.songs)

    def fuzzy(self, title: str, limit: int = 3) -> list[dict]:
        q = title_key(title)
        if len(q) < 2:
            return []
        if self._bm25 is not None:
            if self._index is None:
                self._index = self._bm25.build_index(self.points)
            return self._bm25.search_tracks(self._index, q, limit=limit)
        scored = ((similar(q, s["title"]), s) for s in self.songs)
        return [s for sc, s in sorted(scored, key=lambda t: -t[0])[:limit] if sc >= 0.6]


_LINE_SPLIT_RE = re.compile(r"\s(?:—|–|\||-)\s")
_LINE_NUM_RE = re.compile(r"^\d{1,3}[.)]\s+")
_LINE_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_QUOTES = "\"'«»“”‘’„"


def parse_track_line(line: str) -> dict | None:
    if not line:
        return None
    s = _LINE_YEAR_RE.sub("", _LINE_NUM_RE.sub("", line.strip()))
    parts = _LINE_SPLIT_RE.split(s, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (p.strip().strip(_QUOTES).strip() for p in parts)
    if not left or not right or len(left) > 80 or len(right) > 120:
        return None
    return {"artist": left, "title": right}


def _artist_matches(q: str, s: str) -> bool:
    na, ns = fold(q or ""), fold(s or "")
    return bool(na) and bool(ns) and (na in ns or ns in na)


def resolve_tracklines(lines: list[str], catalog: Catalog, max_fuzzy: int = 120) -> list[dict]:
    """Пересечение извлечённого треклиста с библиотекой (обе ориентации строки)."""
    parsed, seen = [], set()
    for ln in lines:
        p = parse_track_line(ln)
        if not p:
            continue
        key = (fold(p["artist"]), fold(p["title"]))
        if key in seen or (key[1], key[0]) in seen:
            continue
        seen.add(key)
        ym = re.search(r"\((\d{4})\)", ln)
        p["year"] = ym.group(1) if ym else None
        parsed.append(p)

    out, out_ids, fuzzy_left = [], set(), max_fuzzy
    for p in parsed:
        picked, mode = None, "none"
        for artist, title in ((p["artist"], p["title"]), (p["title"], p["artist"])):
            picked = next((c for c in catalog.by_title.get(title_key(title), [])
                           if _artist_matches(artist, c.get("artist"))), None)
            if picked:
                mode = "exact"
                break
        if not picked and fuzzy_left > 0:
            fuzzy_left -= 1
            qk = title_key(p["title"])
            for h in catalog.fuzzy(p["title"]):
                hk = title_key(h.get("title") or "")
                title_ok = (qk and hk and qk in hk) or similar(p["title"], h.get("title")) >= 0.75
                artist_ok = (not p["artist"] or _artist_matches(p["artist"], h.get("artist"))
                             or similar(p["artist"], h.get("artist")) >= 0.75)
                if title_ok and artist_ok:
                    picked, mode = h, "fuzzy"
                    break
        if not picked or picked.get("track_id") in out_ids:
            continue
        out_ids.add(picked["track_id"])
        out.append({"track_id": picked["track_id"], "title": picked.get("title"),
                    "artist": picked.get("artist"), "match": mode,
                    "year": p.get("year") or picked.get("year")})
    return out


def extract_year_range(wish: str) -> tuple[int, int] | None:
    if not wish:
        return None
    low = wish.lower()
    m = re.search(r"(?:после|after|since|начиная с)\s+((?:19|20)\d{2})", low)
    if m:
        bump = 0 if ("начиная" in m.group(0) or "since" in m.group(0)) else 1
        return (int(m.group(1)) + bump, 3000)
    m = re.search(r"(?:до|before)\s+((?:19|20)\d{2})", low)
    if m:
        return (1900, int(m.group(1)) - 1)
    m = re.search(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b", low)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:^|\s)['’]?(\d0)(?:-?х|s|х)(?:\s|$|[,.!?])", low)
    if m:
        dec = int(m.group(1))
        base = 1900 + dec if dec >= 30 else 2000 + dec
        return (base, base + 9)
    return (2000, 2009) if re.search(r"нулев", low) else None


# ── 6. ГЛАВНАЯ ФУНКЦИЯ: ПОИСК → РЕРАНК → ФЕТЧ → ИЗВЛЕЧЕНИЕ ───────────────────

def web_search(query: str, fetch_content: bool = False, max_results: int = 5,
               catalog: Catalog | None = None, verbose: bool = True) -> dict:
    """Один прогон боевой ветки rank="playlist".

    Возвращает всё, что интересно для отладки:
      {"prompt": текст для LLM, "lines": все строки треклистов,
       "matches": попадания в библиотеку, "pages": лог по каждой странице,
       "ranked": ранжированный пул}
    """
    pool = search_searxng(query)
    ranked = rank_results(pool, query)
    if verbose:
        print(f"[search] {query!r}: пул {len(pool)} → {len(ranked)} после реранка")
        for r in ranked[:8]:
            print(f"   {r.get('_score', 0):>5} {urlparse(r['url']).netloc:<28} {r['title'][:60]}")
    if not ranked:
        return {"prompt": "No results found", "lines": [], "matches": [],
                "pages": [], "ranked": []}

    want_full = 3 if fetch_content else 1
    rows, pages, all_lines, full_got, tries = [], [], [], 0, 0
    for r in ranked:
        url, title = r.get("url", ""), r.get("title", "")
        snippet = r.get("content", "") or ""
        if url and full_got < want_full and tries < MAX_FETCH_TRIES:
            tries += 1
            body = fetch_page(url)
            readable = is_readable(body)
            lines = tracklines(body) if readable else []
            pages.append({"url": url, "chars": len(body), "readable": readable,
                          "lines": len(lines), "fetcher": LAST_FETCH.get("fetcher"),
                          "tried": list(LAST_FETCH.get("tried") or []),
                          "verdict": ("list" if len(lines) >= MIN_TRACK_LINES
                                      else "prose" if readable else "unreadable")})
            if verbose:
                print(f"[fetch] {pages[-1]['verdict']:<10} "
                      f"{str(pages[-1]['fetcher']):<15} lines={len(lines):<4} {url[:70]}")
            if readable and len(lines) >= MIN_TRACK_LINES:
                all_lines.extend(lines)
                body_for_model = (
                    f"(tracklist page: {len(lines)} track lines extracted and "
                    "auto-checked against the library — see LIBRARY MATCHES. "
                    "Do NOT copy lines into get_songs. Sample:)\n"
                    + sample_lines(lines, MODEL_SAMPLE_CHARS))
                rows.append(("list", title, url, body_for_model, snippet))
                full_got += 1
                continue
            if readable:
                rows.append(("prose", title, url, body, snippet))
                continue
        rows.append(("snippet", title, url, "", snippet))

    prose_slots = want_full - full_got
    out = []
    for kind, title, url, body, snippet in rows:
        if kind == "list":
            out.append(f"### {title}\nURL: {url}\n\n{body}")
        elif kind == "prose" and prose_slots > 0:
            prose_slots -= 1
            out.append(f"### {title}\nURL: {url}\n\n{body[:4000]}")
        elif len(out) < max_results:
            out.append(f"### {title}\nURL: {url}\nSnippet: {snippet}")
    prompt = "\n\n---\n\n".join(out)

    matches = resolve_tracklines(all_lines, catalog) if (catalog and all_lines) else []
    if matches:
        section = "\n".join(f"M{i + 1}. {m['artist']} — {m['title']}"
                            + (f" ({m['year']})" if m.get("year") else "")
                            for i, m in enumerate(matches[:40]))
        prompt += ("\n\nLIBRARY MATCHES — these songs ARE in the user's library. "
                   "Put the keys (e.g. \"M3\") straight into track_ids:\n" + section)
    if verbose:
        print(f"[result] строк треклиста {len(all_lines)}, матчей {len(matches)}, "
              f"промпт {len(prompt)} симв.")
    return {"prompt": prompt, "lines": all_lines, "matches": matches,
            "pages": pages, "ranked": ranked}


# ── 7. LLM НА СЕРВЕРЕ ────────────────────────────────────────────────────────

def ask_llm(messages: list[dict], tools: list[dict] | None = None,
            temperature: float = 0.3, max_tokens: int = 3000) -> dict:
    """Один вызов OpenAI-совместимого эндпоинта. Возвращает message целиком."""
    base = (cfg("LLM_BASE_URL") or "").rstrip("/")
    if "<" in base or not base:
        raise RuntimeError(f"LLM_BASE_URL не задан ({base!r}) — "
                           "вызовите L.configure(LLM_BASE_URL=...)")
    payload = {"model": cfg("LLM_MODEL"), "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    resp = httpx.post(f"{base}/chat/completions", json=payload, timeout=300,
                      headers={"Authorization": f"Bearer {cfg('LLM_API_KEY')}"})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


# ── 8. АГЕНТНЫЙ ЦИКЛ (упрощённая копия playlist_agent) ───────────────────────

INSTRUCTIONS = """You build a music playlist from the user's request by finding real songs and matching them against THIS user's music library.

- ALWAYS call web_search first. fetch_content=true downloads full page text — use it whenever you need a LIST from a page (tracklist, chart, "best of" ranking); snippets almost never contain songs.
- When a web_search response contains a "LIBRARY MATCHES" section, those songs are ALREADY verified to be in the library. Put their keys (e.g. "M3") directly into track_ids — do NOT re-check them with get_songs. If LIBRARY MATCHES cover ~15 songs, finish.
- Song titles MUST come from web_search results, never from your own memory. Album/OST/compilation names are not songs.
- Never run two searches with near-identical wording.
- When done, answer with ONLY minified JSON: {"title": "...", "track_ids": ["M1", ...], "comment": "..."}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for real song titles (artist hits, soundtrack tracklists).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "fetch_content": {"type": "boolean",
                              "description": "true = download full page bodies"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_songs",
        "description": "Match proposed {title, artist} songs against the user's library.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "artist": {"type": "string"}}}},
        }, "required": ["items"]}}},
]

MAX_WEB_SEARCHES = 6
MAX_LLM_REQUESTS = 15


def run_agent(wish: str, catalog: Catalog, lang: str = "ru", seed: bool = True,
              verbose: bool = True) -> dict:
    """Полный прогон: seed-поиск кодом → цикл LLM с двумя инструментами.

    Возвращает {"draft": ..., "matches": {key: match}, "trace": [...]}.
    """
    automap: dict[str, dict] = {}      # "M1" → матч
    trace, searched = [], set()
    year_range = extract_year_range(wish)

    def _register(matches: list[dict]) -> str:
        if year_range:
            lo, hi = year_range
            matches = [m for m in matches
                       if not m.get("year") or lo <= int(m["year"]) <= hi]
        lines = []
        by_tid = {m["track_id"]: k for k, m in automap.items()}
        for m in matches:
            key = by_tid.get(m["track_id"]) or f"M{len(automap) + 1}"
            automap[key] = m
            by_tid[m["track_id"]] = key
            lines.append(f"{key}. {m['artist']} — {m['title']}"
                         + (f" ({m['year']})" if m.get("year") else ""))
        return "\n".join(lines[:40])

    def _web_search(query: str, fetch_content: bool = False) -> str:
        norm = " ".join(query.lower().split())
        if len(searched) >= MAX_WEB_SEARCHES:
            return "(web search limit reached — build the playlist from what you have.)"
        if norm in searched:
            return "(already searched exactly this — search a genuinely DIFFERENT angle.)"
        searched.add(norm)
        res = web_search(query, fetch_content, catalog=catalog, verbose=verbose)
        trace.append({"tool": "web_search", "query": query,
                      "fetch_content": fetch_content, "found": len(res["matches"])})
        text = res["prompt"]
        if res["matches"]:
            section = _register(res["matches"])
            text += ("\n\nLIBRARY MATCHES (verified in the library, use these keys):\n"
                     + section)
        return text

    def _get_songs(items: list[dict]) -> str:
        if not searched:
            return "(refused: call web_search first.)"
        lines = [f"{i.get('artist', '')} — {i.get('title', '')}" for i in items[:30]]
        matches = resolve_tracklines(lines, catalog)
        trace.append({"tool": "get_songs", "items": len(items), "found": len(matches)})
        if verbose:
            print(f"[get_songs] {len(items)} позиций → {len(matches)} в библиотеке")
        return _register(matches) or "(nothing matched the library)"

    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"[lang={lang}] {wish}"}]

    if seed:   # боевой seed: первый поиск делает КОД дословным текстом желания
        seed_text = _web_search(wish.strip(), True)
        keys = "; ".join(f"{k}. {m['artist']} — {m['title']}" for k, m in automap.items())
        messages.append({"role": "user", "content":
                         f"(A web search for this wish already ran. Results:\n{seed_text[:4000]}\n\n"
                         + (f"LIBRARY MATCHES so far: {keys})" if keys
                            else "Nothing matched the library yet — phrase your searches differently.)")})

    for step in range(MAX_LLM_REQUESTS):
        msg = ask_llm(messages, tools=TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            content = (msg.get("content") or "").strip()
            if verbose:
                print(f"[agent] финал на шаге {step + 1}: {content[:300]}")
            try:
                draft = json.loads(re.sub(r"^```(?:json)?|```$", "", content,
                                          flags=re.M).strip())
            except Exception:
                draft = {"raw": content}
            draft["tracks"] = [automap[k] for k in draft.get("track_ids", [])
                               if k in automap]
            return {"draft": draft, "matches": automap, "trace": trace}
        for call in calls:
            fn = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            if verbose:
                print(f"[agent] → {fn}({str(args)[:120]})")
            result = (_web_search(args.get("query", ""), bool(args.get("fetch_content")))
                      if fn == "web_search" else _get_songs(args.get("items") or []))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": fn, "content": result})

    return {"draft": {"track_ids": list(automap), "comment": "request limit"},
            "matches": automap, "trace": trace}
