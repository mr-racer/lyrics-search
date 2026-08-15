"""Getting readable text out of an arbitrary web page.

Two orthogonal halves, deliberately separate: a **fetcher cascade** that returns
HTML and an **extractor** that turns HTML into markdown. They do not know about
each other — the body from ``curl_cffi`` is parsed by exactly the same extractor
as a body from anywhere else — which is what lets a page be fetched one way and
read another (the MediaWiki API path and the Reddit feed path both hand their
HTML to the same extractor).

**Why trafilatura is not in the cascade.** It ships a perfectly good downloader,
and upstream in the lab it was the first fetcher tried. It cannot be used here:
``trafilatura.fetch_url`` builds its own ``urllib3.PoolManager``, and a
``PoolManager`` — unlike a ``ProxyManager`` — does not read proxy environment
variables. This deployment routes external outbound through ``HTTP_PROXY`` read
out of ``.env`` (docker-compose blanks it in the process environment on purpose,
so internal traffic stays direct), and there is no per-call way to hand that to
trafilatura. Its own fetcher is urllib3 with a User-Agent, i.e. roughly what
``curl_plain`` already does, so nothing is lost by keeping trafilatura as the
extractor alone and letting curl do the downloading.

Nothing here reads configuration. The proxy arrives as an argument, because a
resource that reaches into the settings layer is a resource you cannot test.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Shorter than this is a stub or a challenge, not a page.
MIN_HTML_CHARS = 500

# Page titles that mean "we served you a challenge, not the page". Matched on
# the TITLE alone, because that is the one part a bot wall always fills in and
# no real music page ever carries these words.
#
# This exists because of Reddit, and it is not a Reddit-specific problem. Asked
# for a thread from a blocked IP, Reddit answers **200 OK with 167 KB of
# markup** titled "Prove your humanity" — no comments, no post. Every signal the
# fetcher normally trusts says success, so without this the challenge page is
# chunked, embedded and ranked like any other source, and the only symptom is an
# answer built out of nothing. A 403 would have been kinder.
_BOT_WALL_TITLES = re.compile(
    r"""(?ix)
      prove\s+your\s+humanity
    | just\s+a\s+moment
    | attention\s+required
    | are\s+you\s+a\s+robot
    | verify\s+(?:you\s+are|your)\s+human
    | access\s+denied
    | security\s+check
    """
)


def looks_like_a_bot_wall(title: str) -> bool:
    return bool(title and _BOT_WALL_TITLES.search(title))


# ── the fetcher cascade ──────────────────────────────────────────────────────


def _fetch_curl_plain(url: str, timeout: float, proxies: Optional[dict]) -> str:
    """Bare curl_cffi, no impersonation — the way ``genius_service`` fetches."""
    from curl_cffi import requests as curl_requests

    resp = curl_requests.get(url, timeout=timeout, allow_redirects=True,
                             proxies=proxies or None)
    resp.raise_for_status()
    return resp.text


def _fetch_curl_chrome124(url: str, timeout: float, proxies: Optional[dict]) -> str:
    """A full browser fingerprint — the only thing Fandom and some CDNs answer."""
    from curl_cffi import requests as curl_requests

    resp = curl_requests.get(url, timeout=timeout, allow_redirects=True,
                             impersonate="chrome124", proxies=proxies or None)
    resp.raise_for_status()
    return resp.text


def _fetch_httpx(url: str, timeout: float, proxies: Optional[dict]) -> str:
    import httpx

    # httpx wants one URL string where requests wants a dict; prefer the https
    # mapping and fall back, so the `http=...` -only form still proxies.
    proxy = (proxies or {}).get("https") or (proxies or {}).get("http")
    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     proxy=proxy, headers={"User-Agent": _UA})
    resp.raise_for_status()
    return resp.text


FETCHERS = {
    "curl_plain": _fetch_curl_plain,
    "curl_chrome124": _fetch_curl_chrome124,
    "httpx": _fetch_httpx,
}

# Cheapest first. curl_plain answers for most of the open web; the browser
# fingerprint is what gets past Fandom-style CDN checks; httpx is the third
# opinion, and differs enough in TLS/header shape to occasionally win.
FETCH_ORDER = ("curl_plain", "curl_chrome124", "httpx")


def fetch_html(url: str, *, timeout: float = 12.0,
               proxies: Optional[dict] = None,
               order: Optional[tuple] = None) -> tuple[str, str]:
    """``(html, fetcher_name)`` — the first fetcher to return a real body wins.

    Raises ``RuntimeError`` naming every refusal when they all fail. The caller
    is expected to treat that as one lost page, not a lost run.
    """
    tried: list[str] = []
    for name in (order or FETCH_ORDER):
        fetcher = FETCHERS.get(name)
        if fetcher is None:
            raise KeyError(f"no such fetcher {name!r}; have: {list(FETCHERS)}")
        try:
            html = fetcher(url, timeout, proxies)
        except Exception as exc:  # noqa: BLE001 — try the next one
            tried.append(f"{name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if html and len(html) >= MIN_HTML_CHARS:
            return html, name
        tried.append(f"{name}: empty/{len(html or '')} chars")
    raise RuntimeError("every fetcher refused: " + "; ".join(tried))


# ── extraction ───────────────────────────────────────────────────────────────
# trafilatura mangles wiki tables: it loses colspan, glues reference markers
# ([1]) into the cell text, and sometimes drops a wide table entirely. So tables
# are lifted out BEFORE extraction, converted here, and a text placeholder is
# put in the tree in their place — then swapped back for markdown afterwards.

_DROP = ('.//sup[contains(@class,"reference")] | .//style | .//script'
         ' | .//sup[@class="noprint"] | .//span[contains(@class,"mw-editsection")]')

_META_FIELDS = ("title", "author", "url", "hostname", "description", "sitename",
                "date", "categories", "tags", "license", "id", "fingerprint",
                "language", "image", "pagetype")

# Wiki furniture: a table of contents, a navbox or an infobox is chrome, and it
# is chrome dense with exactly the proper nouns a tracklist parser looks for.
SKIP_TABLE_CLASSES = ("toc", "navbox", "infobox")


def _drop_keep_tail(el) -> None:
    """Remove an element but keep its tail text.

    A plain ``parent.remove(el)`` takes ``el.tail`` with it — for a
    ``<sup class=reference>`` sitting mid-cell that eats every word AFTER the
    footnote marker.
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
        return 1               # colspan="100%" and other hand-written markup


def _cell(c) -> str:
    for bad in c.xpath(_DROP):
        _drop_keep_tail(bad)
    for br in c.xpath('.//br'):
        # Without this ``text_content()`` welds words together: wikis break long
        # titles inside narrow columns ("Human After<br/>All" → "Human AfterAll").
        br.tail = " " + (br.tail or "")
    return " ".join(c.text_content().split()).replace("|", r"\|")


def _table_to_md(table) -> str:
    """``<table>`` → a markdown table.

    ``colspan`` expands into empty cells; ``rowspan`` is ignored (tracklists do
    not use it, and honouring it would need a second pass over the grid).
    """
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
    width = max(map(len, grid))
    grid = [r + [""] * (width - len(r)) for r in grid]
    head, *body = grid
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _meta_to_dict(doc) -> dict:
    """``bare_extraction`` returns a dict in some versions, a Document in others."""
    if isinstance(doc, dict):
        return {k: doc.get(k) for k in _META_FIELDS if k in doc}
    if hasattr(doc, "as_dict"):
        d = doc.as_dict()
        return {k: d.get(k) for k in _META_FIELDS if k in d}
    return {k: getattr(doc, k, None) for k in _META_FIELDS}


def extract_trafilatura(html: Any, url: Optional[str] = None,
                        skip_table_classes=SKIP_TABLE_CLASSES) -> dict:
    """HTML → ``{"text": markdown, "meta": {...}}``, tables converted by us."""
    from lxml import html as LH
    from trafilatura import bare_extraction
    from trafilatura.xml import xmltotxt

    tree = html if hasattr(html, "xpath") else LH.fromstring(html)

    tables, placeholders = [], []
    for tb in tree.xpath('//table'):
        if tb.xpath('ancestor::table'):
            continue                      # nested — it travels with the outer one
        cls = tb.get("class") or ""
        if any(c in cls for c in skip_table_classes):
            continue
        parent = tb.getparent()
        if parent is None:
            continue
        md_table = _table_to_md(tb)
        if not md_table:
            continue
        # The placeholder is long on purpose: trafilatura drops paragraphs it
        # considers too short, and a short marker disappears with them.
        key = f"XTABLEPLACEHOLDERX{len(tables)}X" + " placeholder" * 8
        placeholders.append(key)
        tables.append(md_table)
        parent.replace(tb, LH.fromstring(f"<p>{key}</p>"))

    doc = bare_extraction(
        LH.tostring(tree, encoding="unicode"), url=url, with_metadata=True,
        include_formatting=True, include_tables=False, include_comments=False,
        favor_recall=True,
    )
    if doc is None:
        return {"text": "", "meta": {}}

    body = doc["body"] if isinstance(doc, dict) else doc.body
    md = xmltotxt(body, include_formatting=True) if body is not None else ""
    for key, table in zip(placeholders, tables):
        md = md.replace(key, "\n\n" + table + "\n\n")

    meta = _meta_to_dict(doc)
    meta["tables"] = len(tables)
    return {"text": md.strip(), "meta": meta}


def extract_readability(html: Any, url: Optional[str] = None) -> dict:
    """The fallback: readability → plain text. No metadata, no tables."""
    from bs4 import BeautifulSoup
    from readability import Document

    soup = BeautifulSoup(Document(html).summary(), "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return {"text": "\n".join(ln for ln in text.splitlines() if ln.strip()),
            "meta": {}}


def extract_page(html: Any, url: Optional[str] = None) -> dict:
    """HTML → ``{"text", "meta"}``, falling back to readability.

    The fallback is for a missing or broken trafilatura only. A page trafilatura
    reads as empty is left empty — readability would not do better, and quietly
    swapping extractors per page makes two runs of the same query incomparable.
    """
    try:
        return extract_trafilatura(html, url)
    except ImportError as exc:
        logger.warning("[web_fetch] trafilatura unavailable (%s) — readability", exc)
        return extract_readability(html, url)


# ── appendix trimming ────────────────────────────────────────────────────────
# A MediaWiki article ends in an appendix — References, External links, Further
# reading — that is a third of its length and none of its meaning. Left in, it
# costs three ways: chunks made of nothing but citation lines get embedded and
# reranked like any other; those lines are dense with proper nouns and years,
# which is exactly what BM25 scores highly, so they out-rank prose on a name
# query; and it is all paid for twice, once per model.
#
# The match is deliberately narrow: a heading whose ENTIRE text is a known
# appendix title. "References" as a whole heading is the appendix on every
# article; "References to earlier work" is a section someone wrote on purpose,
# and a substring rule would eat it.

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

APPENDIX_HEADINGS = frozenset({
    "references", "reference", "notes", "notes and references",
    "citations", "footnotes", "sources", "bibliography",
    "further reading", "external links", "see also", "works cited",
    "примечания", "ссылки", "литература", "источники", "см. также",
    "внешние ссылки", "комментарии",
})


def _normalise_heading(text: str) -> str:
    text = re.sub(r"^\s*\d+(\.\d+)*[.)]?\s+", "", text or "")
    text = re.sub(r"[*_`\[\]]", "", text)
    return " ".join(text.lower().split()).rstrip(":").strip()


def strip_appendix(markdown: str) -> tuple[str, int]:
    """Drop everything from the first appendix heading on.

    Returns ``(markdown, characters_removed)``. The heading itself goes too — a
    lone "References" at the end of the last real chunk is noise in the embedded
    text.
    """
    if not markdown:
        return markdown, 0

    lines = markdown.split("\n")
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and _normalise_heading(match.group(2)) in APPENDIX_HEADINGS:
            kept = "\n".join(lines[:i]).rstrip()
            removed = len(markdown) - len(kept)
            # An appendix starting in the first fifth means we misread the page
            # — a stub, or a heading structure we do not understand. Keeping it
            # whole is the safer error.
            if len(kept) < len(markdown) * 0.2:
                logger.info("[web_fetch] appendix at %d%% of the page — left alone",
                            round(100 * len(kept) / max(len(markdown), 1)))
                return markdown, 0
            return kept, removed
    return markdown, 0
