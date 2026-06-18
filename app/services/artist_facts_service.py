"""Fetch and cache interesting artist facts from SongFacts.

Facts are stored in SQLite (``MetadataDB``) for structured querying.
A fallback to the legacy ``.txt`` cache is kept for backward compatibility.
"""

import asyncio
import logging
import re
from html import unescape
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..resources.metadata_db import MetadataDB
from app.services.proxy_config import get_proxy

logger = logging.getLogger(__name__)

# Legacy cache directory — kept for backward-compat fallback
_FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "facts"
REQUEST_TIMEOUT = 10  # seconds


def _slugify(artist: str) -> str:
    """'The Weeknd' -> 'the-weeknd'"""
    # Normalize all dash variants to ASCII hyphen.
    cleaned_artist = re.sub(r'[‐‑‒–—―−]', '-', artist)
    # Strip noisy punctuation (apostrophes — straight + curly, quotes, +, &, .)
    # so that artists like "Guns N' Roses" produce a stable slug regardless
    # of which apostrophe codepoint shows up in source metadata.
    cleaned_artist = re.sub(
        "[+&.'`‘’‚‛“”„‟′″ʼ«»]",
        '', cleaned_artist,
    )
    return "-".join(cleaned_artist.lower().split())


def _fetch_facts_html(artist: str) -> Optional[str]:
    """GET songfacts.com/facts/{slug} and return text, or None on failure."""
    slug = _slugify(artist)
    url = f"https://www.songfacts.com/facts/{slug}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, proxies=get_proxy())
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.debug("[ArtistFacts] Failed to fetch facts for %s: %s", artist, e)
        return None


def _parse_facts(html_string: str) -> List[str]:
    """Extract fact list items from SongFacts HTML."""
    soup = BeautifulSoup(html_string, "html.parser")
    container = soup.find("ul", class_="artistfacts-results")
    if not container:
        return []

    facts: List[str] = []
    for li in container.find_all("li"):
        inner_div = li.find("div", class_="inner")
        if inner_div:
            raw = inner_div.get_text(separator=" ", strip=True)
            cleaned = re.sub(r"\s+", " ", raw).strip()
            cleaned = unescape(cleaned)
            if cleaned:
                facts.append(cleaned)
    return facts


def _legacy_facts_path(collection_name: str, artist: str) -> Path:
    """Return path to legacy cached facts file for a collection + artist."""
    coll_dir = _FACTS_CACHE_DIR / collection_name
    return coll_dir / f"{_slugify(artist)}.txt"


def get_cached_facts(collection_name: str, artist: str) -> Optional[str]:
    """Read cached facts for an artist in a collection, or None.

    Prefers SQLite; falls back to legacy ``.txt`` files.
    """
    slug = _slugify(artist)
    try:
        facts = MetadataDB.get_artist_facts(slug, collection_name)
        if facts:
            return "\n\n".join(facts)
    except Exception as e:
        logger.debug("[ArtistFacts] SQLite read failed for %s: %s", slug, e)

    # Fallback to legacy .txt
    p = _legacy_facts_path(collection_name, artist)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _save_facts_to_sqlite(collection_name: str, artist: str, facts: List[str]) -> None:
    """Save parsed facts to SQLite."""
    slug = _slugify(artist)
    MetadataDB.add_artist_facts_batch(slug, collection_name, facts, source="songfacts.com")


async def fetch_artist_facts(
    artist: str,
    collection_name: str,
) -> Optional[str]:
    """Fetch facts for a single artist. Returns cached text or None."""
    cached = get_cached_facts(collection_name, artist)
    if cached:
        return cached

    html = await asyncio.to_thread(_fetch_facts_html, artist)
    if not html:
        return None

    facts = _parse_facts(html)
    if not facts:
        return None

    _save_facts_to_sqlite(collection_name, artist, facts)
    return "\n\n".join(facts)


async def fetch_facts_for_artists(
    artists: List[str],
    collection_name: str,
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str, bool], None]] = None,
) -> Dict[str, str]:
    """Fetch facts for multiple artists sequentially with delay between requests.

    Returns dict of artist -> facts text (only for artists that had facts).
    """
    results: Dict[str, str] = {}
    total = len(artists)
    for idx, artist in enumerate(artists, 1):
        text = await fetch_artist_facts(artist, collection_name)
        found = bool(text)
        if found:
            results[artist] = text
        if progress_callback:
            progress_callback(idx, total, artist, found)
        await asyncio.sleep(delay)
    return results


def load_all_facts_for_collection(
    collection_name: str,
    ai_enabled: bool = True,
) -> Dict[str, str]:
    """Load all cached facts for a collection.

    When ``ai_enabled`` is True, refined facts (from AI Indexing) take
    precedence over originals for any artist that has a refined row.
    An explicit empty refined list (AI ran but kept nothing) returns
    no entry for that artist — the original facts are suppressed.

    When ``ai_enabled`` is False, only original facts are returned.

    Prefers SQLite; falls back to legacy ``.txt`` files if SQLite is empty.
    """
    # Layer 1: originals from SQLite or legacy .txt
    originals: Dict[str, str] = {}
    try:
        originals = MetadataDB.get_all_artist_facts_by_collection(collection_name)
    except Exception as e:
        logger.debug("[ArtistFacts] SQLite collection read failed: %s", e)

    if not originals:
        coll_dir = _FACTS_CACHE_DIR / collection_name
        if coll_dir.is_dir():
            for f in coll_dir.iterdir():
                if f.suffix == ".txt":
                    originals[f.stem] = f.read_text(encoding="utf-8").strip()

    # Layer 2: if AI enabled, overlay refined facts on top
    if ai_enabled:
        try:
            refined = MetadataDB.get_all_refined_artist_facts(collection_name)
            if refined:
                # Start with originals, overlay refined where available
                merged = dict(originals)
                merged.update(refined)
                return merged
        except Exception as e:
            logger.debug("[ArtistFacts] refined facts read failed: %s", e)

    return originals
