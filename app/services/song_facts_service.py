"""Fetch and cache song facts from SongFacts.

Facts are stored in SQLite (``MetadataDB``) for structured querying.
A fallback to the legacy ``.txt`` cache is kept for backward compatibility.
"""

import asyncio
import logging
import re
from html import unescape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from ..resources.metadata_db import MetadataDB

logger = logging.getLogger(__name__)

# Legacy cache directory — kept for backward-compat fallback
_SONG_FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "songfacts"
REQUEST_TIMEOUT = 10


def _slugify(text: str) -> str:
    """'All Falls Down' -> 'all-falls-down'"""
    return "-".join(text.lower().split())


def _fetch_song_facts_html(artist: str, song: str) -> Optional[str]:
    """GET songfacts.com/facts/{artist}/{song} and return text, or None."""
    slug_artist = _slugify(artist)
    slug_song = _slugify(song)
    url = f"https://www.songfacts.com/facts/{slug_artist}/{slug_song}"
    try:
        logger.info("[SongFacts] Fetching facts for '%s — %s' (%s)", artist, song, url)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning("[SongFacts] Failed to fetch facts for '%s — %s': %s", artist, song, e)
        return None


def _parse_song_facts(html_string: str) -> List[str]:
    """Extract song fact list items from SongFacts HTML."""
    soup = BeautifulSoup(html_string, "html.parser")
    container = soup.find("ul", class_="songfacts-results")
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


def _legacy_song_facts_path(collection_name: str, artist: str, song: str) -> Path:
    """Return path to legacy cached song-facts file for a collection + song."""
    coll_dir = _SONG_FACTS_CACHE_DIR / collection_name
    safe_name = f"{_slugify(artist)}-{_slugify(song)}"
    return coll_dir / f"{safe_name}.txt"


def get_song_facts_key(artist: str, song: str) -> str:
    """Build the cache-key for a song (matches file stem / DB slug)."""
    return f"{_slugify(artist)}-{_slugify(song)}"


def get_cached_song_facts(collection_name: str, artist: str, song: str) -> Optional[str]:
    """Read cached song facts, or None.

    Prefers SQLite; falls back to legacy ``.txt`` files.
    """
    key = get_song_facts_key(artist, song)
    try:
        facts = MetadataDB.get_song_facts(key, collection_name)
        if facts:
            return "\n\n".join(facts)
    except Exception as e:
        logger.debug("[SongFacts] SQLite read failed for %s: %s", key, e)

    # Fallback to legacy .txt
    p = _legacy_song_facts_path(collection_name, artist, song)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _save_song_facts_to_sqlite(collection_name: str, artist: str, song: str, facts: List[str]) -> None:
    """Save parsed song facts to SQLite."""
    key = get_song_facts_key(artist, song)
    MetadataDB.add_song_facts_batch(key, collection_name, facts, source="songfacts.com")


async def fetch_song_facts(
    artist: str,
    song: str,
    collection_name: str,
) -> Optional[str]:
    """Fetch facts for a single song. Returns cached text or None."""
    cached = get_cached_song_facts(collection_name, artist, song)
    if cached:
        return cached

    html = await asyncio.to_thread(_fetch_song_facts_html, artist, song)
    if not html:
        return None

    facts = _parse_song_facts(html)
    if not facts:
        logger.info("[SongFacts] No facts found for '%s — %s'", artist, song)
        return None

    _save_song_facts_to_sqlite(collection_name, artist, song, facts)
    logger.info("[SongFacts] Cached %d facts for '%s — %s'", len(facts), artist, song)
    return "\n\n".join(facts)


async def fetch_facts_for_songs(
    songs: List[Tuple[str, str]],
    collection_name: str,
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, str]:
    """Fetch facts for multiple songs sequentially with delay between requests.

    Args:
        songs: list of (artist, song_title) tuples.
        collection_name: collection scope.
        delay: seconds between requests.
        progress_callback: optional callback(current, total, label) after each song.
    Returns:
        dict of '{artist} — {song}' -> facts text (only for songs that had facts).
    """
    results: Dict[str, str] = {}
    total = len(songs)
    for idx, (artist, song) in enumerate(songs, 1):
        key = f"{artist} — {song}"
        text = await fetch_song_facts(artist, song, collection_name)
        if text:
            results[key] = text
        if progress_callback:
            progress_callback(idx, total, f"{artist} — {song}")
        await asyncio.sleep(delay)
    return results


def load_all_song_facts_for_collection(collection_name: str) -> Dict[str, str]:
    """Load all cached song facts for a collection.

    Prefers SQLite; falls back to legacy ``.txt`` files if SQLite is empty.
    """
    try:
        facts = MetadataDB.get_all_song_facts_by_collection(collection_name)
        if facts:
            return facts
    except Exception as e:
        logger.debug("[SongFacts] SQLite collection read failed: %s", e)

    # Fallback to legacy .txt
    coll_dir = _SONG_FACTS_CACHE_DIR / collection_name
    if not coll_dir.is_dir():
        return {}

    result: Dict[str, str] = {}
    for f in coll_dir.iterdir():
        if f.suffix == ".txt":
            result[f.stem] = f.read_text(encoding="utf-8").strip()
    return result
