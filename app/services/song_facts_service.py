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
from app.services.proxy_config import get_proxy

logger = logging.getLogger(__name__)

# Legacy cache directory — kept for backward-compat fallback
_SONG_FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "songfacts"
REQUEST_TIMEOUT = 10


def _slugify(text: str) -> str:
    """'All Falls Down' -> 'all-falls-down'

    Strips noisy punctuation (apostrophes, quotes, commas, ?, !) before
    slugifying. Covers both straight ASCII and the various smart-quote
    codepoints that leak in from iTunes/ID3 tags and pasted web text —
    most importantly U+2019 (RIGHT SINGLE QUOTATION MARK '), which the
    original regex missed and which produced broken songfacts.com URLs
    for any track with a possessive (e.g. "We're Good", "Don't Start Now").
    """
    cleaned_title = re.sub(
        "['`,?!‘’‚‛“”„‟′″ʼ«»]",
        '', text,
    )
    brackets_delete_pattern = r'^(.+?)\s*\(.*'
    cleaned_title = re.sub(brackets_delete_pattern, r'\1', cleaned_title)

    return "-".join(cleaned_title.lower().split())


def _fetch_song_facts_html(artist: str, song: str) -> Optional[str]:
    """GET songfacts.com/facts/{artist}/{song} and return text, or None."""
    slug_artist = _slugify(artist)
    slug_song = _slugify(song)
    url = f"https://www.songfacts.com/facts/{slug_artist}/{slug_song}"
    try:
        logger.info("[SongFacts] Fetching facts for '%s — %s' (%s)", artist, song, url)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, proxies=get_proxy())
        logger.info("[SongFacts] '%s — %s' → HTTP %s, %d bytes", artist, song, resp.status_code, len(resp.text))
        resp.raise_for_status()
        return resp.text
    except requests.HTTPError as e:
        logger.warning("[SongFacts] HTTP error for '%s — %s' (%s): HTTP %s", artist, song, url, e.response.status_code)
        return None
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
        logger.info("[SongFacts] using cached facts for '%s — %s'", artist, song)
        return cached

    html = await asyncio.to_thread(_fetch_song_facts_html, artist, song)
    if not html:
        return None

    facts = _parse_song_facts(html)
    if not facts:
        logger.info("[SongFacts] no facts parsed for '%s — %s' (HTML fetched but no facts found)", artist, song)
        return None

    _save_song_facts_to_sqlite(collection_name, artist, song, facts)
    logger.info("[SongFacts] cached %d facts for '%s — %s'", len(facts), artist, song)
    return "\n\n".join(facts)


async def fetch_facts_for_songs(
    songs: List[Tuple[str, str]],
    collection_name: str,
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str, bool], None]] = None,
) -> Dict[str, str]:
    """Fetch facts for multiple songs sequentially with delay between requests.

    Args:
        songs: list of (artist, song_title) tuples.
        collection_name: collection scope.
        delay: seconds between requests.
        progress_callback: optional callback(current, total, label, found) after each song.
    Returns:
        dict of '{artist} — {song}' -> facts text (only for songs that had facts).
    """
    results: Dict[str, str] = {}
    total = len(songs)
    for idx, (artist, song) in enumerate(songs, 1):
        key = f"{artist} — {song}"
        text = await fetch_song_facts(artist, song, collection_name)
        found = bool(text)
        if found:
            results[key] = text
        if progress_callback:
            progress_callback(idx, total, f"{artist} — {song}", found)
        await asyncio.sleep(delay)
    return results


def load_all_song_facts_for_collection(
    collection_name: str,
    ai_enabled: bool = True,
) -> Dict[str, str]:
    """Load all cached song facts for a collection.

    When ``ai_enabled`` is True, refined facts (from AI Indexing) take
    precedence over originals for any song that has a refined row.

    When ``ai_enabled`` is False, only original facts are returned.

    Prefers SQLite; falls back to legacy ``.txt`` files if SQLite is empty.
    """
    # Layer 1: originals from SQLite or legacy .txt
    originals: Dict[str, str] = {}
    try:
        originals = MetadataDB.get_all_song_facts_by_collection(collection_name)
    except Exception as e:
        logger.debug("[SongFacts] SQLite collection read failed: %s", e)

    if not originals:
        coll_dir = _SONG_FACTS_CACHE_DIR / collection_name
        if coll_dir.is_dir():
            for f in coll_dir.iterdir():
                if f.suffix == ".txt":
                    originals[f.stem] = f.read_text(encoding="utf-8").strip()

    # Layer 2: if AI enabled, overlay refined facts on top
    if ai_enabled:
        try:
            refined = MetadataDB.get_all_refined_song_facts(collection_name)
            if refined:
                merged = dict(originals)
                merged.update(refined)
                return merged
        except Exception as e:
            logger.debug("[SongFacts] refined facts read failed: %s", e)

    return originals
