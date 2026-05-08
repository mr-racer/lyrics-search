"""Fetch and cache interesting artist facts from SongFacts."""

import asyncio
import logging
import re
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "facts"
REQUEST_TIMEOUT = 10  # seconds


def _slugify(artist: str) -> str:
    """'The Weeknd' -> 'the-weeknd'"""
    return "-".join(artist.lower().split())


def _fetch_facts_html(artist: str) -> Optional[str]:
    """GET songfacts.com/facts/{slug} and return text, or None on failure."""
    slug = _slugify(artist)
    url = f"https://www.songfacts.com/facts/{slug}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
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


def _cache_path(collection_name: str, artist: str) -> Path:
    """Return path to cached facts file for a collection + artist."""
    coll_dir = FACTS_CACHE_DIR / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    return coll_dir / f"{_slugify(artist)}.txt"


def get_cached_facts(collection_name: str, artist: str) -> Optional[str]:
    """Read cached facts for an artist in a collection, or None."""
    p = _cache_path(collection_name, artist)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _save_facts(collection_name: str, artist: str, text: str) -> None:
    p = _cache_path(collection_name, artist)
    p.write_text(text, encoding="utf-8")


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

    text = "\n\n".join(facts)
    _save_facts(collection_name, artist, text)
    return text


async def fetch_facts_for_artists(
    artists: List[str],
    collection_name: str,
    delay: float = 0.5,
) -> Dict[str, str]:
    """Fetch facts for multiple artists sequentially with delay between requests.

    Returns dict of artist -> facts text (only for artists that had facts).
    """
    results: Dict[str, str] = {}
    for artist in artists:
        text = await fetch_artist_facts(artist, collection_name)
        if text:
            results[artist] = text
        await asyncio.sleep(delay)
    return results


def load_all_facts_for_collection(collection_name: str) -> Dict[str, str]:
    """Load all cached facts for a collection from disk (sync, no network)."""
    coll_dir = FACTS_CACHE_DIR / collection_name
    if not coll_dir.is_dir():
        return {}

    facts: Dict[str, str] = {}
    for f in coll_dir.iterdir():
        if f.suffix == ".txt":
            # slug -> file, use slug as key; frontend can resolve display name
            facts[f.stem] = f.read_text(encoding="utf-8").strip()
    return facts
