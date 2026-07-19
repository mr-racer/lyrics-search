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


def _artist_query(artist: str) -> str:
    """Reduce a raw artist tag to the PRIMARY performer for song-facts lookup.

    songfacts.com lists each song under one page keyed by the primary performer,
    so a collab tag like "DNCE, Nicki Minaj" must query ``/facts/dnce/{song}``,
    not ``/facts/dnce-nicki-minaj/{song}`` (which 404s). Lazy import keeps this
    module free of an import-order dependency on ``artist_split``.
    """
    try:
        from app.services.artist_split import primary_artist
        return primary_artist(artist) or artist
    except Exception:
        return artist


def _slugify(text: str) -> str:
    """'All Falls Down' -> 'all-falls-down'

    Strips noisy punctuation (apostrophes, quotes, commas, ?, !) before
    slugifying. Covers both straight ASCII and the various smart-quote
    codepoints that leak in from iTunes/ID3 tags and pasted web text —
    most importantly U+2019 (RIGHT SINGLE QUOTATION MARK '), which the
    original regex missed and which produced broken songfacts.com URLs
    for any track with a possessive (e.g. "We're Good", "Don't Start Now").
    Also normalizes Unicode dash variants to ASCII hyphen so that
    "My–Band" and "My-Band" produce the same slug.
    """
    # Unicode dash variants → ASCII hyphen
    cleaned_title = re.sub(r"[‐‑‒–—―−]", "-", text)
    cleaned_title = re.sub(
        "['`,?!''‚‛""„‟′″ʼ«»]",
        '', cleaned_title,
    )
    brackets_delete_pattern = r'^(.+?)\s*\(.+'
    cleaned_title = re.sub(brackets_delete_pattern, r'\1', cleaned_title)

    return "-".join(cleaned_title.lower().split())


def _fetch_song_facts_html(artist: str, song: str) -> Optional[str]:
    """GET songfacts.com/facts/{artist}/{song} and return text, or None."""
    slug_artist = _slugify(_artist_query(artist))
    slug_song = _slugify(song)
    url = f"https://www.songfacts.com/facts/{slug_artist}/{slug_song}"
    try:
        logger.info("[SongFacts] Fetching facts for '%s — %s' (%s)", artist, song, url)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, proxies=get_proxy())
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
    safe_name = f"{_slugify(_artist_query(artist))}-{_slugify(song)}"
    return coll_dir / f"{safe_name}.txt"


def get_song_facts_key(artist: str, song: str) -> str:
    """Build the cache-key for a song (matches file stem / DB slug).

    The artist component is reduced to the PRIMARY performer (see ``_artist_query``)
    so the cache key matches the URL actually fetched for collab tracks.
    """
    return f"{_slugify(_artist_query(artist))}-{_slugify(song)}"


def apply_song_relations(tracks) -> None:
    """Overlay producer/label/samples/sampled_by from the GLiNER2+LLM
    extraction (``songs.producers``/``label``/``samples_json``) onto a list of
    ``TrackMetadata``-like objects, keyed by the same slug song facts use.

    Mutates in place; a track whose slug has no extraction yet (or whose
    artist/title is empty) is left untouched, so its existing (currently
    always-empty, Qdrant-payload-sourced) fields pass through unchanged.
    Never raises — a lookup failure just skips the overlay.
    """
    keyed: Dict[str, list] = {}
    for t in tracks:
        artist = getattr(t, "artist", None)
        title = getattr(t, "title", None)
        if not artist or not title:
            continue
        keyed.setdefault(get_song_facts_key(artist, title), []).append(t)
    if not keyed:
        return
    try:
        relations = MetadataDB.get_song_relations_bulk(list(keyed.keys()))
    except Exception:
        logger.warning("[song_facts] relations overlay failed (non-fatal)", exc_info=True)
        return
    for slug, rel in relations.items():
        for t in keyed.get(slug, []):
            if rel.get("producer"):
                t.producer = rel["producer"]
            if rel.get("label"):
                t.label = rel["label"]
            if rel.get("samples"):
                t.samples = rel["samples"]
            if rel.get("sampled_by"):
                t.sampled_by = rel["sampled_by"]


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
    """Save parsed song facts to SQLite.

    Passes the real (tag) title + primary-artist display name/slug down so the
    songs/artists rows keep human attribution — without them the writer would
    only be able to insert slug-derived placeholders ("50 cent in da club")."""
    key = get_song_facts_key(artist, song)
    primary = _artist_query(artist)
    MetadataDB.add_song_facts_batch(
        key, collection_name, facts, source="songfacts.com",
        artist_name=primary, title=song, artist_slug=_slugify(primary),
    )


async def _apply_fact_relations(slug: str, facts: List[str], title: str, artist: str) -> None:
    """Best-effort producer/sample extraction hook, run inline after facts save.

    NLP (GLiNER2) and LLM errors here must never break facts fetching/indexing
    -- everything is swallowed, same pattern as
    ``app/services/yandex/enrichment.py``. The heavy work (GLiNER2 + LLM) runs
    off the event loop inside ``process_song_facts_async`` since
    ``fetch_song_facts`` is awaited from ``library_service``'s
    ``asyncio.gather`` on the running loop.
    """
    try:
        from .fact_relations import process_song_facts_async

        res = await process_song_facts_async(slug, facts, title, artist, MetadataDB)
        producers = res.get("producers", []) or []
        samples = res.get("samples", []) or []
        sampled_by = res.get("sampled_by", []) or []
        n_samples = len(samples) + len(sampled_by)
        logger.info(
            "[facts-re] slug=%s producers=%d samples=%d/%d",
            slug, len(producers), n_samples, len(facts),
        )
    except Exception as e:
        logger.warning("[facts-re] slug=%s failed: %s", slug, e)


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
    key = get_song_facts_key(artist, song)
    await _apply_fact_relations(key, facts, song, artist)
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
