"""Migrate cached .txt fact files into the SQLite metadata store.

Reads from the existing ``cache/facts/{collection}/`` and
``cache/songfacts/{collection}/`` directories and inserts the content
into SQLite. Each file is split on ``\\n\\n`` boundaries so that
individual facts become separate rows.

Safe to run multiple times: artists and songs are upserted by slug,
and duplicate facts are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from ..resources.metadata_db import MetadataDB

logger = logging.getLogger(__name__)

FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "facts"
SONG_FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "songfacts"


def _read_facts_files(cache_dir: Path) -> Dict[str, List[str]]:
    """Read all .txt files from a collection directory.

    Returns a dict of ``{slug: [fact1, fact2, ...]}``.
    """
    result: Dict[str, List[str]] = {}
    if not cache_dir.is_dir():
        return result

    for f in cache_dir.iterdir():
        if f.suffix != ".txt":
            continue
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        facts = [part.strip() for part in text.split("\n\n") if part.strip()]
        result[f.stem] = facts
    return result


def migrate_artist_facts() -> int:
    """Migrate artist facts from ``cache/facts/`` to SQLite.

    Returns the number of artists migrated.
    """
    if not FACTS_CACHE_DIR.is_dir():
        logger.info("[Migrate] No artist facts cache directory found")
        return 0

    total = 0
    for collection_dir in FACTS_CACHE_DIR.iterdir():
        if not collection_dir.is_dir():
            continue
        collection_name = collection_dir.name
        facts_by_slug = _read_facts_files(collection_dir)

        for slug, facts in facts_by_slug.items():
            if not facts:
                continue
            MetadataDB.add_artist_facts_batch(slug, collection_name, facts, source="songfacts.com")
            total += 1

    logger.info("[Migrate] Migrated artist facts for %d artists across collections", total)
    return total


def migrate_song_facts() -> int:
    """Migrate song facts from ``cache/songfacts/`` to SQLite.

    Returns the number of songs migrated.
    """
    if not SONG_FACTS_CACHE_DIR.is_dir():
        logger.info("[Migrate] No song facts cache directory found")
        return 0

    total = 0
    for collection_dir in SONG_FACTS_CACHE_DIR.iterdir():
        if not collection_dir.is_dir():
            continue
        collection_name = collection_dir.name
        facts_by_slug = _read_facts_files(collection_dir)

        for slug, facts in facts_by_slug.items():
            if not facts:
                continue
            # Song slug format: "artist-title". Ensure the artist row exists
            # before inserting the song (foreign key constraint).
            parts = slug.split("-", 1)
            artist_slug = parts[0]
            artist_name = artist_slug.replace("-", " ")
            MetadataDB.upsert_artist(artist_slug, artist_name, collection_name)

            MetadataDB.add_song_facts_batch(slug, collection_name, facts, source="songfacts.com")
            total += 1

    logger.info("[Migrate] Migrated song facts for %d songs across collections", total)
    return total


def migrate_all() -> Dict[str, int]:
    """Run all migrations. Returns a summary dict."""
    MetadataDB.init()
    MetadataDB.close()  # let services open their own connection

    artists = migrate_artist_facts()
    songs = migrate_song_facts()

    return {
        "artists_migrated": artists,
        "songs_migrated": songs,
    }
