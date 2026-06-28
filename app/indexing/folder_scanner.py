"""Folder-level orchestrator: walk audio files + parallel enrichment.

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .audio_optimization import optimize_m4a_for_streaming
from .lyrics_fetchers import get_lyrics
from .metadata_readers import get_metadata, normalize_genre, read_embedded_lyrics

logger = logging.getLogger(__name__)


def process_file(
    filepath: Path,
    better_lyrics_quality: bool,
    *,
    enrich_client=None,
) -> dict | None:
    """Load metadata, enrich online where needed, then fetch lyrics.

    For M4A files, optimizes the file first to enable HTTP Range requests.

    ``enrich_client``: optional Yandex client used by the metadata-enrichment
    stage (feature #2). Built once per job by the caller (account token if
    linked, else anonymous) and passed in; ``None`` lets enrichment lazily fall
    back to a shared anonymous client.

    Returns:
        Full metadata dict or None if the track should be skipped.
    """
    # Optimize M4A files for streaming (move moov atom to beginning)
    if filepath.suffix.lower() == ".m4a":
        optimize_m4a_for_streaming(str(filepath))

    meta = get_metadata(filepath)
    if not meta or not meta["title"] or not meta["artist"]:
        logger.info("[scan] skip (missing title/artist): %s", filepath.name)
        return None

    # Feature #2: backfill empty fields (album/year/genre) from the Yandex
    # catalog BEFORE genre normalisation, so an enriched genre gets normalised
    # too. Best-effort — never raises.
    try:
        from app.services.yandex.enrichment import enrich_metadata
        meta = enrich_metadata(meta, client=enrich_client)
    except Exception:
        logger.debug("[scan] enrichment unavailable for %s", filepath.name, exc_info=True)

    # Prefer lyrics embedded in the file (e.g. written by the Yandex import
    # downloader) over an online fetch — matches the "embed Yandex data" choice.
    lyrics = read_embedded_lyrics(filepath)
    if not lyrics:
        lyrics = get_lyrics(meta["title"], meta["artist"], better_lyrics_quality)
    if not lyrics:
        logger.info("[scan] no lyrics: %s — %s", meta['artist'], meta['title'])

    if meta.get('genre'):
        meta['genre'] = normalize_genre(meta['genre'])

    return {**meta, "lyrics": lyrics, "file_path": str(filepath)}


def read_tags_only(filepath: Path, *, enrich_client=None) -> dict | None:
    """Tag-read pass WITHOUT the online lyrics fetch (for the parallel pipeline).

    Identical to :func:`process_file` up to — but excluding — the online
    ``get_lyrics`` call. ``meta["lyrics"]`` is the text **embedded** in the file
    (e.g. written by the Yandex downloader) or ``""`` when absent. The pipeline
    then fetches online lyrics only for tracks whose ``lyrics`` came back empty,
    concurrently with CLAP encoding and facts — so embedded/Yandex lyrics
    naturally skip the network stage.

    Returns the metadata dict (with ``lyrics`` + ``file_path``) or ``None`` when
    the file has no title/artist (unidentifiable — caller should skip it).
    """
    if filepath.suffix.lower() == ".m4a":
        optimize_m4a_for_streaming(str(filepath))

    meta = get_metadata(filepath)
    if not meta or not meta["title"] or not meta["artist"]:
        logger.info("[scan] skip (missing title/artist): %s", filepath.name)
        return None

    try:
        from app.services.yandex.enrichment import enrich_metadata
        meta = enrich_metadata(meta, client=enrich_client)
    except Exception:
        logger.debug("[scan] enrichment unavailable for %s", filepath.name, exc_info=True)

    lyrics = read_embedded_lyrics(filepath) or ""

    if meta.get('genre'):
        meta['genre'] = normalize_genre(meta['genre'])

    return {**meta, "lyrics": lyrics, "file_path": str(filepath)}


def fetch_online_lyrics(meta: dict, better_lyrics_quality: bool = False) -> str:
    """Online lyrics fetch for one already tag-read track. Returns text or ``""``.

    Split out of :func:`process_file` so the pipeline can run it on the network
    lane for only the tracks that lack embedded lyrics.
    """
    lyrics = get_lyrics(meta["title"], meta["artist"], better_lyrics_quality)
    if not lyrics:
        logger.info("[scan] no lyrics: %s — %s", meta.get('artist'), meta.get('title'))
    return lyrics or ""
