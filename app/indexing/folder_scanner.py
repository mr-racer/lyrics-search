"""Folder-level orchestrator: walk audio files + parallel enrichment.

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

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


def scan_and_enrich_folder(music_folder: str, workers: int = 8, better_lyrics_quality: bool = False, progress_callback=None):
    """Scan folder, enrich & fetch lyrics in parallel, return results keyed by artist/title.

    Args:
        music_folder: Path to scan for audio files
        workers: Number of parallel workers (1 if better_lyrics_quality)
        better_lyrics_quality: Use all providers including Musixmatch
        progress_callback: Optional callback(current, total, message) called after each file
    """

    if better_lyrics_quality:
        workers = 1

    audio_files = [
        p for p in Path(music_folder).rglob("*")
        if p.suffix.lower() in (".flac", ".m4a", ".mp3")
    ]
    logger.info("[scan] found %d files, workers: %d", len(audio_files), workers)

    results = {}
    processed_count = 0
    found_count = 0
    not_found_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, better_lyrics_quality): f for f in audio_files}
        start_time = time.time()

        for future in tqdm(as_completed(futures), total=len(audio_files)):
            meta = future.result()
            processed_count += 1

            if meta is None:
                # process_file returns None only when title/artist tags are missing
                # — that track can't be identified (distinct from "no lyrics"). Count
                # it and skip; this also guards the .get() below against None.
                not_found_count += 1
            elif meta.get("lyrics"):
                found_count += 1
                results.setdefault(f"{meta['artist']} — {meta['title']}", meta)
            else:
                # No lyrics found — still index the track. build_text_for_embedding
                # falls back to title/artist and the CLAP audio vector carries sonic
                # search; empty (not a placeholder) so no junk text gets embedded.
                not_found_count += 1
                meta['lyrics'] = ''
                results.setdefault(f"{meta['artist']} — {meta['title']}", meta)

            if progress_callback:
                fp = futures[future]

                elapsed = time.time() - start_time
                rate_per_min = None
                eta_seconds = None
                if elapsed > 1 and processed_count > 1:
                    rate_per_sec = processed_count / elapsed
                    rate_per_min = round(rate_per_sec * 60, 1)
                    remaining = len(audio_files) - processed_count
                    eta_seconds = max(0, int(remaining / rate_per_sec)) if rate_per_sec > 0 else None

                if meta:
                    last_track = f"{meta['artist']} — {meta['title']}"
                    status = f"[{processed_count}/{len(audio_files)}] ✓ {last_track}"
                    last_success = True
                else:
                    last_track = fp.stem
                    status = f"[{processed_count}/{len(audio_files)}] ✗ {fp.name}"
                    last_success = False

                details = {
                    "found":        found_count,
                    "not_found":    not_found_count,
                    "rate_per_min": rate_per_min,
                    "eta_seconds":  eta_seconds,
                    "last_track":   last_track,
                    "last_success": last_success,
                }
                progress_callback(processed_count, len(audio_files), status, details)

    logger.info("[scan] done: %d/%d lyrics found", len(results), len(audio_files))
    return results


# Backward-compat alias — callers still use this name.
fetch_lyrics_bulk = scan_and_enrich_folder
