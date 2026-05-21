"""Backward-compatibility shim.

Actual implementations moved to ``app/indexing/`` package during Refactor 3:
    metadata_readers   — get_flac/alac/mp3_metadata, validate_year, normalize_genre, NOISE, genre_map
    cover_art          — extract_cover_art, save_cover_art, normalize_image, COVERS_DIR
    lyrics_fetchers    — fetch_lyrics_ovh, get_lyrics, research_with_musixmatch, PROVIDERS, TIME_*
    audio_optimization — optimize_m4a_for_streaming
    folder_scanner     — process_file, scan_and_enrich_folder (alias: fetch_lyrics_bulk)

This shim will be deleted in Refactor 6 once all call sites migrate.
"""
from app.indexing.metadata_readers import (  # noqa: F401
    NOISE,
    _CURRENT_YEAR,
    _MIN_YEAR,
    genre_map,
    get_alac_metadata,
    get_flac_metadata,
    get_metadata,
    get_mp3_metadata,
    normalize_genre,
    validate_year,
)
from app.indexing.cover_art import (  # noqa: F401
    COVERS_DIR,
    extract_cover_art,
    normalize_image,
    save_cover_art,
)
from app.indexing.lyrics_fetchers import (  # noqa: F401
    PROVIDERS,
    TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS,
    TIME_BETWEEN_REQUESTS_OVH,
    TIME_BETWEEN_REQUESTS_STANDARD,
    fetch_lyrics_ovh,
    get_lyrics,
    research_with_musixmatch,
)
from app.indexing.audio_optimization import optimize_m4a_for_streaming  # noqa: F401
from app.indexing.folder_scanner import (  # noqa: F401
    fetch_lyrics_bulk,
    process_file,
    scan_and_enrich_folder,
)

# MAX_DURATION lived here historically; preserved for callers.
MAX_DURATION = 420
