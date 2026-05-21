"""Indexing pipeline — scan music folder, read metadata, fetch lyrics, extract covers.

Extracted from legacy file_processor/ during Refactor 3. Sub-modules:
    metadata_readers — Mutagen-based audio metadata + genre normalisation
    cover_art        — extract embedded cover art, save content-addressed
    lyrics_fetchers  — lyrics.ovh + syncedlyrics fallback chain
    audio_optimization — ffmpeg M4A faststart for HTTP Range streaming
    folder_scanner   — orchestrator: walk folder + parallel enrichment
"""
