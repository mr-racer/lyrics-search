"""Thin service layer over MetadataDB.record_playback_event.

Kept as a separate module so future query helpers (recently_played,
session_summary, etc.) can live here without bloating MetadataDB.
"""

from __future__ import annotations

from app.resources.metadata_db import MetadataDB


def record_event(
    *,
    session_id: str,
    collection_name: str,
    track_id: str,
    played_sec: float,
    total_dur: float | None,
) -> int:
    """Persist a playback event. Returns the new row id."""
    return MetadataDB.record_playback_event(
        session_id=session_id,
        collection_name=collection_name,
        track_id=track_id,
        played_sec=played_sec,
        total_dur=total_dur,
    )
