"""Thin service layer over MetadataDB.record_playback_event.

Kept as a separate module so future query helpers (recently_played,
session_summary, etc.) can live here without bloating MetadataDB.
"""

from __future__ import annotations

from app.domain.models import RecentTrack, RecentTracksResponse
from app.resources.metadata_db import MetadataDB
from app.services._payload_coerce import coerce_float, coerce_year


def record_event(
    *,
    session_id: str,
    collection_name: str,
    track_id: str,
    played_sec: float,
    total_dur: float | None,
    interacted: bool | None = None,
    influence: bool = True,
) -> int:
    """Persist a playback event. Returns the new row id."""
    return MetadataDB.record_playback_event(
        session_id=session_id,
        collection_name=collection_name,
        track_id=track_id,
        played_sec=played_sec,
        total_dur=total_dur,
        interacted=interacted,
        influence=influence,
    )


def get_recent(*, qdrant_client, collection_name: str, limit: int = 50) -> RecentTracksResponse:
    triples = MetadataDB.get_recent_tracks(collection_name, limit)
    if not triples:
        return RecentTracksResponse(tracks=[], collection_name=collection_name)

    ids = [tid for tid, _, _ in triples]
    meta_by_id = {tid: (lp, pc) for tid, lp, pc in triples}

    try:
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        points = []

    tracks = []
    for p in points:
        pl = p.payload or {}
        lp, pc = meta_by_id.get(str(p.id), ("", 0))
        tracks.append(RecentTrack(
            track_id=str(p.id),
            title=pl.get("title") or "—",
            artist=pl.get("artist") or "—",
            album=pl.get("album"),
            year=coerce_year(pl.get("year")),
            duration=coerce_float(pl.get("duration")),
            cover_art_path=pl.get("cover_art_path"),
            genre=pl.get("genre"),
            last_played=lp,
            play_count=pc,
        ))
    tracks.sort(key=lambda t: t.last_played, reverse=True)
    return RecentTracksResponse(tracks=tracks, collection_name=collection_name)
