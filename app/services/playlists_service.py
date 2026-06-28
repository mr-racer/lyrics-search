"""Service layer for Plan 19 Custom Playlists CRUD.

Mirrors the shape of `playback_service` — module-level functions on top
of MetadataDB classmethods + Qdrant batch-retrieve for track resolution.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from app.domain.models import (
    PlaylistDetail, PlaylistSummary, PlaylistTrack,
)
from app.resources.metadata_db import MetadataDB
from app.services._payload_coerce import coerce_float, coerce_year
from app.services.artist_split import artist_refs as _artist_refs


# ─── Exceptions ──────────────────────────────────────────────────────────

class PlaylistNotFound(Exception):
    """Playlist id does not exist."""


class PlaylistNameCollision(Exception):
    """A playlist with this name already exists in the collection."""


class TrackAlreadyInPlaylist(Exception):
    """The given track is already a member of this playlist."""


class TrackNotInPlaylist(Exception):
    """The given track is not a member of this playlist."""


# ─── Helpers ─────────────────────────────────────────────────────────────

def _resolve_payloads(qdrant, collection_name: str, track_ids: list[str]) -> dict[str, dict]:
    """Return {track_id: payload_dict} for every id Qdrant knows about. Orphans omitted."""
    if not track_ids or qdrant is None:
        return {}
    try:
        points = qdrant.retrieve(
            collection_name=collection_name,
            ids=track_ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return {}
    return {str(p.id): (p.payload or {}) for p in points}


def _build_summary(
    row: dict,
    track_rows: list[dict],
    payloads: dict[str, dict],
    contains_track: Optional[bool] = None,
) -> PlaylistSummary:
    """Compose a PlaylistSummary. cover_track_ids = first 4 by position; cover_art_paths parallel."""
    first_four = track_rows[:4]
    cover_ids = [tr["track_id"] for tr in first_four]
    cover_paths = [
        (payloads.get(tr["track_id"]) or {}).get("cover_art_path") for tr in first_four
    ]
    return PlaylistSummary(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        track_count=len(track_rows),
        cover_track_ids=cover_ids,
        cover_art_paths=cover_paths,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        contains_track=contains_track,
    )


# ─── Public API ──────────────────────────────────────────────────────────

def create(collection_name: str, name: str, description: Optional[str], *, qdrant) -> PlaylistSummary:
    try:
        pid = MetadataDB.create_playlist(collection_name, name, description)
    except sqlite3.IntegrityError as e:
        raise PlaylistNameCollision(
            f"Playlist with this name already exists in collection {collection_name!r}"
        ) from e
    row = MetadataDB.get_playlist_row(pid)
    return _build_summary(row, [], {})


def list_playlists(collection_name: str, *, include_track_id: Optional[str], qdrant) -> list[PlaylistSummary]:
    rows = MetadataDB.list_playlists(collection_name)
    out: list[PlaylistSummary] = []
    all_first_four_ids: set[str] = set()
    per_playlist_tracks: dict[int, list[dict]] = {}
    for r in rows:
        tracks = MetadataDB.list_playlist_tracks(r["id"])
        per_playlist_tracks[r["id"]] = tracks
        for tr in tracks[:4]:
            all_first_four_ids.add(tr["track_id"])
    payloads = _resolve_payloads(qdrant, collection_name, list(all_first_four_ids))

    membership_ids: set[int] = set()
    if include_track_id:
        membership_ids = set(MetadataDB.playlists_containing_track(collection_name, include_track_id))

    for r in rows:
        contains = None
        if include_track_id is not None:
            contains = r["id"] in membership_ids
        out.append(_build_summary(r, per_playlist_tracks[r["id"]], payloads, contains_track=contains))
    return out


def get(playlist_id: int, *, qdrant) -> PlaylistDetail:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    track_rows = MetadataDB.list_playlist_tracks(playlist_id)
    track_ids = [tr["track_id"] for tr in track_rows]
    payloads = _resolve_payloads(qdrant, row["collection_name"], track_ids)
    tracks: list[PlaylistTrack] = []
    missing: list[str] = []
    for tr in track_rows:
        pl = payloads.get(tr["track_id"])
        if pl is None:
            missing.append(tr["track_id"])
            continue
        tracks.append(PlaylistTrack(
            track_id=tr["track_id"],
            position=tr["position"],
            added_at=tr["added_at"],
            title=pl.get("title") or "—",
            artist=pl.get("artist") or "—",
            album=pl.get("album"),
            year=coerce_year(pl.get("year")),
            duration=coerce_float(pl.get("duration")),
            cover_art_path=pl.get("cover_art_path"),
            artist_refs=_artist_refs(pl.get("artist")),
        ))
    return PlaylistDetail(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        collection_name=row["collection_name"],
        tracks=tracks,
        missing_track_ids=missing,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def update(
    playlist_id: int,
    *,
    name: Optional[str],
    description: Optional[str],
    clear_description: bool,
    qdrant,
) -> PlaylistSummary:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    try:
        MetadataDB.update_playlist(
            playlist_id,
            name=name,
            description=description,
            clear_description=clear_description,
        )
    except sqlite3.IntegrityError as e:
        raise PlaylistNameCollision(
            f"Playlist with this name already exists in collection {row['collection_name']!r}"
        ) from e
    updated_row = MetadataDB.get_playlist_row(playlist_id)
    tracks = MetadataDB.list_playlist_tracks(playlist_id)
    payloads = _resolve_payloads(
        qdrant, row["collection_name"], [tr["track_id"] for tr in tracks[:4]]
    )
    return _build_summary(updated_row, tracks, payloads)


def delete(playlist_id: int) -> None:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    MetadataDB.delete_playlist(playlist_id)


def add_track(playlist_id: int, track_id: str, *, qdrant) -> PlaylistSummary:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    try:
        MetadataDB.add_track_to_playlist(playlist_id, track_id)
    except sqlite3.IntegrityError as e:
        raise TrackAlreadyInPlaylist(
            f"track {track_id!r} already in playlist {playlist_id}"
        ) from e
    updated_row = MetadataDB.get_playlist_row(playlist_id)
    tracks = MetadataDB.list_playlist_tracks(playlist_id)
    payloads = _resolve_payloads(
        qdrant, row["collection_name"], [tr["track_id"] for tr in tracks[:4]]
    )
    return _build_summary(updated_row, tracks, payloads)


def remove_track(playlist_id: int, track_id: str) -> None:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    removed = MetadataDB.remove_track_from_playlist(playlist_id, track_id)
    if not removed:
        raise TrackNotInPlaylist(f"track {track_id!r} not in playlist {playlist_id}")


def reorder(playlist_id: int, track_ids: list[str], *, qdrant) -> PlaylistSummary:
    row = MetadataDB.get_playlist_row(playlist_id)
    if row is None:
        raise PlaylistNotFound(f"playlist {playlist_id} not found")
    MetadataDB.reorder_playlist(playlist_id, track_ids)
    updated_row = MetadataDB.get_playlist_row(playlist_id)
    tracks = MetadataDB.list_playlist_tracks(playlist_id)
    payloads = _resolve_payloads(
        qdrant, row["collection_name"], [tr["track_id"] for tr in tracks[:4]]
    )
    return _build_summary(updated_row, tracks, payloads)
