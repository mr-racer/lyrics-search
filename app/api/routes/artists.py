"""Artist Atlas aggregate endpoint."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.domain.models import ArtistAggregate, ArtistAlbum, TrackMetadata
from app.resources.metadata_db import MetadataDB
from app.services.artist_facts_service import _slugify as _slugify_artist

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/artists", tags=["Artists"])


def _track_from_payload(point_id: str, p: dict) -> TrackMetadata:
    """Map a Qdrant payload dict to TrackMetadata. Returns minimal-field model."""
    return TrackMetadata(
        track_id=point_id,
        title=p.get("title") or "",
        artist=p.get("artist") or "",
        album=p.get("album"),
        year=p.get("year") if isinstance(p.get("year"), int) else None,
        genre=p.get("genre"),
        duration_sec=float(p.get("duration") or 0.0),
        file_path=p.get("file_path") or "",
        lyrics=p.get("lyrics"),
        cover_art_path=p.get("cover_art_path"),
        producer=p.get("producer"),
        label=p.get("label"),
        samples=p.get("samples"),
        sampled_by=p.get("sampled_by"),
    )


def _decade_range(years: list[int]) -> Optional[str]:
    """Return e.g. '2010s' or '2010s-2020s' from a list of int years."""
    valid = sorted({(y // 10) * 10 for y in years if isinstance(y, int) and y > 0})
    if not valid:
        return None
    if len(valid) == 1:
        return f"{valid[0]}s"
    return f"{valid[0]}s-{valid[-1]}s"


@router.get("/{slug}", response_model=ArtistAggregate)
def get_artist(
    slug: str,
    collection: str = Query(..., description="Target collection name"),
    lang: str = Query("en", description="Bio language"),
    request: Request = None,
) -> ArtistAggregate:
    """Aggregate an artist's universe — used by the Atlas screen."""
    if request is None or request.app.state.db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    db = request.app.state.db_client

    # Tolerate slug variants from clients that use a simpler slugifier than
    # artist_facts_service._slugify (which strips smart quotes / +, &, . etc).
    # Re-slugifying the input through the canonical function makes
    # "guns-n'-roses" / "guns-n-roses" / "Guns N' Roses" all resolve.
    canonical_slug = _slugify_artist(slug.replace("-", " "))

    # Resolve canonical artist name from SQLite (slug → name)
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM artists WHERE slug = ?", (canonical_slug,),
    ).fetchone()

    # Scroll Qdrant collecting this artist's tracks (slug-matched)
    artist_tracks: list[TrackMetadata] = []
    offset = None
    while True:
        try:
            points, offset = db.qdrant.scroll(
                collection_name=collection, limit=64, offset=offset,
                with_payload=True, with_vectors=False,
            )
        except Exception as e:
            logger.warning("[artists] scroll failed: %s", e)
            break
        if not points:
            break
        for pt in points:
            p = pt.payload or {}
            artist_name = (p.get("artist") or "").strip()
            if not artist_name:
                continue
            if _slugify_artist(artist_name) != canonical_slug:
                continue
            artist_tracks.append(_track_from_payload(str(pt.id), p))
        if offset is None:
            break

    if not artist_tracks and not row:
        raise HTTPException(status_code=404, detail=f"unknown artist: {canonical_slug}")

    # Prefer the artist name from Qdrant track payloads (those are the real
    # tagged metadata). The SQLite ``artists.name`` column can be a slug-derived
    # placeholder (e.g. "dua lipa") because ``add_artist_facts_batch`` does
    # ``ON CONFLICT DO UPDATE SET name = slug.replace("-", " ")``. Fall back to
    # the SQLite row name only when no tracks are present, and finally to a
    # title-cased slug.
    name = (
        (artist_tracks[0].artist if artist_tracks else None)
        or (row[0] if row else None)
        or canonical_slug.replace("-", " ").title()
    )

    # Group tracks by album
    by_album: dict[str, list[TrackMetadata]] = defaultdict(list)
    for t in artist_tracks:
        by_album[t.album or "—"].append(t)
    albums: list[ArtistAlbum] = []
    for album_title, tracks in by_album.items():
        years = [t.year for t in tracks if t.year]
        # Representative cover = first track's that has one
        cover = next((t.cover_art_path for t in tracks if t.cover_art_path), None)
        albums.append(ArtistAlbum(
            title=album_title,
            year=min(years) if years else None,
            cover_art_path=cover,
            tracks=tracks,
        ))
    albums.sort(key=lambda a: (a.year or 9999, a.title))

    facts = MetadataDB.get_artist_facts(canonical_slug, collection)
    bio = MetadataDB.get_artist_bio(canonical_slug, collection, lang)
    primary_genre = next((t.genre for t in artist_tracks if t.genre), None)

    return ArtistAggregate(
        slug=canonical_slug,
        name=name,
        genre=primary_genre,
        track_count=len(artist_tracks),
        album_count=len(albums),
        decade_range=_decade_range([t.year for t in artist_tracks if t.year]),
        bio=bio,
        facts=facts,
        albums=albums,
    )
