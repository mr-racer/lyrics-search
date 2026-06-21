"""Artist Atlas aggregate endpoint."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.domain.models import ArtistAggregate, ArtistAlbum, TrackMetadata, User
from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import PAYLOAD_EXCLUDE_LYRICS
from app.services.artist_facts_service import _slugify as _slugify_artist
from app.services.artist_split import split_artists, artist_slugs, name_for_slug
from app.services._payload_coerce import coerce_float, coerce_year

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/artists", tags=["Artists"])


def _coerce_float(val) -> float:
    """Atlas duration coercion: shared :func:`coerce_float` range-parsing, but
    None/empty/unparseable collapses to 0.0 so the Atlas endpoint never 500s on
    dirty payload data (the shared helper returns None for those cases)."""
    return coerce_float(val) or 0.0


def _coerce_year(val) -> Optional[int]:
    """Atlas year coercion — thin alias over the shared :func:`coerce_year`."""
    return coerce_year(val)


def _track_from_payload(point_id: str, p: dict) -> TrackMetadata:
    """Map a Qdrant payload dict to TrackMetadata. Returns minimal-field model."""
    # Prefer payload fields; if either is missing, recompute BOTH from the raw
    # `artist` so names[i] and slugs[i] always describe the same participant.
    names = p.get("artists")
    slugs = p.get("artist_slugs")
    if not names or not slugs:
        raw = p.get("artist") or ""
        names = split_artists(raw)
        slugs = artist_slugs(raw)
    primary = p.get("primary_artist_slug") or (slugs[0] if slugs else None)
    return TrackMetadata(
        track_id=point_id,
        title=p.get("title") or "",
        artist=p.get("artist") or "",
        artists=names or None,
        primary_artist_slug=primary,
        album=p.get("album"),
        year=_coerce_year(p.get("year")),
        genre=p.get("genre"),
        duration_sec=_coerce_float(p.get("duration")),
        file_path=p.get("file_path") or "",
        lyrics=p.get("lyrics"),
        cover_art_path=p.get("cover_art_path"),
        producer=p.get("producer"),
        label=p.get("label"),
        samples=p.get("samples"),
        sampled_by=p.get("sampled_by"),
        track_number=p.get("track_number"),
        disc_number=p.get("disc_number"),
    )


def _decade_range(years: list[int]) -> Optional[str]:
    """Return e.g. '2010s' or '2010s-2020s' from a list of int years."""
    valid = sorted({(y // 10) * 10 for y in years if isinstance(y, int) and y > 0})
    if not valid:
        return None
    if len(valid) == 1:
        return f"{valid[0]}s"
    return f"{valid[0]}s-{valid[-1]}s"


def _album_track_sort_key(tracks: list[TrackMetadata]) -> list[TrackMetadata]:
    """Order an album's tracks by real position.

    If EVERY track has a ``track_number``, sort by ``(disc_number or 1,
    track_number)``. Otherwise fall back to title order for the whole album — the
    agreed behaviour for incomplete numbering (no separate trailing bucket for
    un-numbered tracks). ``sorted`` is stable, so equal keys keep scroll order.
    """
    if tracks and all(t.track_number is not None for t in tracks):
        return sorted(tracks, key=lambda t: (t.disc_number or 1, t.track_number))
    return sorted(tracks, key=lambda t: (t.title or "").lower())


def _track_from_row(row: dict) -> TrackMetadata:
    """Map a SQLite track_metadata row dict to TrackMetadata."""
    names = row.get("artists")
    slugs = row.get("artist_slugs")
    raw = row.get("artist") or ""
    if not names or not slugs:
        from app.services.artist_split import split_artists, artist_slugs as _artist_slugs
        names = split_artists(raw)
        slugs = _artist_slugs(raw)
    primary = row.get("primary_artist_slug") or (slugs[0] if slugs else None)
    return TrackMetadata(
        track_id=row.get("track_id") or "",
        title=row.get("title") or "",
        artist=raw,
        artists=names or None,
        primary_artist_slug=primary,
        album=row.get("album"),
        year=row.get("year"),
        genre=row.get("genre"),
        duration_sec=_coerce_float(row.get("duration")),
        file_path=row.get("file_path") or "",
        cover_art_path=row.get("cover_art_path"),
        producer=row.get("producer"),
        label=row.get("label"),
        track_number=row.get("track_number"),
        disc_number=row.get("disc_number"),
    )


def _tracks_from_qdrant(db, collection: str, canonical_slug: str) -> list[TrackMetadata]:
    """Fallback: scroll Qdrant to collect tracks for an artist.

    Used when SQLite track_metadata is empty (pre-backfill) or on error.
    """
    from qdrant_client import models as qm

    flt = qm.Filter(must=[
        qm.FieldCondition(
            key="artist_slugs",
            match=qm.MatchValue(value=canonical_slug),
        )
    ])
    artist_tracks: list[TrackMetadata] = []
    offset = None
    while True:
        try:
            points, offset = db.qdrant.scroll(
                collection_name=collection, limit=64, offset=offset,
                with_payload=PAYLOAD_EXCLUDE_LYRICS, with_vectors=False,
                scroll_filter=flt,
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
            slugs = p.get("artist_slugs") or artist_slugs(artist_name)
            if canonical_slug not in slugs:
                continue
            artist_tracks.append(_track_from_payload(str(pt.id), p))
        if offset is None:
            break
    return artist_tracks


def build_artist_aggregate(db, collection: str, canonical_slug: str, lang: str) -> ArtistAggregate:
    """Build the Atlas aggregate for one artist. Shared by GET /artists/{slug}
    and GET /library/featured-artist. Raises HTTPException(404) if unknown."""
    # Resolve canonical artist name from SQLite (slug -> name)
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM artists WHERE slug = ?", (canonical_slug,),
    ).fetchone()

    # Try SQLite track_metadata first (fast indexed query)
    sqlite_tracks = MetadataDB.get_tracks_for_artist(collection, canonical_slug)

    if sqlite_tracks:
        artist_tracks = [_track_from_row(r) for r in sqlite_tracks]
    else:
        # Fallback: scroll Qdrant (pre-backfill or missing data)
        logger.info("[artists] no track_metadata for %s in %s — falling back to Qdrant scroll",
                     canonical_slug, collection)
        artist_tracks = _tracks_from_qdrant(db, collection, canonical_slug)

    if not artist_tracks and not row:
        raise HTTPException(status_code=404, detail=f"unknown artist: {canonical_slug}")

    # Display name = THIS artist's own participant name, taken from a track where
    # they appear (raw tags are the real tagged metadata). Crucially NOT the whole
    # raw ``artist`` tag of the first track: that may be a collaboration, so
    # "Dua Lipa x Angele" would mislabel the dua-lipa page as the collab. The
    # SQLite ``artists.name`` can be a slug-derived placeholder
    # (``add_artist_facts_batch`` does ``ON CONFLICT SET name = slug.replace``),
    # so it's only a fallback when no tracks are present; finally a title-cased slug.
    name = None
    for t in artist_tracks:
        name = name_for_slug(t.artist, canonical_slug)
        if name:
            break
    name = name or (row[0] if row else None) or canonical_slug.replace("-", " ").title()

    # Group tracks by album
    by_album: dict[str, list[TrackMetadata]] = defaultdict(list)
    for t in artist_tracks:
        by_album[t.album or "—"].append(t)
    # One reactions lookup for all of the artist's tracks (no per-album N+1);
    # drives the gold-star marker (3+ liked tracks) on the Atlas screen.
    reactions = MetadataDB.get_reactions_for_tracks(
        collection, [t.track_id for t in artist_tracks],
    )
    liked_ids = {tid for tid, r in reactions.items() if r == "like"}
    albums: list[ArtistAlbum] = []
    for album_title, tracks in by_album.items():
        tracks = _album_track_sort_key(tracks)
        years = [t.year for t in tracks if t.year]
        # Representative cover = first track's that has one
        cover = next((t.cover_art_path for t in tracks if t.cover_art_path), None)
        albums.append(ArtistAlbum(
            title=album_title,
            year=min(years) if years else None,
            cover_art_path=cover,
            tracks=tracks,
            liked_track_count=sum(1 for t in tracks if t.track_id in liked_ids),
        ))
    albums.sort(key=lambda a: (a.year or 9999, a.title))

    # Prefer refined facts (from AI Indexing) when present; fall back to
    # originals only when no refined row exists for this artist. `is not None`
    # is critical — an explicit [] from refined (AI ran, kept nothing) must
    # short-circuit instead of falling back. Keyed by the same slug the
    # refined_facts task writes under (_slugify_artist of the artist name).
    refined_facts = MetadataDB.get_refined_facts(
        scope="artist", scope_key=canonical_slug, collection_name=collection, lang=lang,
    )
    facts = (
        refined_facts if refined_facts is not None
        else MetadataDB.get_artist_facts(canonical_slug, collection)
    )
    bio = MetadataDB.get_artist_bio(canonical_slug, collection, lang)
    primary_genre = next((t.genre for t in artist_tracks if t.genre), None)

    audiodb = MetadataDB.get_artist_audiodb(canonical_slug, collection) or {}

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
        mood=audiodb.get("mood"),
        country_code=audiodb.get("country_code"),
        country=audiodb.get("country"),
        label=audiodb.get("label"),
        cutout_path=audiodb.get("cutout_path"),
        thumb_path=audiodb.get("thumb_path"),
        audiodb_mbid=audiodb.get("audiodb_mbid"),
    )


@router.get("/{slug}", response_model=ArtistAggregate)
def get_artist(
    slug: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", description="Bio language"),
) -> ArtistAggregate:
    """Aggregate an artist's universe — used by the Atlas screen."""
    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    if request.app.state.db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    canonical_slug = _slugify_artist(slug.replace("-", " "))
    return build_artist_aggregate(request.app.state.db_client, derived, canonical_slug, lang)
