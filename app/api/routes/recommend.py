"""Recommendation routes — autoplay queue, personalized stream («Поток»)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.domain.models import (
    AutoplayQueueResponse,
    SimilarTracksResponse,
    StreamNextResponse,
    StreamSettingsIn,
    StreamTrack,
    User,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.resources.metadata_db import MetadataDB
from app.services import autoplay_service, stream_service
from app.services._payload_coerce import coerce_float, coerce_year

router = APIRouter(prefix="/recommend", tags=["Recommend"])


@router.get("/autoplay-queue", response_model=AutoplayQueueResponse)
def autoplay_queue(
    request: Request,
    current_user: User = Depends(get_current_user),
    seed_track_id: str = Query(..., description="Track id used as similarity anchor"),
    exclude_ids: str | None = Query(
        None,
        description="Comma-separated track ids to suppress (e.g., already played this session). "
                    "Capped at 200 server-side.",
    ),
    limit: int = Query(20, ge=1, le=50),
) -> AutoplayQueueResponse:
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()]
    return autoplay_service.next_queue(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        seed_track_id=seed_track_id,
        exclude_ids=excluded,
        limit=limit,
    )


def _candidate_to_stream_track(c: "stream_service.StreamCandidate") -> StreamTrack:
    p = c.payload or {}
    return StreamTrack(
        track_id=c.track_id,
        title=p.get("title") or "—",
        artist=p.get("artist") or "—",
        album=p.get("album"),
        year=coerce_year(p.get("year")),
        genre=p.get("genre"),
        duration_sec=coerce_float(p.get("duration")) or 0.0,
        file_path=p.get("file_path") or "",
        cover_art_path=p.get("cover_art_path"),
        pool=c.pool,
        anchor_track_id=c.anchor_track_id,
        axis_match=c.axis_match,
        score=round(c.score, 4) if c.score is not None else None,
    )


@router.get("/stream/next", response_model=StreamNextResponse)
def stream_next(
    request: Request,
    current_user: User = Depends(get_current_user),
    session_id: str = Query(..., min_length=1),
    n: int = Query(stream_service.DEFAULT_CHUNK_N, ge=1, le=10),
    liked_share: float | None = Query(
        None, ge=0.0, le=1.0,
        description="Liked/new slider override; falls back to the persisted "
                    "collection setting, then the default 0.30.",
    ),
    exclude_ids: str | None = Query(
        None,
        description="Comma-separated track ids the client has prefetched but "
                    "not yet played (stateless-gap cover). Capped at 50.",
    ),
) -> StreamNextResponse:
    """Next chunk of the session-aware personalized stream (design §7).

    Stateless: profiles are rebuilt from playback_events + track_reactions on
    every call, so the stream survives server restarts and tab reloads.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)

    if liked_share is None:
        liked_share = MetadataDB.get_stream_liked_share(derived)

    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()][:50]
    result = stream_service.next_chunk(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        session_id=session_id,
        n=n,
        liked_share=liked_share,
        exclude_ids=excluded,
    )
    return StreamNextResponse(
        session_id=session_id,
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
        diagnostics=result["diagnostics"],
    )


@router.put("/stream/settings")
def stream_settings(
    body: StreamSettingsIn,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Persist the liked/new slider for the user's collection."""
    derived = derive_collection_for_user(current_user)
    MetadataDB.set_stream_liked_share(derived, body.liked_share)
    return {"liked_share": body.liked_share}


@router.get("/similar", response_model=SimilarTracksResponse)
def similar(
    request: Request,
    current_user: User = Depends(get_current_user),
    track_id: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    exclude_ids: str | None = Query(
        None, description="Comma-separated track ids to suppress. Capped at 100.",
    ),
) -> SimilarTracksResponse:
    """Tracks similar to a seed: CLAP neighbors re-ranked by sonic-axis closeness.

    Replaces the old frontend hack (text search for «artist title» in audio
    mode) and doubles as the ai-playlist agent's similar_tracks tool.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()][:100]
    result = stream_service.similar_tracks(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        seed_track_id=track_id,
        limit=limit,
        exclude_ids=excluded,
    )
    return SimilarTracksResponse(
        seed_track_id=result["seed_track_id"],
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
    )


@router.get("/sonic-sibling")
def sonic_sibling_stub(
    track_id: str = Query(...),
    current_user: User = Depends(get_current_user),
):
    """Slot reserved for the Sonic Sibling endpoint (deferred to a future plan).

    See PLATFORM_DESIGN.md §5.1 and Plan 3 spec § Non-goals.
    """
    raise HTTPException(
        status_code=501,
        detail="Sonic Sibling is deferred — see PLATFORM_DESIGN.md §5.1 for status.",
    )
