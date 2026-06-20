"""Recommendation routes — autoplay queue, personalized stream («Поток»)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from app.domain.models import (
    AIPlaylistIn,
    AIPlaylistResponse,
    AIPlaylistStep,
    AIPlaylistTrack,
    AutoplayQueueResponse,
    AxisPlaylistIn,
    AxisPlaylistResponse,
    ProfileEnrichIn,
    ProfileEnrichResponse,
    ProfileIsland,
    ProfileIslandTrack,
    SimilarTracksResponse,
    StreamNextResponse,
    StreamProfileResponse,
    StreamSettingsIn,
    StreamTrack,
    User,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.resources.metadata_db import MetadataDB
from app.services import autoplay_service, recsys_ai_service, stream_service
from app.services._payload_coerce import coerce_float, coerce_year
from app.services.settings_service import settings_service

logger = logging.getLogger(__name__)

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


@router.get("/profile", response_model=StreamProfileResponse)
def stream_profile(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", min_length=2, max_length=5),
) -> StreamProfileResponse:
    """Explainable long-term taste: 6 axes (z + level), confidence, islands.

    Pure long-term (no session blending) — the stable «кто я как слушатель»
    view. LLM enrichment (portrait + island names) is attached from cache when
    fresh (populated by POST /recommend/profile/ai-enrich; a stale hash means
    the taste drifted and the frontend should offer a regenerate button).
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    result = stream_service.long_term_profile(
        qdrant_client=db_client.qdrant, collection_name=derived,
    )
    enrich = recsys_ai_service.get_cached_enrichment(derived, lang, result["islands"]) or {}
    island_names = enrich.get("island_names") or {}
    islands = [
        ProfileIsland(
            track_id=i["track_id"], weight=i["weight"],
            tracks=[ProfileIslandTrack(**t) for t in i["tracks"]],
            name=island_names.get(i["track_id"]),
        )
        for i in result["islands"]
    ]
    stored_share = MetadataDB.get_stream_liked_share(derived)
    return StreamProfileResponse(
        axes=result["axes"],
        confidence=result["confidence"],
        n_signals=result["n_signals"],
        islands=islands,
        portrait=enrich.get("portrait"),
        headline=enrich.get("headline"),
        axis_stats_source=result["axis_stats_source"],
        liked_share=stored_share if stored_share is not None else 0.3,
    )


@router.get("/taste-vibe")
def taste_vibe(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", min_length=2, max_length=5),
) -> dict:
    """One short "wave" phrase describing the listener's current taste, for the
    For-You hero (replaces a concrete song title).

    Returns a fresh cached AI phrase when available; otherwise an instant
    deterministic phrase and — when AI is enabled — schedules generation so the
    next load gets the LLM version. Never blocks the hero on the LLM.
    Shape: ``{"phrase": str | None, "source": "ai" | "fallback" | None}``.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return {"phrase": None, "source": None}
    derived = derive_collection_for_user(current_user)
    try:
        result = recsys_ai_service.taste_vibe_cached_or_fallback(
            qdrant_client=db_client.qdrant, collection_name=derived, lang=lang,
        )
    except Exception:
        logger.exception("[taste-vibe] read path failed")
        return {"phrase": None, "source": None}
    if result.pop("needs_generation", False) and settings_service.ai_available():
        background_tasks.add_task(
            recsys_ai_service.generate_taste_vibe,
            qdrant_client=db_client.qdrant, collection_name=derived, lang=lang,
        )
    return result


@router.post("/profile/ai-enrich", response_model=ProfileEnrichResponse)
async def profile_ai_enrich(
    body: ProfileEnrichIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> ProfileEnrichResponse:
    """Generate (and cache) the LLM listener portrait + island names (AI mode)."""
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        result = await recsys_ai_service.enrich_profile(
            qdrant_client=db_client.qdrant,
            collection_name=derived,
            lang=body.lang,
            llm_base_url=body.llm_base_url,
            llm_model=body.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM enrichment failed: {e}")
    return ProfileEnrichResponse(
        portrait=result["portrait"], island_names=result["island_names"],
        headline=result.get("headline"),
    )


@router.post("/ai-playlist", response_model=AIPlaylistResponse)
async def ai_playlist_route(
    body: AIPlaylistIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AIPlaylistResponse:
    """One wish → curated playlist (AI mode): plan → execute → select."""
    db_client = request.app.state.db_client
    search_service = getattr(request.app.state, "search_service", None)
    if db_client is None or db_client.qdrant is None or search_service is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        result = await recsys_ai_service.ai_playlist(
            search_service=search_service,
            qdrant_client=db_client.qdrant,
            collection_name=derived,
            prompt=body.prompt,
            lang=body.lang,
            limit=body.limit,
            llm_base_url=body.llm_base_url,
            llm_model=body.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI playlist failed: {e}")
    tracks = [
        AIPlaylistTrack(
            track_id=t["track_id"],
            title=t["title"],
            artist=t["artist"],
            album=t.get("album"),
            year=coerce_year(t.get("year")),
            genre=t.get("genre"),
            duration_sec=coerce_float(t.get("duration")) or 0.0,
            file_path=t.get("file_path") or "",
            cover_art_path=t.get("cover_art_path"),
            reason=t.get("reason"),
            source_tool=t.get("tool"),
        )
        for t in result["tracks"]
    ]
    return AIPlaylistResponse(
        title=result["title"],
        steps=[AIPlaylistStep(**s) for s in result["steps"]],
        tracks=tracks,
    )


@router.post("/axis-playlist", response_model=AxisPlaylistResponse)
def axis_playlist(
    body: AxisPlaylistIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AxisPlaylistResponse:
    """Rank the collection against target z-values from the radar knobs."""
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    # Clamp targets to a sane z-range; unknown axis names are ignored by the
    # service (it iterates AXIS_NAMES), so no need to 422 on extras.
    targets = {k: max(-3.0, min(3.0, v)) for k, v in body.targets.items()}
    result = stream_service.axis_playlist(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        axis_targets=targets,
        limit=body.limit,
    )
    return AxisPlaylistResponse(
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
        diagnostics=result["diagnostics"],
    )


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
