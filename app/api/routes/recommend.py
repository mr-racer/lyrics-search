"""Recommendation routes — autoplay queue, Sonic Sibling (deferred), etc."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.domain.models import AutoplayQueueResponse, ForYouSeedResponse
from app.services import autoplay_service, personalization_service

router = APIRouter(prefix="/recommend", tags=["Recommend"])


@router.get("/autoplay-queue", response_model=AutoplayQueueResponse)
def autoplay_queue(
    request: Request,
    collection: str = Query(..., description="Qdrant collection name"),
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

    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()]
    return autoplay_service.next_queue(
        qdrant_client=db_client.qdrant,
        collection_name=collection,
        seed_track_id=seed_track_id,
        exclude_ids=excluded,
        limit=limit,
    )


@router.get("/for-you-seed", response_model=ForYouSeedResponse)
def for_you_seed(
    request: Request,
    collection: str = Query(..., description="Qdrant collection name"),
) -> ForYouSeedResponse:
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return personalization_service.pick_for_you_seed(
        qdrant_client=db_client.qdrant, collection_name=collection,
    )


@router.get("/sonic-sibling")
def sonic_sibling_stub(
    track_id: str = Query(...),
    collection: str = Query(...),
):
    """Slot reserved for the Sonic Sibling endpoint (deferred to a future plan).

    See PLATFORM_DESIGN.md §5.1 and Plan 3 spec § Non-goals.
    """
    raise HTTPException(
        status_code=501,
        detail="Sonic Sibling is deferred — see PLATFORM_DESIGN.md §5.1 for status.",
    )
