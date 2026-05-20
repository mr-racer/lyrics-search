"""Playback event ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.domain.models import (
    PlaybackEventIn,
    PlaybackEventOut,
    RecentTracksResponse,
)
from app.services import playback_service

router = APIRouter(prefix="/playback", tags=["Playback"])


@router.post("/events", response_model=PlaybackEventOut)
def record_playback_event(req: PlaybackEventIn) -> PlaybackEventOut:
    new_id = playback_service.record_event(
        session_id=req.session_id,
        collection_name=req.collection_name,
        track_id=req.track_id,
        played_sec=req.played_sec,
        total_dur=req.total_dur,
    )
    return PlaybackEventOut(id=new_id)


@router.get("/recent", response_model=RecentTracksResponse)
async def get_recent(
    request: Request,
    collection_name: str = Query(..., description="Collection name (required)"),
    limit: int = Query(50, ge=1, le=200),
) -> RecentTracksResponse:
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return RecentTracksResponse(tracks=[], collection_name=collection_name)
    return playback_service.get_recent(
        qdrant_client=db_client.qdrant,
        collection_name=collection_name,
        limit=limit,
    )
