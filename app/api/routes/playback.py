"""Playback event ingestion endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from app.domain.models import (
    PlaybackEventIn,
    PlaybackEventOut,
    RecentTracksResponse,
)
from app.services import playback_service

router = APIRouter(prefix="/playback", tags=["Playback"])


@router.post("/events", response_model=PlaybackEventOut)
async def record_playback_event(request: Request) -> PlaybackEventOut:
    """Record a playback event.

    The browser delivers these via ``navigator.sendBeacon`` on pause / track
    change / pagehide. A beacon (or any cross-origin POST) carrying
    ``Content-Type: application/json`` triggers a CORS preflight that beacons
    can't satisfy, so the client sends the JSON payload as ``text/plain`` — a
    CORS-safelisted type that needs no preflight. We therefore parse the body
    ourselves instead of declaring a JSON Pydantic parameter (which would 422
    on a non-JSON content-type). Plain ``fetch`` with application/json still
    works through the same path.
    """
    raw = await request.body()
    try:
        req = PlaybackEventIn.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=str(e))
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
