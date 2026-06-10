"""Playback event ingestion endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError

from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.domain.models import (
    PlaybackEventIn,
    PlaybackEventOut,
    RecentTracksResponse,
    User,
)
from app.services import playback_service

router = APIRouter(prefix="/playback", tags=["Playback"])


@router.post("/events", response_model=PlaybackEventOut)
async def record_playback_event(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> PlaybackEventOut:
    """Record a playback event.

    The browser delivers these via ``navigator.sendBeacon`` on pause / track
    change / pagehide. A beacon (or any cross-origin POST) carrying
    ``Content-Type: application/json`` triggers a CORS preflight that beacons
    can't satisfy, so the client sends the JSON payload as ``text/plain`` — a
    CORS-safelisted type that needs no preflight. We therefore parse the body
    ourselves instead of declaring a JSON Pydantic parameter (which would 422
    on a non-JSON content-type). Plain ``fetch`` with application/json still
    works through the same path.

    Phase D (D-soft): ``collection_name`` in the request body is accepted for
    backward compat but ignored. The server derives the collection from the JWT.
    """
    raw = await request.body()
    try:
        req = PlaybackEventIn.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    derived = derive_collection_for_user(current_user)
    new_id = playback_service.record_event(
        session_id=req.session_id,
        collection_name=derived,
        track_id=req.track_id,
        played_sec=req.played_sec,
        total_dur=req.total_dur,
        interacted=req.interacted,
    )
    return PlaybackEventOut(id=new_id)


@router.get("/recent", response_model=RecentTracksResponse)
async def get_recent(
    request: Request,
    current_user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
) -> RecentTracksResponse:
    """Return recently played tracks for the current user's collection.

    Phase D (D-soft): ``collection_name`` query param is accepted for backward
    compat but ignored. The server derives the collection from the JWT.
    """
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return RecentTracksResponse(tracks=[], collection_name=derived)
    return playback_service.get_recent(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        limit=limit,
    )
