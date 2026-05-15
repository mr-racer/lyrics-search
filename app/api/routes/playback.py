"""Playback event ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.domain.models import PlaybackEventIn, PlaybackEventOut
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
