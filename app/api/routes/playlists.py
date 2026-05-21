"""HTTP routes for Plan 19 Custom Playlists CRUD."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.domain.models import (
    PlaylistCreate, PlaylistDetail, PlaylistReorderRequest, PlaylistSummary,
    PlaylistTrackAdd, PlaylistUpdate, PlaylistsResponse,
)
from app.services import playlists_service
from app.services.playlists_service import (
    PlaylistNameCollision, PlaylistNotFound, TrackAlreadyInPlaylist, TrackNotInPlaylist,
)


router = APIRouter(prefix="/playlists", tags=["Playlists"])


def _qdrant(request: Request):
    db_client = getattr(request.app.state, "db_client", None)
    return getattr(db_client, "qdrant", None) if db_client is not None else None


@router.post("", response_model=PlaylistSummary, status_code=201)
def create_playlist(req: PlaylistCreate, request: Request) -> PlaylistSummary:
    try:
        return playlists_service.create(
            req.collection_name, req.name, req.description, qdrant=_qdrant(request)
        )
    except PlaylistNameCollision as e:
        raise HTTPException(409, str(e))


@router.get("", response_model=PlaylistsResponse)
def list_playlists(
    request: Request,
    collection_name: str = Query(..., min_length=1),
    include_track_id: Optional[str] = Query(None),
) -> PlaylistsResponse:
    summaries = playlists_service.list_playlists(
        collection_name,
        include_track_id=include_track_id,
        qdrant=_qdrant(request),
    )
    return PlaylistsResponse(playlists=summaries, collection_name=collection_name)


@router.get("/{playlist_id}", response_model=PlaylistDetail)
def get_playlist(playlist_id: int, request: Request) -> PlaylistDetail:
    try:
        return playlists_service.get(playlist_id, qdrant=_qdrant(request))
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))


@router.put("/{playlist_id}", response_model=PlaylistSummary)
def update_playlist(playlist_id: int, req: PlaylistUpdate, request: Request) -> PlaylistSummary:
    if req.name is None and req.description is None and not req.clear_description:
        raise HTTPException(400, "at least one of name/description/clear_description must be provided")
    try:
        return playlists_service.update(
            playlist_id,
            name=req.name,
            description=req.description,
            clear_description=req.clear_description,
            qdrant=_qdrant(request),
        )
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))
    except PlaylistNameCollision as e:
        raise HTTPException(409, str(e))


@router.delete("/{playlist_id}", status_code=204)
def delete_playlist(playlist_id: int) -> Response:
    try:
        playlists_service.delete(playlist_id)
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))
    return Response(status_code=204)


@router.post("/{playlist_id}/tracks", response_model=PlaylistSummary, status_code=201)
def add_track(playlist_id: int, req: PlaylistTrackAdd, request: Request) -> PlaylistSummary:
    try:
        return playlists_service.add_track(playlist_id, req.track_id, qdrant=_qdrant(request))
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))
    except TrackAlreadyInPlaylist as e:
        raise HTTPException(409, str(e))


@router.delete("/{playlist_id}/tracks/{track_id}", status_code=204)
def remove_track(playlist_id: int, track_id: str) -> Response:
    try:
        playlists_service.remove_track(playlist_id, track_id)
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))
    except TrackNotInPlaylist as e:
        raise HTTPException(404, str(e))
    return Response(status_code=204)


@router.post("/{playlist_id}/reorder", response_model=PlaylistSummary)
def reorder(playlist_id: int, req: PlaylistReorderRequest, request: Request) -> PlaylistSummary:
    try:
        return playlists_service.reorder(playlist_id, req.track_ids, qdrant=_qdrant(request))
    except PlaylistNotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
