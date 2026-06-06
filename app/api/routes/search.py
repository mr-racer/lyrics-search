"""Search endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.domain.models import (
    SearchRequest, SearchResponse, TrackHit, User,
    TrackReactionRequest, TrackReactionResponse,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user, deprecated_collection_warning
from app.resources.model_registry import ModelRegistry
from app.services.audio_streaming import get_streamable_path

router = APIRouter(prefix="/search", tags=["Search"])


# ── Dependencies ───────────────────────────────────────────────────────────────

def get_search_service(request: Request):
    """Dependency: get SearchService from app state."""
    return request.app.state.search_service


def get_db_client(request: Request):
    """Dependency: get LyricsDB from app state."""
    return request.app.state.db_client


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=SearchResponse)
async def search_tracks(
    req: SearchRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> SearchResponse:
    """
    Search tracks by lyrics or audio.
    mode: "text" (dense+BM25), "audio" (CLAP), or "hybrid" (all three)
    """
    service = request.app.state.search_service
    if service is None:
        raise HTTPException(status_code=503, detail="Search service unavailable — is Qdrant running?")

    # Sanitize legacy "null"/"undefined" string values coming from a frontend
    # that ever wrote ``localStorage.setItem('text_model', null)`` (Web Storage
    # API stringifies non-string values).
    requested_model = req.text_model
    if requested_model in ("", "null", "undefined", "None"):
        requested_model = None

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)
    deprecated_collection_warning(req.collection_name, derived, "/search/")

    # Phase B: no more db.lyrics_db.* mutation. The stateless engine resolves the
    # model per call — SearchService._resolve_model_name reads collection_settings
    # (the indexed model) → user.text_model_name → default. An explicit
    # requested_model still wins as a per-request override.
    hits = await service.search(
        query=req.query,
        mode=req.mode,
        text_model=requested_model,
        filters=req.filters,
        limit=req.limit,
        collection_name=derived,
        account_id=current_user.id,
    )

    return SearchResponse(hits=hits, query=req.query, mode=req.mode)


@router.get("/models/text")
async def list_text_models():
    """Return catalog of available text embedding models."""
    return ModelRegistry.list_text_models()


@router.get("/models/loaded")
async def get_loaded_models():
    """Return names of currently loaded text models."""
    return {
        "text_models": ModelRegistry.get_loaded_text_models(),
        "clap_available": ModelRegistry.is_clap_available(),
    }


# ── Audio streaming ──────────────────────────────────────────────────────────

@router.get("/tracks/{track_id}/stream")
async def stream_track(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    collection_name: str | None = Query(None, deprecated=True),
):
    """
    Serve a music file by its Qdrant track_id using FastAPI's FileResponse.
    """
    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)
    deprecated_collection_warning(collection_name, derived, "/search/tracks/{id}/stream")

    db = request.app.state.db_client
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # 1. Look up file_path from Qdrant payload
    try:
        result = db.qdrant.retrieve(
            collection_name=derived,
            ids=[track_id],
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Track not found: {e}")

    if not result:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found in collection {derived}")

    payload = result[0].payload or {}
    file_path = payload.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Track has no file_path in database")

    audio_path = Path(file_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found on disk: {file_path}",
        )

    # ALAC m4a is transcoded to FLAC on the fly (lossless→lossless, bit-exact);
    # other formats served as-is. FileResponse handles Range internally.
    # Phase B §6.6: scope the transcoded-cache namespace by collection_name so
    # two accounts sharing a track_id don't serve each other's cached blob.
    # Phase D: derived is acct_<id>; delete_collection purges under the same key.
    serve_path, content_type = await get_streamable_path(
        account_id=derived, track_id=track_id, file_path=audio_path,
    )

    return FileResponse(
        serve_path,
        media_type=content_type,
        filename=serve_path.name,
    )


# ── Track reactions ────────────────────────────────────────────────────────────

@router.post("/tracks/{track_id}/reaction", response_model=TrackReactionResponse)
async def set_track_reaction(
    track_id: str,
    req: TrackReactionRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> TrackReactionResponse:
    """Set or remove a like/dislike reaction on a track (scoped by collection)."""
    from app.resources.metadata_db import MetadataDB

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)
    deprecated_collection_warning(req.collection_name, derived, "POST /search/tracks/{id}/reaction")

    MetadataDB.init()
    MetadataDB.set_reaction(track_id, derived, req.reaction)

    return TrackReactionResponse(
        track_id=track_id,
        collection_name=derived,
        reaction=req.reaction,
    )


@router.get("/tracks/{track_id}/reaction", response_model=TrackReactionResponse)
async def get_track_reaction(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    collection_name: str | None = Query(None, deprecated=True),
) -> TrackReactionResponse:
    """Get the current reaction for a track in a collection."""
    from app.resources.metadata_db import MetadataDB

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)
    deprecated_collection_warning(collection_name, derived, "GET /search/tracks/{id}/reaction")

    MetadataDB.init()
    reaction = MetadataDB.get_reaction(track_id, derived)

    return TrackReactionResponse(
        track_id=track_id,
        collection_name=derived,
        reaction=reaction,
    )


# ── Legacy stub ────────────────────────────────────────────────────────────────

@router.get("/{track_id}")
async def get_track(track_id: str):
    """Get track by ID."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
