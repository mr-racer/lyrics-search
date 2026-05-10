"""Search endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.domain.models import (
    SearchRequest, SearchResponse, TrackHit,
    TrackReactionRequest, TrackReactionResponse,
)
from app.resources.model_registry import ModelRegistry

router = APIRouter(prefix="/search", tags=["Search"])


# ── MIME types for common audio formats ──────────────────────────────────────

_AUDIO_CONTENT_TYPES: dict[str, str] = {
    ".mp3":  "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a":  "audio/mp4",
    ".aac":  "audio/aac",
    ".ogg":  "audio/ogg",
    ".wav":  "audio/wav",
    ".opus": "audio/opus",
}


def _content_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return _AUDIO_CONTENT_TYPES.get(ext, "application/octet-stream")


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
) -> SearchResponse:
    """
    Search tracks by lyrics or audio.
    mode: "text" (dense+BM25), "audio" (CLAP), or "hybrid" (all three)
    """
    service = request.app.state.search_service
    if service is None:
        raise HTTPException(status_code=503, detail="Search service unavailable — is Qdrant running?")

    # Load text model if specified
    if req.text_model:
        ModelRegistry.load_text_model(req.text_model)

    hits = await service.search(
        query=req.query,
        mode=req.mode,
        text_model=req.text_model,
        filters=req.filters,
        limit=req.limit,
        collection_name=req.collection_name,
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
    collection_name: str,
    request: Request,
):
    """
    Serve a music file by its Qdrant track_id using FastAPI's FileResponse.
    """
    db = request.app.state.db_client
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # 1. Look up file_path from Qdrant payload
    try:
        result = db.qdrant.retrieve(
            collection_name=collection_name,
            ids=[track_id],
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Track not found: {e}")

    if not result:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found in collection {collection_name}")

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

    # 2. Serve via FileResponse (handles Range internally, returns 200/206 as needed)
    content_type = _content_type(file_path)

    return FileResponse(
        audio_path,
        media_type=content_type,
        filename=audio_path.name,
    )


# ── Track reactions ────────────────────────────────────────────────────────────

@router.post("/tracks/{track_id}/reaction", response_model=TrackReactionResponse)
async def set_track_reaction(
    track_id: str,
    req: TrackReactionRequest,
    request: Request,
) -> TrackReactionResponse:
    """Set or remove a like/dislike reaction on a track (scoped by collection)."""
    from app.resources.metadata_db import MetadataDB

    MetadataDB.init()
    MetadataDB.set_reaction(track_id, req.collection_name, req.reaction)

    return TrackReactionResponse(
        track_id=track_id,
        collection_name=req.collection_name,
        reaction=req.reaction,
    )


@router.get("/tracks/{track_id}/reaction", response_model=TrackReactionResponse)
async def get_track_reaction(
    track_id: str,
    collection_name: str,
    request: Request,
) -> TrackReactionResponse:
    """Get the current reaction for a track in a collection."""
    from app.resources.metadata_db import MetadataDB

    MetadataDB.init()
    reaction = MetadataDB.get_reaction(track_id, collection_name)

    return TrackReactionResponse(
        track_id=track_id,
        collection_name=collection_name,
        reaction=reaction,
    )


# ── Legacy stub ────────────────────────────────────────────────────────────────

@router.get("/{track_id}")
async def get_track(track_id: str):
    """Get track by ID."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
