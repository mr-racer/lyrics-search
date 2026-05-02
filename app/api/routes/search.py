"""Search endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.domain.models import SearchRequest, SearchResponse, TrackHit
from app.resources.model_registry import ModelRegistry

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_service(request: Request):
    """Dependency: get SearchService from app state."""
    return request.app.state.search_service


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


@router.get("/{track_id}")
async def get_track(track_id: str):
    """Get track by ID."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
