"""Search endpoints."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.domain.models import (
    SearchRequest, SearchResponse, User, CatalogHit,
    TrackReactionRequest, TrackReactionResponse,
)
from app.api.dependencies import get_current_user, get_user_for_stream
from app.api.helpers import derive_collection_for_user
from app.resources.model_registry import ModelRegistry
from app.services.audio_streaming import (
    get_streamable_path,
    get_cached_source,
    put_cached_source,
    drop_source_for_tracks,
)

router = APIRouter(prefix="/search", tags=["Search"])

# Separate router for the audio stream endpoint: it must NOT sit behind the
# blanket get_current_user gate (main.py auth_gate) because <audio> elements
# can't send an Authorization header. Its own get_user_for_stream dependency
# accepts Bearer OR a short-lived ?st= stream token — auth is still mandatory.
stream_router = APIRouter(prefix="/search", tags=["Search"])


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


@router.get("/catalog", response_model=list[CatalogHit])
async def catalog_search(
    request: Request,
    q: str = Query("", description="Query — title / album / artist, in one field."),
    limit: int = Query(12, ge=1, le=50),
    current_user: User = Depends(get_current_user),
) -> list[CatalogHit]:
    """Non-LLM catalog search: artists / albums / songs by name, best-match first.

    Separate from the semantic ``POST /search/`` — matches names (BM25F over
    title/album/artist), not lyrics. The collection is derived from the JWT.
    """
    from app.services import catalog_search_service

    db_client = getattr(request.app.state, "db_client", None)
    if db_client is None:
        return []
    collection = derive_collection_for_user(current_user)
    hits = catalog_search_service.search_catalog(db_client.qdrant, collection, q, limit)
    return [CatalogHit(**h) for h in hits]


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

@stream_router.get("/tracks/{track_id}/stream")
async def stream_track(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_user_for_stream),
):
    """
    Serve a music file by its Qdrant track_id using FastAPI's FileResponse.
    """
    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    db = request.app.state.db_client
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # 1. Resolve file_path (+codec hint). <audio> issues many Range requests per
    # track, so the Qdrant lookup is memoized; the cold miss runs in a thread —
    # the sync qdrant-client would otherwise block the event loop and stutter
    # every concurrent stream.
    async def _lookup_source() -> tuple[str, str | None]:
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: db.qdrant.retrieve(
                    collection_name=derived,
                    ids=[track_id],
                    with_payload=True,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Track not found: {e}")

        if not result:
            raise HTTPException(status_code=404, detail=f"Track {track_id} not found in collection {derived}")

        payload = result[0].payload or {}
        fp = payload.get("file_path")
        if not fp:
            raise HTTPException(status_code=404, detail="Track has no file_path in database")

        put_cached_source(derived, track_id, fp, payload.get("audio_codec"))
        return fp, payload.get("audio_codec")

    cached = get_cached_source(derived, track_id)
    from_cache = cached is not None
    file_path, codec = cached if cached else await _lookup_source()

    audio_path = Path(file_path)
    if not audio_path.exists() and from_cache:
        # Stale cache (file moved/re-indexed) — retry with a fresh lookup.
        drop_source_for_tracks(derived, [track_id])
        file_path, codec = await _lookup_source()
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
        account_id=derived, track_id=track_id, file_path=audio_path, codec=codec,
    )

    # private: the URL carries a per-user stream token; max-age matches the
    # token TTL. Lets the browser serve repeat Range requests (and the client's
    # full-file warmup fetch) from disk cache instead of revalidating.
    return FileResponse(
        serve_path,
        media_type=content_type,
        filename=serve_path.name,
        headers={"Cache-Control": "private, max-age=3600"},
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
) -> TrackReactionResponse:
    """Get the current reaction for a track in a collection."""
    from app.resources.metadata_db import MetadataDB

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    MetadataDB.init()
    reaction = MetadataDB.get_reaction(track_id, derived)

    return TrackReactionResponse(
        track_id=track_id,
        collection_name=derived,
        reaction=reaction,
    )


# ── Track lyrics (on-demand) ───────────────────────────────────────────────────

@router.get("/tracks/{track_id}/lyrics")
async def get_track_lyrics(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return a track's full lyrics by id.

    The player fetches lyrics here on every track change so EVERY playback
    source shows them — search hits carry `lyrics` inline, but autoplay-queue,
    recently-played, liked and stream payloads do not. The lyrics live in the
    Qdrant payload regardless, so one lookup covers all sources.
    """
    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    db = request.app.state.db_client
    if db is None or db.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        result = db.qdrant.retrieve(
            collection_name=derived,
            ids=[track_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Track not found: {e}")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Track {track_id} not found in collection {derived}",
        )

    raw = (result[0].payload or {}).get("lyrics") or ""
    return {"track_id": track_id, "lyrics": raw.strip() or None}


# ── Legacy stub ────────────────────────────────────────────────────────────────

@router.get("/{track_id}")
async def get_track(track_id: str):
    """Get track by ID."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
