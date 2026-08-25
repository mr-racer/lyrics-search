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

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    # ``req.text_model`` is accepted and ignored — there is one embedding model
    # app-wide now, and rejecting the field would 422 every cached PWA client.
    hits = await service.search(
        query=req.query,
        mode=req.mode,
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
    # search_catalog runs sync Qdrant queries — keep them off the event loop.
    hits = await asyncio.get_running_loop().run_in_executor(
        None, catalog_search_service.search_catalog, db_client.qdrant, collection, q, limit,
    )
    return [CatalogHit(**h) for h in hits]


@router.get("/models/text")
async def list_text_models():
    """The pinned text embedding model. A list of one, kept as a list so
    existing clients keep parsing it."""
    return {ModelRegistry.TEXT_MODEL_NAME: {"dim": ModelRegistry.VECTOR_DIM}}


@router.get("/models/loaded")
async def get_loaded_models():
    """Report whether the text model is resident yet."""
    return {
        "text_models": ([ModelRegistry.TEXT_MODEL_NAME]
                        if ModelRegistry.is_text_model_loaded() else []),
        # Which device the text model actually landed on, and why. A silent CPU
        # fallback on a GPU box is the failure that looks like success.
        "text_device": ModelRegistry.text_device(),
        "clap_available": ModelRegistry.is_clap_available(),
        # The assistant ranks on three legs and degrades quietly to fewer. That
        # is the right behaviour and the wrong thing to be silent about: the
        # sparse leg once failed every encode for hours while its weights sat
        # resident, and the only visible symptom was worse answers. The
        # booleans say what is loaded; the counters say what is actually
        # working.
        "retrieval": ModelRegistry.retrieval_status(),
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
        # Sync qdrant-client — threaded so a lyrics lookup (fired on every
        # track change) never stalls concurrent audio byte delivery.
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: db.qdrant.retrieve(
                collection_name=derived,
                ids=[track_id],
                with_payload=True,
                with_vectors=False,
            ),
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
