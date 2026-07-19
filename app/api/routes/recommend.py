"""Recommendation routes — autoplay queue, personalized stream («Поток»)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.domain.models import (
    AIPlaylistIn,
    AIPlaylistResponse,
    AIPlaylistStep,
    AIPlaylistTrack,
    AutoplayQueueResponse,
    AxisPlaylistIn,
    AxisPlaylistResponse,
    ProfileEnrichIn,
    ProfileEnrichResponse,
    ProfileIsland,
    ProfileIslandTrack,
    SimilarTracksResponse,
    StreamNextResponse,
    StreamProfileResponse,
    StreamSettingsIn,
    StreamTrack,
    TasteSignalIn,
    TasteSignalOut,
    TasteSignalState,
    TasteSignalStateOut,
    User,
    VibeAlbumSuggestion,
    VibeAlbumSuggestionsResponse,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.resources.metadata_db import MetadataDB
from app.services import autoplay_service, recsys_ai_service, stream_service
from app.services._payload_coerce import coerce_float, coerce_year
from app.services.artist_split import artist_refs_for_track, display_title_for_track
from app.services.settings_service import settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["Recommend"])


@router.get("/autoplay-queue", response_model=AutoplayQueueResponse)
def autoplay_queue(
    request: Request,
    current_user: User = Depends(get_current_user),
    seed_track_id: str = Query(..., description="Track id used as similarity anchor"),
    exclude_ids: str | None = Query(
        None,
        description="Comma-separated track ids to suppress (e.g., already played this session). "
                    "Capped at 200 server-side.",
    ),
    limit: int = Query(20, ge=1, le=50),
) -> AutoplayQueueResponse:
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Phase D-soft: derive collection from JWT user; ignore client-supplied value.
    derived = derive_collection_for_user(current_user)

    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()]
    return autoplay_service.next_queue(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        seed_track_id=seed_track_id,
        exclude_ids=excluded,
        limit=limit,
    )


def _candidate_to_stream_track(c: "stream_service.StreamCandidate") -> StreamTrack:
    p = c.payload or {}
    return StreamTrack(
        track_id=c.track_id,
        title=p.get("title") or "—",
        title_display=display_title_for_track(p),
        artist=p.get("artist") or "—",
        album=p.get("album"),
        year=coerce_year(p.get("year")),
        genre=p.get("genre"),
        duration_sec=coerce_float(p.get("duration")) or 0.0,
        file_path=p.get("file_path") or "",
        cover_art_path=p.get("cover_art_path"),
        pool=c.pool,
        anchor_track_id=c.anchor_track_id,
        axis_match=c.axis_match,
        score=round(c.score, 4) if c.score is not None else None,
        artist_refs=artist_refs_for_track(p),
    )


@router.get("/stream/next", response_model=StreamNextResponse)
def stream_next(
    request: Request,
    current_user: User = Depends(get_current_user),
    session_id: str = Query(..., min_length=1),
    n: int = Query(stream_service.DEFAULT_CHUNK_N, ge=1, le=10),
    liked_share: float | None = Query(
        None, ge=0.0, le=1.0,
        description="Liked/new slider override; falls back to the persisted "
                    "collection setting, then the default 0.30.",
    ),
    exclude_ids: str | None = Query(
        None,
        description="Comma-separated track ids the client has prefetched but "
                    "not yet played (stateless-gap cover). Capped at 50.",
    ),
) -> StreamNextResponse:
    """Next chunk of the session-aware personalized stream (design §7).

    Stateless: profiles are rebuilt from playback_events + track_reactions on
    every call, so the stream survives server restarts and tab reloads.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)

    if liked_share is None:
        liked_share = MetadataDB.get_stream_liked_share(derived)

    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()][:50]
    result = stream_service.next_chunk(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        session_id=session_id,
        n=n,
        liked_share=liked_share,
        exclude_ids=excluded,
    )
    return StreamNextResponse(
        session_id=session_id,
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
        diagnostics=result["diagnostics"],
        session_adaptation=result.get("session_adaptation"),
    )


@router.put("/stream/settings")
def stream_settings(
    body: StreamSettingsIn,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Persist the liked/new slider for the user's collection."""
    derived = derive_collection_for_user(current_user)
    MetadataDB.set_stream_liked_share(derived, body.liked_share)
    return {"liked_share": body.liked_share}


@router.post("/taste-signal", response_model=TasteSignalOut)
def taste_signal(
    body: TasteSignalIn,
    current_user: User = Depends(get_current_user),
) -> TasteSignalOut:
    """Record an огонёк/вода gesture for the user's collection.

    fire = «давай побольше такого» — a strong ephemeral wave anchor on the
    track's CLAP vector; water = «остудить» — a soft ephemeral demotion of the
    track and its neighbors. Both fade over ~4h × 30→50 session tracks. The
    collection is derived from the JWT (per-account isolation) — the client
    cannot target another account's collection.
    """
    derived = derive_collection_for_user(current_user)
    new_id = MetadataDB.record_taste_signal(
        session_id=body.session_id,
        collection_name=derived,
        track_id=body.track_id,
        kind=body.kind,
    )
    return TasteSignalOut(id=new_id)


@router.get("/taste-signal/state", response_model=TasteSignalStateOut)
def taste_signal_state(
    track_ids: str = Query(..., description="Comma-separated track ids"),
    current_user: User = Depends(get_current_user),
) -> TasteSignalStateOut:
    """Active огонёк/вода per track for the frontend meter/lock.

    For each requested track with a signal, returns the newest reaction
    (latest-wins), its remaining «заряд» (contribution ∈ [0,1], half-life 1 day)
    and whether the same-kind button is locked (contribution still > 0.5). The
    collection is derived from the JWT — the client cannot query another account.
    """
    derived = derive_collection_for_user(current_user)
    ids = [t.strip() for t in track_ids.split(",") if t.strip()]
    latest = MetadataDB.get_latest_taste_signals(derived, ids)
    now = datetime.utcnow()
    states: dict[str, TasteSignalState] = {}
    for tid, (kind, created_iso) in latest.items():
        if kind not in ("fire", "water"):
            continue
        created = stream_service._parse_iso(created_iso)
        if created is None:
            continue
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        contribution = stream_service.reaction_contribution(age_days)
        states[tid] = TasteSignalState(
            kind=kind,
            contribution=round(contribution, 4),
            locked=contribution > 0.5,
        )
    return TasteSignalStateOut(states=states)


@router.get("/profile", response_model=StreamProfileResponse)
def stream_profile(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", min_length=2, max_length=5),
) -> StreamProfileResponse:
    """Explainable long-term taste: 6 axes (z + level), confidence, islands.

    Pure long-term (no session blending) — the stable «кто я как слушатель»
    view. LLM enrichment (portrait + island names) is attached from cache when
    fresh (populated by POST /recommend/profile/ai-enrich; a stale hash means
    the taste drifted and the frontend should offer a regenerate button).
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    result = stream_service.long_term_profile(
        qdrant_client=db_client.qdrant, collection_name=derived,
    )
    enrich = recsys_ai_service.get_cached_enrichment(derived, lang, result["islands"]) or {}
    island_names = enrich.get("island_names") or {}
    islands = [
        ProfileIsland(
            track_id=i["track_id"], weight=i["weight"],
            tracks=[ProfileIslandTrack(**t) for t in i["tracks"]],
            name=island_names.get(i["track_id"]),
        )
        for i in result["islands"]
    ]
    # «Вайбики» reuse the island shape. LLM names are attached from cache when
    # fresh (populated by POST /recommend/vibes/ai-name; the membership hash
    # self-invalidates as the vibes churn) — frontend falls back to
    # genre/artist naming while a name is missing.
    vibe_names = recsys_ai_service.get_cached_vibe_names(
        derived, lang, result.get("vibes", []),
    )
    vibes = [
        ProfileIsland(
            track_id=v["track_id"], weight=v["weight"],
            tracks=[ProfileIslandTrack(**t) for t in v["tracks"]],
            name=vibe_names.get(v["track_id"]),
        )
        for v in result.get("vibes", [])
    ]
    stored_share = MetadataDB.get_stream_liked_share(derived)
    return StreamProfileResponse(
        axes=result["axes"],
        confidence=result["confidence"],
        n_signals=result["n_signals"],
        islands=islands,
        vibes=vibes,
        portrait=enrich.get("portrait"),
        headline=enrich.get("headline"),
        axis_stats_source=result["axis_stats_source"],
        liked_share=stored_share if stored_share is not None else 0.3,
    )


@router.get("/taste-vibe")
async def taste_vibe(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", min_length=2, max_length=5),
) -> dict:
    """One short "wave" phrase describing the listener's current taste, for the
    For-You hero (replaces a concrete song title).

    Returns a fresh cached AI phrase when available. On a cache miss, if AI is
    enabled, the LLM phrase is generated SYNCHRONOUSLY (and cached) so the hero
    shows the real AI line on first load — no «one navigation behind» lag. When
    AI is off, or the LLM fails, returns the instant deterministic phrase.
    Shape: ``{"phrase": str | None, "source": "ai" | "fallback" | None}``.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return {"phrase": None, "source": None}
    derived = derive_collection_for_user(current_user)
    try:
        # Sync Qdrant + SQLite inside — off the event loop, or every request
        # stalls while Qdrant is busy with an indexing job.
        result = await asyncio.to_thread(
            recsys_ai_service.taste_vibe_cached_or_fallback,
            qdrant_client=db_client.qdrant, collection_name=derived, lang=lang,
        )
    except Exception:
        logger.exception("[taste-vibe] read path failed")
        return {"phrase": None, "source": None}
    if result.pop("needs_generation", False) and settings_service.ai_available():
        try:
            generated = await recsys_ai_service.generate_taste_vibe(
                qdrant_client=db_client.qdrant, collection_name=derived, lang=lang,
            )
            if generated and generated.get("phrase"):
                return generated
        except Exception:
            logger.exception("[taste-vibe] synchronous generation failed")
    return result


@router.post("/profile/ai-enrich", response_model=ProfileEnrichResponse)
async def profile_ai_enrich(
    body: ProfileEnrichIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> ProfileEnrichResponse:
    """Generate (and cache) the LLM listener portrait + island names (AI mode)."""
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        result = await recsys_ai_service.enrich_profile(
            qdrant_client=db_client.qdrant,
            collection_name=derived,
            lang=body.lang,
            llm_base_url=body.llm_base_url,
            llm_model=body.llm_model,
        )
    except Exception as e:
        logger.exception("[profile-ai-enrich] failed for %s", derived)
        raise HTTPException(status_code=502, detail=f"LLM enrichment failed: {e}")
    return ProfileEnrichResponse(
        portrait=result["portrait"], island_names=result["island_names"],
        headline=result.get("headline"),
    )


@router.post("/vibes/ai-name")
async def vibes_ai_name(
    body: ProfileEnrichIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Generate (and cache) LLM names for the current «вайбики» (AI mode).

    Mirrors /profile/ai-enrich: the frontend calls this once when the profile
    arrives with unnamed vibes, then merges the names in place. Shape:
    ``{"vibe_names": {vibe_track_id: name}}``.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        return await recsys_ai_service.generate_vibe_names(
            qdrant_client=db_client.qdrant,
            collection_name=derived,
            lang=body.lang,
            llm_base_url=body.llm_base_url,
            llm_model=body.llm_model,
        )
    except Exception as e:
        logger.exception("[vibes-ai-name] failed for %s", derived)
        raise HTTPException(status_code=502, detail=f"LLM vibe naming failed: {e}")


@router.get("/vibes/album-suggestions", response_model=VibeAlbumSuggestionsResponse)
def vibes_album_suggestions(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", min_length=2, max_length=5),
) -> VibeAlbumSuggestionsResponse:
    """Album picks for the library rail: per current «вайбик», the album whose
    mean CLAP is closest to the vibe's centroid but which is not represented
    inside the vibe. ≤2 per vibe, ≤6 total, round-robin for diversity (service
    docstring has the full policy). Vibe names are attached from the ai-name
    cache when fresh — no LLM call happens here (pure vector math), so the
    rail works even with AI off; names just stay empty then.
    """
    db_client = request.app.state.db_client
    library_service = request.app.state.library_service
    if db_client is None or db_client.qdrant is None or library_service is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        albums = library_service.get_albums(
            qdrant_client=db_client.qdrant, collection_name=derived,
        ).albums
    except Exception:
        # Fresh account / collection not indexed yet — an empty rail, not a 500.
        logger.exception("[vibe-albums] get_albums failed for %s", derived)
        albums = []
    result = stream_service.vibe_album_suggestions(
        qdrant_client=db_client.qdrant, collection_name=derived, albums=albums,
    )
    vibe_names = recsys_ai_service.get_cached_vibe_names(
        derived, lang, result.get("vibes", []),
    )
    return VibeAlbumSuggestionsResponse(suggestions=[
        VibeAlbumSuggestion(**s, vibe_name=vibe_names.get(s["vibe_track_id"]))
        for s in result.get("suggestions", [])
    ])


@router.post("/ai-playlist", response_model=AIPlaylistResponse)
async def ai_playlist_route(
    body: AIPlaylistIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AIPlaylistResponse:
    """One wish → curated playlist (AI mode): plan → execute → select."""
    db_client = request.app.state.db_client
    search_service = getattr(request.app.state, "search_service", None)
    if db_client is None or db_client.qdrant is None or search_service is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    try:
        result = await recsys_ai_service.ai_playlist(
            search_service=search_service,
            qdrant_client=db_client.qdrant,
            collection_name=derived,
            prompt=body.prompt,
            lang=body.lang,
            limit=body.limit,
            llm_base_url=body.llm_base_url,
            llm_model=body.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI playlist failed: {e}")
    return _build_ai_playlist_response(result)


def _build_ai_playlist_response(result: dict) -> AIPlaylistResponse:
    """Shape the ai_playlist service dict into the API contract (shared by the
    plain and the streaming route)."""
    tracks = [
        AIPlaylistTrack(
            track_id=t["track_id"],
            title=t["title"],
            title_display=display_title_for_track(t),
            artist=t["artist"],
            album=t.get("album"),
            year=coerce_year(t.get("year")),
            genre=t.get("genre"),
            duration_sec=coerce_float(t.get("duration")) or 0.0,
            file_path=t.get("file_path") or "",
            cover_art_path=t.get("cover_art_path"),
            reason=t.get("reason"),
            source_tool=t.get("tool"),
            artist_refs=artist_refs_for_track(t),
        )
        for t in result["tracks"]
    ]
    return AIPlaylistResponse(
        title=result["title"],
        steps=[AIPlaylistStep(**s) for s in result["steps"]],
        tracks=tracks,
    )


@router.post("/ai-playlist/stream")
async def ai_playlist_stream_route(
    body: AIPlaylistIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming twin of /ai-playlist: NDJSON progress events while the
    pipeline runs, then a final ``result`` (or ``error``) line.

    Line shapes: ``{"type":"status","stage":...}`` /
    ``{"type":"result","payload":<AIPlaylistResponse>}`` /
    ``{"type":"error","message":...}``. EventSource can't POST, hence NDJSON
    over fetch-streaming rather than SSE.
    """
    db_client = request.app.state.db_client
    search_service = getattr(request.app.state, "search_service", None)
    if db_client is None or db_client.qdrant is None or search_service is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_status(event: dict) -> None:
        item = {"type": "status", **event}
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            # same-thread emit: enqueue immediately so ordering vs. the final
            # result is preserved (call_soon would defer past a ready put())
            queue.put_nowait(item)
        else:
            # tool code emitting from a worker thread
            loop.call_soon_threadsafe(queue.put_nowait, item)

    async def run() -> None:
        try:
            result = await recsys_ai_service.ai_playlist(
                search_service=search_service,
                qdrant_client=db_client.qdrant,
                collection_name=derived,
                prompt=body.prompt,
                lang=body.lang,
                limit=body.limit,
                llm_base_url=body.llm_base_url,
                llm_model=body.llm_model,
                on_status=on_status,
            )
            payload = _build_ai_playlist_response(result).model_dump(mode="json")
            await queue.put({"type": "result", "payload": payload})
        except Exception as e:
            logger.exception("[ai-playlist/stream] failed")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)  # end-of-stream sentinel

    task = asyncio.create_task(run())

    async def gen():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            task.cancel()  # no-op when finished; stops the pipeline on disconnect

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        # disable proxy buffering so events reach the browser as they happen
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/axis-playlist", response_model=AxisPlaylistResponse)
def axis_playlist(
    body: AxisPlaylistIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AxisPlaylistResponse:
    """Rank the collection against target z-values from the radar knobs."""
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    # Clamp targets to a sane z-range; unknown axis names are ignored by the
    # service (it iterates AXIS_NAMES), so no need to 422 on extras.
    targets = {k: max(-3.0, min(3.0, v)) for k, v in body.targets.items()}
    result = stream_service.axis_playlist(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        axis_targets=targets,
        limit=body.limit,
    )
    return AxisPlaylistResponse(
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
        diagnostics=result["diagnostics"],
    )


@router.get("/similar", response_model=SimilarTracksResponse)
def similar(
    request: Request,
    current_user: User = Depends(get_current_user),
    track_id: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    exclude_ids: str | None = Query(
        None, description="Comma-separated track ids to suppress. Capped at 100.",
    ),
) -> SimilarTracksResponse:
    """Tracks similar to a seed: CLAP neighbors re-ranked by sonic-axis closeness.

    Replaces the old frontend hack (text search for «artist title» in audio
    mode) and doubles as the ai-playlist agent's similar_tracks tool.
    """
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)
    excluded = [x.strip() for x in (exclude_ids or "").split(",") if x.strip()][:100]
    result = stream_service.similar_tracks(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        seed_track_id=track_id,
        limit=limit,
        exclude_ids=excluded,
    )
    return SimilarTracksResponse(
        seed_track_id=result["seed_track_id"],
        tracks=[_candidate_to_stream_track(c) for c in result["tracks"]],
    )


@router.get("/sonic-sibling")
def sonic_sibling_stub(
    track_id: str = Query(...),
    current_user: User = Depends(get_current_user),
):
    """Slot reserved for the Sonic Sibling endpoint (deferred to a future plan).

    See PLATFORM_DESIGN.md §5.1 and Plan 3 spec § Non-goals.
    """
    raise HTTPException(
        status_code=501,
        detail="Sonic Sibling is deferred — see PLATFORM_DESIGN.md §5.1 for status.",
    )
