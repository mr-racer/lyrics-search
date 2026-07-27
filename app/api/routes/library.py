"""Library endpoints."""

import asyncio
import hashlib
import heapq
import logging
import random
from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.domain.models import ArtistAggregate, IndexRequest, IndexProgress, AIEnabledRequest, LibraryAlbumsResponse, LikedSongsResponse, ListeningStatsResponse, RhythmResponse, WeeklyPulseResponse, EngagementResponse, TasteMapResponse, RediscoverResponse, User, ProducerResolveResponse, ProducerTracksResponse, SamplesResolveRequest, SamplesResolveResponse
from app.api.dependencies import get_current_user, require_mode
from app.api.helpers import derive_collection_for_user, member_index_root, path_within_root
from app.services.library_service import LibraryService
from app.services import track_credits_service
from app.services import uploads_service
from app.services._magic_sniff import sniff_audio_mime
from app.services.similarity_service import load_top_pairs, load_top_pairs_index, build_track_pairs

router = APIRouter(prefix="/library", tags=["Library"])


# ── Browse (payload-only search with relevance scoring) ────────────────────────

def _score_query(q_words: list[str], q_full: str, value: str) -> float:
    """Score a single field value against the query tokens."""
    if not value:
        return 0.0
    low = value.lower()
    s = 0.0
    for w in q_words:
        if w == low:
            s += 10
        elif low.startswith(w):
            s += 5
        elif w in low:
            s += 2
    # full-query bonus
    if q_full in low:
        s += 3
    return s


def _relevance(q_words: list, q_full: str, title: str, artist: str, album: str) -> float:
    """Weighted title/artist/album relevance for /library/browse. Shared by the
    SQLite fast-path and the Qdrant fallback so both rank identically."""
    score = (
        _score_query(q_words, q_full, title) * 3.0
        + _score_query(q_words, q_full, artist) * 2.5
        + _score_query(q_words, q_full, album) * 1.5
    )
    # Multi-word bonus: extra +1 per additional matching word (beyond first).
    match_count = sum(
        1 for w in q_words
        for v in (title, artist, album)
        if w in (v or "").lower()
    )
    if match_count > 1:
        score += (match_count - 1) * 1.0
    return score


# NOTE: most read endpoints below are sync `def` on purpose — their bodies are
# sync Qdrant/SQLite calls, and as `async def` they froze the event loop for
# every other request whenever Qdrant was busy with an indexing job. As plain
# `def` Starlette runs them in its threadpool (same pattern as /taste-map).
@router.get("/browse")
def browse_tracks(
    request: Request = None,
    current_user: User = Depends(get_current_user),
    q: Optional[str] = Query(None, min_length=2, description="Search query (title / artist / album). Omit to return all tracks."),
    limit: int = Query(6, ge=1, le=50, description="Max results"),
) -> list[dict]:
    """Payload-only search across title, artist, album with relevance scoring.

    When q is omitted, returns first N tracks without scoring (scroll mode).
    """
    if request is None:
        return []
    db_client = request.app.state.db_client
    if db_client is None:
        return []

    derived = derive_collection_for_user(current_user)

    has_query = q is not None and len(q.strip()) >= 2
    q_full = q.strip().lower() if has_query else ""
    q_words = list(set(q_full.split())) if has_query else []

    # ── Fast path: read from the SQLite mirror (no Qdrant scroll) ──
    try:
        from app.resources.metadata_db import MetadataDB
        if not has_query:
            rows = MetadataDB.get_browse_rows(derived, limit=limit)
            if rows:
                return rows
        else:
            rows = MetadataDB.get_browse_rows(derived)
            if rows:
                scored = []
                for r in rows:
                    score = _relevance(q_words, q_full, r["title"], r["artist"], r["album"])
                    if score > 0:
                        scored.append({**r, "score": round(score, 2)})
                scored.sort(key=lambda x: -x["score"])
                return scored[:limit]
    except Exception:
        logger.debug("[LibraryService] browse: SQLite lookup failed, falling back to Qdrant", exc_info=True)

    # Collection is ALWAYS the caller's derived account collection.
    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return []

    existing = {c.name for c in cols}
    target_col: str | None = derived if derived in existing else None

    if not target_col:
        return []

    # ── No query: simple scroll, return first N tracks ─────────
    if not has_query:
        result = []
        offset = None
        try:
            while len(result) < limit:
                batch_size = limit - len(result)
                results, next_offset = qdrant.scroll(
                    collection_name=target_col,
                    offset=offset,
                    limit=batch_size,
                    with_payload=["title", "artist", "album", "cover_art_path",
                                  "year", "genre", "duration"],
                    with_vectors=False,
                )
                for point in results:
                    pl = point.payload or {}
                    result.append({
                        "track_id": str(point.id),
                        "title": pl.get("title") or "",
                        "artist": pl.get("artist") or "",
                        "album": pl.get("album") or "",
                        "cover_art_path": pl.get("cover_art_path"),
                        "year": pl.get("year"),
                        "genre": pl.get("genre"),
                        "duration": pl.get("duration"),
                    })
                if next_offset is None or not results:
                    break
                offset = next_offset
        except Exception:
            logger.debug("[LibraryService] browse: scroll failed (partial results returned)")
        return result

    # ── With query: score all points, return top-K ──────────────
    top_k = []

    offset = None
    try:
        while True:
            results, next_offset = qdrant.scroll(
                collection_name=target_col,
                offset=offset,
                limit=2500,
                with_payload=["title", "artist", "album", "cover_art_path",
                              "year", "genre", "duration"],
                with_vectors=False,
            )
            for point in results:
                pl = point.payload or {}
                title = pl.get("title") or ""
                artist = pl.get("artist") or ""
                album = pl.get("album") or ""

                score = _relevance(q_words, q_full, title, artist, album)

                if score <= 0:
                    continue

                entry = (
                    -score,
                    len(top_k),
                    {
                        "track_id": str(point.id),
                        "title": title,
                        "artist": artist,
                        "album": album,
                        "cover_art_path": pl.get("cover_art_path"),
                        "year": pl.get("year"),
                        "genre": pl.get("genre"),
                        "duration": pl.get("duration"),
                        "score": round(score, 2),
                    },
                )
                if len(top_k) < limit:
                    heapq.heappush(top_k, entry)
                else:
                    heapq.heappushpop(top_k, entry)

            if next_offset is None or not results:
                break
            offset = next_offset
    except Exception:
        logger.debug("[LibraryService] browse: scroll failed (partial results returned)")

    # Extract and sort by score desc
    result = sorted(
        [item for (_, _, item) in top_k],
        key=lambda x: -x["score"],
    )
    return result


# ── Random tracks (landing page) ──────────────────────────────────────────────

@router.get("/random")
def get_random_tracks(
    request: Request = None,
    current_user: User = Depends(get_current_user),
    limit: int = Query(1, ge=1, le=20, description="Number of random tracks"),
    pool: int = Query(1000, ge=50, le=5000, description="Upper bound of sampling pool"),
) -> list[dict]:
    """Return ``limit`` random tracks from the collection.

    Implementation: scrolls up to ``pool`` point payloads, then samples
    ``limit`` of them client-side via ``random.sample``. For libraries
    larger than ``pool``, the sample is drawn from the first ``pool``
    scroll order — adequate randomness for landing-page UX, no extra cost.
    """
    if request is None:
        return []
    db_client = request.app.state.db_client
    if db_client is None:
        return []

    derived = derive_collection_for_user(current_user)

    # Collection is ALWAYS the caller's derived account collection.
    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return []

    existing = {c.name for c in cols}
    target_col: str | None = derived if derived in existing else None

    if not target_col:
        return []

    # Scroll up to `pool` points to form the sampling set
    points: list[dict] = []
    offset = None
    try:
        while len(points) < pool:
            batch_size = min(500, pool - len(points))
            results, next_offset = qdrant.scroll(
                collection_name=target_col,
                offset=offset,
                limit=batch_size,
                with_payload=["title", "artist", "album", "year", "genre", "duration_sec", "cover_art_path"],
                with_vectors=False,
            )
            for point in results:
                pl = point.payload or {}
                points.append({
                    "track_id": str(point.id),
                    "title": pl.get("title") or "",
                    "artist": pl.get("artist") or "",
                    "album": pl.get("album") or "",
                    "year": pl.get("year"),
                    "genre": pl.get("genre"),
                    "duration": pl.get("duration_sec"),
                    "cover_art_path": pl.get("cover_art_path"),
                })
            if next_offset is None or not results:
                break
            offset = next_offset
    except Exception:
        logger.debug("[LibraryService] random: scroll failed (partial pool returned)")

    if not points:
        return []

    k = min(limit, len(points))
    return random.sample(points, k)


# ── Collections info ──────────────────────────────────────────────────────────

@router.get("/collections")
def get_collections(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return all Qdrant collections with their point counts.

    Used by the frontend on startup to decide whether to show the
    onboarding screen (no data) or the main search UI.

    Also returns ``user_points`` — the count of tracks in the
    authenticated user's own collection — so the frontend can decide
    per-user whether to show onboarding instead of relying on the
    global total.

    Returns {"collections": [...], "total_points": N, "user_points": N, "qdrant_available": bool}
    """
    db_client = request.app.state.db_client

    # Qdrant was unavailable at startup
    if db_client is None:
        return {"collections": [], "total_points": 0, "user_points": 0, "qdrant_available": False}

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return {"collections": [], "total_points": 0, "user_points": 0, "qdrant_available": False}

    # Enrich each collection with the text model it was indexed with (if any).
    # Legacy collections indexed before this column exists will report None;
    # the frontend may then prompt re-indexing or fall back gracefully.
    from app.resources.metadata_db import MetadataDB
    try:
        MetadataDB.init()
    except Exception:
        pass

    result = []
    user_points = 0
    derived = derive_collection_for_user(current_user)

    for col in cols:
        try:
            info = qdrant.get_collection(col.name)
            count = info.points_count or 0
        except Exception:
            count = 0
        text_model = None
        try:
            text_model = MetadataDB.get_collection_text_model(col.name)
        except Exception:
            pass
        result.append({"name": col.name, "count": count, "text_model": text_model})

        if col.name == derived:
            user_points = count

    return {
        "collections": result,
        "total_points": sum(c["count"] for c in result),
        "user_points": user_points,
        "qdrant_available": True,
    }


@router.get("/settings")
def get_collection_settings(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return persisted per-collection settings for the caller's account.

    Returns {"collection_name": str, "text_model": str|null, "indexed_at": ts|null, "ai_enabled": bool}.
    ai_enabled defaults to True when no row exists (legacy collections stay AI-on).
    The collection is derived from the JWT — clients no longer pass a name.
    """
    from app.resources.metadata_db import MetadataDB
    derived = derive_collection_for_user(current_user)
    try:
        MetadataDB.init()
        settings = MetadataDB.get_collection_settings(derived)
        ai_enabled = MetadataDB.get_collection_ai_enabled(derived)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read settings: {e}")

    base = {"collection_name": derived, "text_model": None, "indexed_at": None}
    if settings is not None:
        base.update(settings)
    base["ai_enabled"] = ai_enabled
    return base


@router.get("/collections/{collection_name}/settings", deprecated=True)
async def get_collection_settings_legacy(collection_name: str) -> dict:
    """Deprecated alias — collection is now inferred from the JWT."""
    raise HTTPException(
        status_code=410,
        detail="Use GET /library/settings — collection inferred from JWT",
    )


@router.patch("/ai-enabled")
def set_collection_ai_enabled(
    req: AIEnabledRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Toggle the per-collection AI features opt-in. Used by Settings panel.

    The collection is derived from the JWT — clients no longer pass a name.
    """
    from app.resources.metadata_db import MetadataDB
    derived = derive_collection_for_user(current_user)
    try:
        MetadataDB.init()
        MetadataDB.set_collection_ai_enabled(derived, req.enabled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set ai_enabled: {e}")
    return {"collection_name": derived, "ai_enabled": req.enabled}


@router.patch("/collections/{collection_name}/ai-enabled", deprecated=True)
async def set_collection_ai_enabled_legacy(collection_name: str, req: AIEnabledRequest) -> dict:
    """Deprecated alias — collection is now inferred from the JWT."""
    raise HTTPException(
        status_code=410,
        detail="Use PATCH /library/ai-enabled — collection inferred from JWT",
    )


# ── Albums overview ───────────────────────────────────────────────────────────

@router.get("/albums", response_model=LibraryAlbumsResponse)
def get_library_albums(
    request: Request,
    current_user: User = Depends(get_current_user),
    sort: str = Query("alphabetical", description="alphabetical | year_desc | year_asc | track_count_desc"),
    label: Optional[str] = Query(None, description="Only albums where ≥1 track carries this record label"),
) -> LibraryAlbumsResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return LibraryAlbumsResponse(
            albums=[], collection_name=derived, qdrant_available=False,
        )
    return LibraryService.get_albums(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        sort=sort,
        label=label,
    )


# ── Producer credits (player drawer) ──────────────────────────────────────────

@router.get("/producers/resolve", response_model=ProducerResolveResponse)
def resolve_track_producers(
    track_id: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
) -> ProducerResolveResponse:
    """Resolve the track's producer names against the user's library.

    SQLite-only (works with Qdrant down). For each name: the canonical artist
    slug when the library also has them as an artist, and how many OTHER
    tracks credit them as producer. Unknown track / no producers → empty list.
    """
    derived = derive_collection_for_user(current_user)
    return ProducerResolveResponse(
        producers=track_credits_service.resolve_producers(derived, track_id),
        collection_name=derived,
    )


@router.get("/producers/tracks", response_model=ProducerTracksResponse)
def get_producer_tracks(
    name: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
) -> ProducerTracksResponse:
    """Tracks in the user's library produced by ``name`` (exact name match,
    case/whitespace-insensitive). SQLite-only."""
    derived = derive_collection_for_user(current_user)
    return ProducerTracksResponse(
        producer=name,
        tracks=track_credits_service.tracks_produced_by(derived, name),
        collection_name=derived,
    )


# ── Sample references (samples popover) ───────────────────────────────────────

@router.post("/samples/resolve", response_model=SamplesResolveResponse)
def resolve_sample_references(
    req: SamplesResolveRequest,
    current_user: User = Depends(get_current_user),
) -> SamplesResolveResponse:
    """Match «Artist — Title» sample strings against the user's library.

    POST because the strings are free text from track relations, not ids.
    Exact-only matching (see ``resolve_sample_refs``) — the player upgrades
    matched rows to playable ones. SQLite-only.
    """
    derived = derive_collection_for_user(current_user)
    return SamplesResolveResponse(
        matches=track_credits_service.resolve_sample_refs(derived, req.items),
        collection_name=derived,
    )


# ── Liked songs ───────────────────────────────────────────────────────────────

@router.get("/liked-songs", response_model=LikedSongsResponse)
def get_library_liked_songs(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> LikedSongsResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return LikedSongsResponse(tracks=[], collection_name=derived)
    return LibraryService.get_liked_songs(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
    )


# ── Rediscover ────────────────────────────────────────────────────────────────

@router.get("/rediscover", response_model=RediscoverResponse)
def get_library_rediscover(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> RediscoverResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return RediscoverResponse(collection_name=derived)
    return LibraryService.get_rediscover(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
    )


# ── Featured artist (deterministic daily rotation) ───────────────────────────

@router.get("/featured-artist", response_model=ArtistAggregate)
def get_library_featured_artist(
    request: Request,
    current_user: User = Depends(get_current_user),
    date: str | None = Query(None, description="YYYY-MM-DD; defaults to today"),
    lang: str = Query("en"),
) -> ArtistAggregate:
    from app.api.routes.artists import build_artist_aggregate
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    artists = LibraryService.list_distinct_artist_slugs(
        qdrant_client=db_client.qdrant, collection_name=derived,
    )
    if not artists:
        raise HTTPException(status_code=404, detail="no artists in collection")
    day = date or _date.today().isoformat()
    idx = int(hashlib.sha1(day.encode()).hexdigest(), 16) % len(artists)
    slug, _name = artists[idx]
    return build_artist_aggregate(db_client, derived, slug, lang)


# ── Listening stats ───────────────────────────────────────────────────────────

@router.get("/listening-stats", response_model=ListeningStatsResponse)
def get_library_listening_stats(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", pattern="^(en|ru)$"),
    tz_offset_minutes: int = Query(
        0, ge=-840, le=840,
        description="Client UTC offset in minutes (e.g. UTC+3 → 180) for local peak-hour bucketing",
    ),
) -> ListeningStatsResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return ListeningStatsResponse()
    return LibraryService.get_listening_stats(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        lang=lang,
        tz_offset_minutes=tz_offset_minutes,
    )


@router.get("/rhythm", response_model=RhythmResponse)
def get_library_rhythm(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", pattern="^(en|ru)$"),
    tz_offset_minutes: int = Query(
        0, ge=-840, le=840,
        description="Client UTC offset in minutes (e.g. UTC+3 → 180) for local-day/hour bucketing",
    ),
) -> RhythmResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return RhythmResponse()
    return LibraryService.get_rhythm(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        lang=lang,
        tz_offset_minutes=tz_offset_minutes,
    )


@router.get("/weekly-pulse", response_model=WeeklyPulseResponse)
def get_library_weekly_pulse(
    request: Request,
    current_user: User = Depends(get_current_user),
    tz_offset_minutes: int = Query(
        0, ge=-840, le=840,
        description="Client UTC offset in minutes (e.g. UTC+3 → 180) for local-day bucketing",
    ),
) -> WeeklyPulseResponse:
    """Light "this week" summary for the home page's under-the-fold widget —
    total listening time and top genre over the last 7 local days."""
    derived = derive_collection_for_user(current_user)
    return LibraryService.get_weekly_pulse(
        collection_name=derived, tz_offset_minutes=tz_offset_minutes,
    )


class GemTrackRef(BaseModel):
    track_id: str
    quote: str = ""


class GemTracksResponse(BaseModel):
    kind: str
    canonical: str
    tracks: list[GemTrackRef]
    # Namedrop only: the library artist the chip refers to. The popover shows
    # a link to their page here instead of the chip navigating there directly
    # — the other tracks that mention them are the point of tapping it.
    artist_slug: str | None = None
    artist_name: str | None = None
    artist_image: str | None = None


@router.get("/gems/tracks", response_model=GemTracksResponse)
def get_gem_tracks(
    kind: str = Query(..., description="capsule | namedrop | popculture | songref"),
    canonical: str = Query(..., description="Canonical entity name as stored"),
    current_user: User = Depends(get_current_user),
) -> GemTracksResponse:
    """All tracks of the library carrying a given lyric-gem entity — the
    mini-playlist behind a tapped gem chip. Track metadata is hydrated by the
    client via the existing GET /metadata/tracks?ids=... batch endpoint."""
    import json as _json

    from app.resources.metadata_db import MetadataDB

    derived = derive_collection_for_user(current_user)
    rows = MetadataDB.get_gem_tracks(derived, kind, canonical)

    artist_slug = None
    if kind == "namedrop":
        for r in rows:
            try:
                artist_slug = (_json.loads(r.get("detail") or "{}") or {}).get("artist_slug")
            except (TypeError, ValueError):
                artist_slug = None
            if artist_slug:
                break

    artist_image = None
    if artist_slug:
        try:
            artist_image = (MetadataDB.get_artist_audiodb(artist_slug, derived) or {}).get(
                "thumb_path",
            )
        except Exception:
            artist_image = None

    return GemTracksResponse(
        kind=kind,
        canonical=canonical,
        tracks=[GemTrackRef(track_id=r["track_id"], quote=r["quote"] or "") for r in rows],
        artist_slug=artist_slug,
        artist_name=canonical if artist_slug else None,
        artist_image=artist_image,
    )


@router.get("/engagement", response_model=EngagementResponse)
def get_library_engagement(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", pattern="^(en|ru)$"),
) -> EngagementResponse:
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return EngagementResponse()
    return LibraryService.get_engagement(
        qdrant_client=db_client.qdrant,
        collection_name=derived,
        lang=lang,
    )


# Sync def (not async): PCA + k-means is CPU-bound, so let Starlette run it in a
# threadpool instead of blocking the event loop. Result is cached per collection.
@router.get("/taste-map", response_model=TasteMapResponse)
def get_library_taste_map(
    request: Request,
    current_user: User = Depends(get_current_user),
    lang: str = Query("en", pattern="^(en|ru)$"),
) -> TasteMapResponse:
    from app.services import taste_map_service
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return TasteMapResponse()
    return taste_map_service.build(
        qdrant_client=db_client.qdrant, collection_name=derived, lang=lang,
    )


# ── Library statistics ────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Library statistics: total tracks, top genres (from Qdrant payload).

    Stats are always for the caller's derived account collection.

    Returns:
        {
          "total_tracks": int,
          "collection_name": str | None,
          "genres": [{"genre": str, "count": int, "pct": int}, ...],  # top-3
          "qdrant_available": bool,
        }
    """
    db_client = request.app.state.db_client
    derived = derive_collection_for_user(current_user)
    empty_payload = {
        "total_tracks": 0,
        "collection_name": None,
        "unique_genres": 0,
        "unique_artists": 0,
        "unique_albums": 0,
        "genres": [],
        "duration_buckets": [],
        "top_artists": [],
        "year_range": None,
        "decades": [],
        "formats": [],
        "lossless_pct": 0,
    }

    if db_client is None:
        return {**empty_payload, "qdrant_available": False}

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return {**empty_payload, "qdrant_available": False}

    if not cols:
        return {**empty_payload, "qdrant_available": True}

    # Stats always target the caller's derived account collection.
    existing = {c.name for c in cols}

    target_col: str | None = None
    target_count: int = 0

    if derived in existing:
        try:
            info = qdrant.get_collection(derived)
            target_col = derived
            target_count = info.points_count or 0
        except Exception:
            pass

    if not target_col:
        return {**empty_payload, "collection_name": derived, "qdrant_available": True}

    # Scroll through ALL points (payload only, no vectors — fast even on large collections)
    genre_counter: Counter = Counter()
    duration_counter: Counter = Counter()
    artist_counter: Counter = Counter()
    year_counter: Counter = Counter()
    # Distinct albums, keyed case-insensitively on the title — mirrors how
    # /library/albums groups, so the landing's "N albums" matches the library.
    album_keys: set[str] = set()
    format_counter: Counter = Counter()
    artist_slug_map: dict[str, str] = {}
    fmt_labels = {
        "flac": "FLAC", "mp3": "MP3", "m4a": "M4A", "aac": "AAC", "alac": "ALAC",
        "wav": "WAV", "aiff": "AIFF", "aif": "AIFF", "ogg": "OGG", "oga": "OGG",
        "opus": "OPUS", "wma": "WMA", "ape": "APE",
    }
    lossless_fmts = {"FLAC", "WAV", "AIFF", "ALAC", "APE"}

    # Shared lyrics-free light scroll, cached per collection (same scan the
    # albums grid / home page warm) — avoids a second whole-library re-scroll
    # on every library entry.
    from app.resources.qdrant_utils import light_points
    try:
        for _tid, pl in light_points(qdrant, target_col):
            genre = pl.get("genre")
            if genre and str(genre).strip():
                genre_counter[str(genre).strip()] += 1
            # Prefer the new bucket field `duration_range`; fall back to the
            # legacy clobbered `duration` only if it's still a range string
            # (pre-fix indexes). Numeric `duration` is the real seconds and
            # has no place in a bucket histogram.
            bucket = pl.get("duration_range")
            if bucket is None:
                raw = pl.get("duration")
                if isinstance(raw, str) and "-" in raw:
                    bucket = raw
            if bucket and str(bucket).strip():
                duration_counter[str(bucket).strip()] += 1
            artist = pl.get("artist")
            if artist and str(artist).strip():
                a = str(artist).strip()
                artist_counter[a] += 1
                if a not in artist_slug_map:
                    sl = pl.get("primary_artist_slug")
                    if sl:
                        artist_slug_map[a] = str(sl)
            album = pl.get("album")
            if album and str(album).strip():
                album_keys.add(str(album).strip().lower())
            fp = str(pl.get("file_path") or "")
            base = fp.replace("\\", "/").rsplit("/", 1)[-1]
            ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
            fmt = fmt_labels.get(ext)
            if fmt:
                format_counter[fmt] += 1
            year = pl.get("year")
            try:
                yi = int(year) if year is not None else None
            except (TypeError, ValueError):
                yi = None
            if yi and 1900 <= yi <= 2100:
                year_counter[yi] += 1
    except Exception:
        logger.debug("[LibraryService] stats: scroll failed (partial stats returned)")

    total_sampled = sum(genre_counter.values()) or 1
    unique_genre_count = len(genre_counter)
    top_genres = [
        {"genre": g, "count": c, "pct": round(c / total_sampled * 100)}
        for g, c in genre_counter.most_common(5)
    ]

    total_dur = sum(duration_counter.values()) or 1
    top_durations = [
        {"range": r, "count": c, "pct": round(c / total_dur * 100)}
        for r, c in duration_counter.most_common(6)
    ]

    total_artists = sum(artist_counter.values()) or 1
    unique_artist_count = len(artist_counter)
    # Reuse each artist's cached AudioDB photo (thumb/cutout) as the avatar —
    # a cheap SQLite read per top artist, no network. None when never fetched.
    def _artist_image(slug: str | None) -> str | None:
        if not slug:
            return None
        from app.resources.metadata_db import MetadataDB
        try:
            ad = MetadataDB.get_artist_audiodb(slug, target_col) or {}
        except Exception:
            return None
        return ad.get("thumb_path") or ad.get("cutout_path")

    top_artists = [
        {"artist": a, "count": c, "pct": round(c / total_artists * 100),
         "slug": artist_slug_map.get(a),
         "image": _artist_image(artist_slug_map.get(a))}
        for a, c in artist_counter.most_common(5)
    ]

    total_fmt = sum(format_counter.values()) or 1
    formats = [
        {"format": f, "count": c, "pct": round(c / total_fmt * 100)}
        for f, c in format_counter.most_common(6)
    ]
    lossless_pct = (
        round(sum(c for f, c in format_counter.items() if f in lossless_fmts)
              / total_fmt * 100)
        if format_counter else 0
    )

    year_range = None
    decades: list[dict] = []
    if year_counter:
        years_seen = list(year_counter.elements())
        year_range = {"min": min(years_seen), "max": max(years_seen)}
        decade_counter: Counter = Counter()
        for y, c in year_counter.items():
            decade_counter[(y // 10) * 10] += c
        total_years = sum(decade_counter.values()) or 1
        decades = [
            {"decade": d, "count": c, "pct": round(c / total_years * 100)}
            for d, c in sorted(decade_counter.items())
        ]

    return {
        "total_tracks": target_count,
        "collection_name": target_col,
        "unique_genres": unique_genre_count,
        "unique_artists": unique_artist_count,
        "unique_albums": len(album_keys),
        "genres": top_genres,
        "duration_buckets": top_durations,
        "top_artists": top_artists,
        "year_range": year_range,
        "decades": decades,
        "formats": formats,
        "lossless_pct": lossless_pct,
        "qdrant_available": True,
    }


# ── Top similar/dissimilar pairs ──────────────────────────────────────────────

@router.get("/top-pairs")
def get_top_pairs(
    request: Request,
    current_user: User = Depends(get_current_user),
    sample: int | None = Query(
        None, ge=1, le=200,
        description="If set, return only a random sample of this many "
                    "`dissimilar` entries (top-level fields only, `similar` "
                    "omitted). The home «Something new» card needs a handful, "
                    "not the whole matrix — this keeps the response in KB, not MB.",
    ),
) -> dict:
    """Get cached top-similar and top-dissimilar track pairs.

    Returns cached data if available, otherwise empty lists. With ``sample`` the
    response is trimmed to a small random ``dissimilar`` slice (the only thing
    the home discovery card consumes); without it the full cached matrix is
    returned for backward compatibility.

    Returns:
        {
          "similar": [...],
          "dissimilar": [...],
          "collection_name": str | None,
          "computed_at": float | None,
        }
    """
    derived = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None:
        return {
            "similar": [],
            "dissimilar": [],
            "collection_name": None,
            "computed_at": None,
            "qdrant_available": False,
        }

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return {
            "similar": [],
            "dissimilar": [],
            "collection_name": None,
            "computed_at": None,
            "qdrant_available": False,
        }

    if not cols:
        return {
            "similar": [],
            "dissimilar": [],
            "collection_name": None,
            "computed_at": None,
            "qdrant_available": True,
        }

    # Top pairs always target the caller's derived account collection.
    existing = {c.name for c in cols}
    target_col: str | None = derived if derived in existing else None

    if not target_col:
        return {
            "similar": [],
            "dissimilar": [],
            "collection_name": None,
            "computed_at": None,
            "qdrant_available": True,
        }

    # Load cached pairs
    cached = load_top_pairs(target_col)
    if cached:
        similar = cached.get("similar", [])
        dissimilar = cached.get("dissimilar", [])
        if sample is not None:
            # The home «Something new» card uses only a few distinct `dissimilar`
            # tracks (top-level fields), never `similar` nor the nested neighbor
            # arrays. Ship a small random sample instead of the whole matrix —
            # 10 MB → a few KB, with fresh variety on each load.
            pool = dissimilar
            if len(pool) > sample:
                pool = random.sample(pool, sample)
            dissimilar = [
                {
                    "song": e.get("song"),
                    "track_id": e.get("track_id"),
                    "cover_art_path": e.get("cover_art_path"),
                }
                for e in pool
            ]
            similar = []
        return {
            "similar": similar,
            "dissimilar": dissimilar,
            "collection_name": cached.get("collection_name"),
            "computed_at": cached.get("computed_at"),
            "qdrant_available": True,
        }

    # No cache yet
    return {
        "similar": [],
        "dissimilar": [],
        "collection_name": target_col,
        "computed_at": None,
        "qdrant_available": True,
    }


@router.get("/top-pairs/{track_id}")
def get_top_pairs_for_track(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Top-3 sonically similar + top-3 contrasting neighbours for ONE track.

    Reads the memoized CLAP top-pairs cache (a few KB out, never the full
    matrix) and enriches the neighbours with live payload via a single Qdrant
    retrieve. ``available`` is False when Qdrant is down or the cache has no
    entry for this track (e.g. the similarity analysis has not been run, or the
    track was indexed after it). The player hides the block on ``available:false``.
    """
    empty = {"available": False, "similar": [], "dissimilar": []}
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return empty

    derived = derive_collection_for_user(current_user)
    index = load_top_pairs_index(derived)
    if not index:
        return empty
    entry = index.get(track_id)
    if not entry:
        return empty

    sim = entry.get("similar", [])
    diss = entry.get("dissimilar", [])
    neighbor_ids = list({
        n.get("track_id") for n in (sim + diss) if n.get("track_id")
    })
    retrieve_ids = list({*neighbor_ids, track_id})

    payload_by_id: dict = {}
    if retrieve_ids:
        try:
            points = db_client.qdrant.retrieve(
                collection_name=derived,
                ids=retrieve_ids,
                with_payload=True,
                with_vectors=False,
            )
            payload_by_id = {str(p.id): (p.payload or {}) for p in points}
        except Exception:
            payload_by_id = {}

    result = build_track_pairs(
        sim, diss, payload_by_id, top_k=3,
        seed_album=payload_by_id.get(track_id, {}).get("album"),
        drop_same_album_similar=True, min_duration=60.0,
    )
    return {
        "available": True,
        "similar": result["similar"],
        "dissimilar": result["dissimilar"],
    }


# ── Native folder picker (server-side, returns real FS path) ─────────────────

@router.get("/pick-folder")
async def pick_folder() -> dict:
    """Open a native OS folder-picker dialog on the server machine.

    Returns the absolute path chosen by the user, or an empty string if
    the dialog was cancelled.  Uses tkinter which is bundled with CPython.
    """
    def _open_dialog() -> str:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()                          # hide the root window
        root.wm_attributes("-topmost", True)     # bring dialog to front
        path = filedialog.askdirectory(
            title="Выбери папку с музыкой",
            mustexist=True,
        )
        root.destroy()
        return path or ""

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _open_dialog)
    return {"path": path}


# ── Index folder ──────────────────────────────────────────────────────────────

@router.post("/index")
async def index_folder(
    req: IndexRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Index a folder of music files that already live on this host.

    Sharing mode: any authenticated user (single-tenant instance). Server
    mode: OWNER ONLY — the owner runs the host machine, so the setup wizard
    and settings can index a local folder directly; members must not be able
    to point the indexer (and the streamer) at arbitrary host paths, they
    upload via POST /library/upload instead.

    Returns {"status": "completed", "count": N, "message": "..."}
    """
    # Authorization before service availability: a member must see 403 even
    # while Qdrant is down (don't leak service state to unauthorized callers).
    from app.resources.metadata_db import MetadataDB
    cfg = MetadataDB.get_instance_config()
    if cfg is not None and cfg.get("mode") == "server" and current_user.role != "owner":
        # Members are owner-only for folder indexing UNLESS the operator opted in
        # by setting MEMBER_INDEX_ROOT (a trusted, usually read-only mount). Then a
        # member may index that root or any path beneath it — but nothing else, so
        # they still can't point the indexer/streamer at arbitrary host files.
        root = member_index_root()
        if not root or not path_within_root(req.folder_path, root):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"members may only index {root} in server mode"
                    if root else
                    "folder indexing is owner-only in server mode — upload files instead"
                ),
            )

    # Service availability next — if Qdrant/the whole library service is down,
    # 503 takes precedence over a bad folder (and preserves the pre-Phase-C
    # /index contract).
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable — is Qdrant running?")

    derived = derive_collection_for_user(current_user)

    # Sanity check: folder must exist on the host. Done BEFORE delegating so a
    # typo gives a clean 400 instead of a confusing mid-pipeline failure.
    if not Path(req.folder_path).is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"folder does not exist on this host: {req.folder_path}",
        )
    # In server mode the embedding model is an instance-wide admin decision
    # (same policy as /library/upload/batch-commit): the request body cannot
    # override it. Without this a member's mounted-folder index sent no
    # text_model (the member UI omits it), so the pipeline silently fell back to
    # the jina default instead of honoring the Settings-chosen model. In sharing
    # mode the single owner picks the model per-index, so keep the body value.
    text_model = req.text_model
    if cfg is not None and cfg.get("mode") == "server":
        from app.services.settings_service import settings_service
        text_model = settings_service.embed_model()

    # Phase B: key the in-flight indexing slot by the authenticated account, so
    # two accounts can index concurrently while one account can't double-start.
    result = await service.index_folder(
        folder_path=req.folder_path,
        collection_name=derived,
        better_lyrics_quality=req.better_lyrics_quality,
        text_model=text_model,
        enhance_by_musicbrainz=req.enhance_by_musicbrainz,
        account_id=current_user.id,
    )
    return result


# ── Server-mode uploads (Phase C) ──────────────────────────────────────────────

@router.post("/upload", dependencies=[Depends(require_mode("server"))])
def upload_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Stream-upload one audio file into the caller's managed library.

    Server mode only. Pipeline:
      1. Stream the multipart body into ``media/_quarantine/<id>.tmp`` computing
         SHA-256 incrementally; reject if size > MAX_UPLOAD_BYTES.
      2. Sniff the first 64 KB via libmagic; reject non-audio (falling back to the
         extension whitelist only when the sniff is inconclusive / libmagic absent).
      3. Idempotency: if (account_id, sha256) is already 'done', drop the
         quarantine copy and return the existing track_id.
      4. Atomic rename into ``media/<account_id>/audio/<sha>.<ext>``.
      5. Insert a pending_uploads row with status='uploaded'.

    Returns: {upload_id, sha256, size, status}.
    """
    import uuid as _uuid
    from pathlib import Path as _Path
    from app.resources.metadata_db import MetadataDB

    if file.filename is None:
        raise HTTPException(status_code=400, detail="missing filename")

    account_id = current_user.id
    media_root = uploads_service.media_root_default()
    quarantine_root = uploads_service.quarantine_root_default()
    quarantine_root.mkdir(parents=True, exist_ok=True)

    q_path = quarantine_root / f"{_uuid.uuid4().hex}.tmp"

    try:
        sha256, total, head = uploads_service.write_to_quarantine(
            file.file, q_path, max_bytes=uploads_service.MAX_UPLOAD_BYTES,
        )
    except uploads_service.UploadOversize as e:
        raise HTTPException(status_code=400, detail=f"upload too large: {e}")
    except Exception as e:
        logger.exception("[upload] quarantine write failed")
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    # MIME gate: a definite non-audio sniff rejects even if the filename claims
    # an audio extension (stops a .flac-named shell script). Only when libmagic
    # is unavailable (octet-stream) do we trust the extension whitelist.
    mime = sniff_audio_mime(head)
    ext_ok = _Path(file.filename).suffix.lower() in uploads_service._AUDIO_EXTENSIONS
    accept = ext_ok if mime == "application/octet-stream" else mime.startswith("audio/")
    if not accept:
        q_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"file is not audio (mime={mime})")

    # Idempotency: same SHA already indexed for this account → return existing.
    MetadataDB.init()
    existing = MetadataDB.find_done_upload_by_sha(account_id=account_id, sha256=sha256)
    if existing:
        q_path.unlink(missing_ok=True)
        return {
            "upload_id": existing["upload_id"],
            "sha256": sha256,
            "size": existing["size_bytes"],
            "status": "done",
            "track_id": existing["track_id"],
        }

    ext = uploads_service.choose_extension(mime, original=file.filename)
    final_path = uploads_service.atomic_promote_to_managed(
        quarantine_path=q_path,
        media_root=media_root,
        account_id=account_id,
        sha256=sha256,
        extension=ext,
    )

    upload_id = MetadataDB.create_pending_upload(
        account_id=account_id,
        sha256=sha256,
        original_filename=file.filename,
        size_bytes=total,
        storage_path=str(final_path),
    )

    return {
        "upload_id": upload_id,
        "sha256": sha256,
        "size": total,
        "status": "uploaded",
    }


@router.get("/upload/{upload_id}", dependencies=[Depends(require_mode("server"))])
def get_upload_status(
    upload_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Look up an upload's status. Owner-aware: 404 if the row belongs to another account."""
    from app.resources.metadata_db import MetadataDB
    MetadataDB.init()
    row = MetadataDB.get_pending_upload(upload_id)
    if not row or row["account_id"] != current_user.id:
        # Deliberately collapse "not yours" with "doesn't exist" — no leak.
        raise HTTPException(status_code=404, detail="not found")
    return {
        "upload_id": row["upload_id"],
        "sha256": row["sha256"],
        "size": row["size_bytes"],
        "status": row["status"],
        "track_id": row["track_id"],
        "error": row["error"],
    }


class _BatchCommitRequest(BaseModel):
    upload_ids: list[str]
    # IGNORED (kept for graceful back-compat): in server mode the embedding
    # model is the admin's instance setting (EMBED_MODEL), not a member choice.
    text_model: str | None = None
    # UI language → language of auto-generated AI enrichment (bios/facts/vibes).
    lang: str = "ru"


@router.post("/upload/batch-commit", dependencies=[Depends(require_mode("server"))])
async def batch_commit_uploads(
    req: _BatchCommitRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Enqueue indexing for a set of already-uploaded files (server mode)."""
    from app.resources.metadata_db import MetadataDB
    from app.services.settings_service import settings_service
    if not req.upload_ids:
        raise HTTPException(status_code=400, detail="upload_ids must not be empty")
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable")

    # Filter to the caller's OWN uploads to prevent cross-account triggering.
    MetadataDB.init()
    mine = [
        uid for uid in req.upload_ids
        if (row := MetadataDB.get_pending_upload(uid)) and row["account_id"] == current_user.id
    ]
    if not mine:
        raise HTTPException(
            status_code=400, detail="none of the upload_ids belong to the caller",
        )

    # Hard-lock the embedding model to the instance setting (admin decides).
    # None → backend default (jina-small), same as before when unset.
    job_id = service.enqueue_upload_indexing(
        account_id=current_user.id, upload_ids=mine,
        text_model=settings_service.embed_model(),
        lang=(req.lang or "ru").strip().lower(),
    )
    return {"job_id": job_id}


@router.delete("/tracks/{track_id}", dependencies=[Depends(require_mode("server"))])
def delete_track(
    track_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Remove a track from Qdrant and delete its file from ``media/<account>/``.

    Server mode only. Owner-aware: 404 if the track's storage_path is not under
    the caller's account namespace (prevents path-traversal abuse via a forged
    track_id, and refuses pre-Phase-C folder-scan tracks).
    """
    import os as _os
    from app.services.audio_streaming import drop_transcoded_for_tracks, drop_source_for_tracks

    collection_name = derive_collection_for_user(current_user)
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    qdrant = db_client.qdrant
    try:
        result = qdrant.retrieve(collection_name=collection_name, ids=[track_id], with_payload=True)
    except Exception:
        raise HTTPException(status_code=404, detail="track not found")
    if not result:
        raise HTTPException(status_code=404, detail="track not found")

    payload = result[0].payload or {}
    file_path = payload.get("file_path") or ""
    expected_prefix = str(uploads_service.media_root_default() / current_user.id) + _os.sep
    if not file_path.startswith(expected_prefix):
        # Cross-account or pre-Phase-C (folder-scan) track — refuse.
        raise HTTPException(status_code=404, detail="track not found")

    # Delete from Qdrant first; on failure we don't orphan the row by removing
    # the file too.
    try:
        qdrant.delete(collection_name=collection_name, points_selector=[track_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant delete failed: {e}")

    # Drop the SQLite mirror row + slug join so the albums grid / artist pages /
    # recommendations / stats (all SQLite-backed now) stop serving a ghost, then
    # invalidate the cached light index so the change surfaces immediately.
    try:
        from app.resources.metadata_db import MetadataDB
        from app.resources.qdrant_utils import invalidate_light_cache
        MetadataDB.delete_track_metadata(collection_name, track_id)
        invalidate_light_cache(collection_name)
    except Exception:
        logger.warning("[delete_track] SQLite mirror cleanup failed for %s", track_id, exc_info=True)

    # Delete the physical file (idempotent — missing is fine).
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[delete_track] file unlink failed for %s: %s", file_path, e)

    # Drop the per-account transcoded cache (Phase B §6.6 keyed by collection_name)
    # + the in-memory track-source cache for the stream hot path.
    try:
        drop_transcoded_for_tracks(account_id=collection_name, track_ids=[track_id])
        drop_source_for_tracks(collection_name, [track_id])
    except Exception:
        pass

    return {"ok": True, "track_id": track_id}


# ── Status / progress ─────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return current indexing status for the authenticated account."""
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable — is Qdrant running?")
    return await service.get_status(account_id=current_user.id)


@router.get("/progress/{job_id}")
async def get_progress(job_id: str) -> IndexProgress:
    """Get indexing progress (not yet implemented)."""
    raise HTTPException(status_code=501, detail="Job tracking not yet implemented")


# ── Delete collection ─────────────────────────────────────────────────────────

@router.delete("/collection/{collection_name}", deprecated=True)
async def delete_collection(collection_name: str, request: Request):
    """Removed in Phase D — self-serve collection delete is no longer exposed.

    Account owners wipe an account via the admin endpoint instead.
    """
    raise HTTPException(
        status_code=410,
        detail="Self-serve collection delete removed — owners use POST /admin/accounts/{user_id}/wipe",
    )


# ── Sonic Descriptor ──────────────────────────────────────────────────────────

@router.get("/sonic-descriptor/{track_slug}")
def get_sonic_descriptor(track_slug: str) -> dict:
    """Return tags + sonic_class + audio_signature for a track. 404 if track unknown."""
    from app.resources.metadata_db import MetadataDB
    MetadataDB.init()
    desc = MetadataDB.get_sonic_descriptor(track_slug)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"Track {track_slug} not found")
    return {"track_id": track_slug, **desc}


@router.get("/sonic-facets")
def get_sonic_facets(top_k: int = 50) -> dict:
    """Return aggregate counts of top-K sonic_tags across the library.

    Powers the SonicFiltersChips UI in SearchSectionV2 — the chip set is
    derived from real library data, not a hardcoded vocabulary."""
    from app.resources.metadata_db import MetadataDB
    MetadataDB.init()
    return MetadataDB.get_sonic_facets(top_k=top_k)


@router.get("/year-facets")
def get_year_facets(
    top_k: int = 30,
    collection_name: Optional[str] = Query(None, description="Collection to aggregate (omit for all)"),
    request: Request = None,
) -> dict:
    """Aggregate year_range values from Qdrant payload.

    If collection_name is provided, aggregates only that collection.
    Otherwise aggregates all available collections.
    Powers the DecadeFiltersChips UI in SearchSection.
    """
    if request is None or request.app.state.db_client is None:
        return {"year_ranges": []}

    # ── Fast path: SQLite mirror (decade buckets from the `year` column) ──
    try:
        from app.resources.metadata_db import MetadataDB
        MetadataDB.init()
        sqlite_counts = MetadataDB.get_year_facets_from_sqlite(collection_name)
        if sqlite_counts:
            counter = Counter(sqlite_counts)
            return {
                "year_ranges": [
                    {"value": v, "count": n} for v, n in counter.most_common(top_k)
                ],
            }
    except Exception:
        logger.debug("[year-facets] SQLite lookup failed, falling back to Qdrant", exc_info=True)

    qdrant = request.app.state.db_client.qdrant
    counter: Counter = Counter()

    try:
        cols = qdrant.get_collections().collections
    except Exception:
        return {"year_ranges": []}

    # If a specific collection is requested, filter to just that one.
    if collection_name:
        existing_names = {c.name for c in cols}
        cols = [c for c in cols if c.name == collection_name] if collection_name in existing_names else []

    for col in cols:
        offset = None
        while True:
            try:
                points, next_offset = qdrant.scroll(
                    collection_name=col.name,
                    limit=512,
                    with_payload=["year_range"],
                    with_vectors=False,
                    offset=offset,
                )
            except Exception:
                break
            for p in points:
                yr = (p.payload or {}).get("year_range")
                if yr:
                    counter[yr] += 1
            if next_offset is None or not points:
                break
            offset = next_offset

    return {
        "year_ranges": [
            {"value": v, "count": n} for v, n in counter.most_common(top_k)
        ],
    }


@router.get("/sonic-prompts")
async def get_sonic_prompts(request: Request) -> dict:
    """Return the current prompt vocabulary JSON."""
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    import json as _json
    if not svc.prompt_vocab_path.exists():
        from app.services.sonic_descriptor_service import DEFAULT_VOCAB
        return DEFAULT_VOCAB
    return _json.loads(svc.prompt_vocab_path.read_text(encoding="utf-8"))


@router.put("/sonic-prompts")
async def put_sonic_prompts(payload: dict, request: Request) -> dict:
    """Overwrite the prompt vocabulary. Invalidates cached embeddings — re-tagging required.

    The caller is responsible for triggering bulk re-tagging via a separate endpoint or
    by running a background job.
    """
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    if "groups" not in payload:
        raise HTTPException(status_code=400, detail="payload must contain 'groups' object")

    import json as _json
    svc.prompt_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    svc.prompt_vocab_path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Invalidate caches
    svc._prompts = None
    svc._prompt_embeddings = None
    if svc.embeddings_path.exists():
        svc.embeddings_path.unlink()

    n_prompts = sum(len(v) for v in payload.get("groups", {}).values())
    return {"ok": True, "n_prompts": n_prompts}
