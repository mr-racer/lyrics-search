"""Library endpoints."""

import asyncio
import heapq
import logging
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

from app.domain.models import IndexRequest, IndexProgress, AIEnabledRequest, LibraryAlbumsResponse, LikedSongsResponse
from app.services.library_service import LibraryService
from app.services.similarity_service import load_top_pairs

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


@router.get("/browse")
async def browse_tracks(
    q: Optional[str] = Query(None, min_length=2, description="Search query (title / artist / album). Omit to return all tracks."),
    limit: int = Query(6, ge=1, le=50, description="Max results"),
    collection_name: Optional[str] = Query(None, description="Collection to browse"),
    request: Request = None,
) -> list[dict]:
    """Payload-only search across title, artist, album with relevance scoring.

    When q is omitted, returns first N tracks without scoring (scroll mode).
    """
    if request is None:
        return []
    db_client = request.app.state.db_client
    if db_client is None:
        return []

    has_query = q is not None and len(q.strip()) >= 2
    q_full = q.strip().lower() if has_query else ""
    q_words = list(set(q_full.split())) if has_query else []

    # Resolve target collection (same logic as /stats)
    DEFAULT_COLLECTION = "music_explorer"
    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return []

    existing = {c.name for c in cols}
    pick = collection_name if collection_name else DEFAULT_COLLECTION
    target_col: str | None = pick if pick in existing else None

    if not target_col:
        # fall back to collection with most points
        best_count = 0
        for col in cols:
            try:
                info = qdrant.get_collection(col.name)
                cnt = info.points_count or 0
                if cnt > best_count:
                    best_count = cnt
                    target_col = col.name
            except Exception:
                pass

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
                    with_payload=["title", "artist", "album", "cover_art_path"],
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
                with_payload=["title", "artist", "album", "cover_art_path"],
                with_vectors=False,
            )
            for point in results:
                pl = point.payload or {}
                title = pl.get("title") or ""
                artist = pl.get("artist") or ""
                album = pl.get("album") or ""

                score = (
                    _score_query(q_words, q_full, title) * 3.0
                    + _score_query(q_words, q_full, artist) * 2.5
                    + _score_query(q_words, q_full, album) * 1.5
                )

                # Multi-word bonus: extra +1 per additional matching word (beyond first)
                match_count = sum(
                    1 for w in q_words
                    for v in (title, artist, album)
                    if w in (v or "").lower()
                )
                if match_count > 1:
                    score += (match_count - 1) * 1.0

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
async def get_random_tracks(
    collection_name: Optional[str] = Query(None, description="Collection to sample from"),
    limit: int = Query(1, ge=1, le=20, description="Number of random tracks"),
    pool: int = Query(1000, ge=50, le=5000, description="Upper bound of sampling pool"),
    request: Request = None,
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

    # Resolve target collection (same logic as /browse)
    DEFAULT_COLLECTION = "music_explorer"
    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return []

    existing = {c.name for c in cols}
    pick = collection_name if collection_name else DEFAULT_COLLECTION
    target_col: str | None = pick if pick in existing else None

    if not target_col:
        best_count = 0
        for col in cols:
            try:
                info = qdrant.get_collection(col.name)
                cnt = info.points_count or 0
                if cnt > best_count:
                    best_count = cnt
                    target_col = col.name
            except Exception:
                pass

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
async def get_collections(request: Request) -> dict:
    """Return all Qdrant collections with their point counts.

    Used by the frontend on startup to decide whether to show the
    onboarding screen (no data) or the main search UI.

    Returns {"collections": [...], "total_points": N, "qdrant_available": bool}
    """
    db_client = request.app.state.db_client

    # Qdrant was unavailable at startup
    if db_client is None:
        return {"collections": [], "total_points": 0, "qdrant_available": False}

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception as e:
        return {"collections": [], "total_points": 0, "qdrant_available": False}

    # Enrich each collection with the text model it was indexed with (if any).
    # Legacy collections indexed before this column existed will report None;
    # the frontend may then prompt re-indexing or fall back gracefully.
    from app.resources.metadata_db import MetadataDB
    try:
        MetadataDB.init()
    except Exception:
        pass

    result = []
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

    return {
        "collections": result,
        "total_points": sum(c["count"] for c in result),
        "qdrant_available": True,
    }


@router.get("/collections/{collection_name}/settings")
async def get_collection_settings(collection_name: str) -> dict:
    """Return persisted per-collection settings.

    Returns {"collection_name": str, "text_model": str|null, "indexed_at": ts|null, "ai_enabled": bool}.
    ai_enabled defaults to True when no row exists (legacy collections stay AI-on).
    """
    from app.resources.metadata_db import MetadataDB
    try:
        MetadataDB.init()
        settings = MetadataDB.get_collection_settings(collection_name)
        ai_enabled = MetadataDB.get_collection_ai_enabled(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read settings: {e}")

    base = {"collection_name": collection_name, "text_model": None, "indexed_at": None}
    if settings is not None:
        base.update(settings)
    base["ai_enabled"] = ai_enabled
    return base


@router.patch("/collections/{collection_name}/ai-enabled")
async def set_collection_ai_enabled(collection_name: str, req: AIEnabledRequest) -> dict:
    """Toggle the per-collection AI features opt-in. Used by Settings panel."""
    from app.resources.metadata_db import MetadataDB
    try:
        MetadataDB.init()
        MetadataDB.set_collection_ai_enabled(collection_name, req.enabled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set ai_enabled: {e}")
    return {"collection_name": collection_name, "ai_enabled": req.enabled}


# ── Albums overview ───────────────────────────────────────────────────────────

@router.get("/albums", response_model=LibraryAlbumsResponse)
async def get_library_albums(
    request: Request,
    collection_name: str = Query(..., description="Collection name (required)"),
    sort: str = Query("alphabetical", description="alphabetical | year_desc | year_asc | track_count_desc"),
) -> LibraryAlbumsResponse:
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return LibraryAlbumsResponse(
            albums=[], collection_name=collection_name, qdrant_available=False,
        )
    return LibraryService.get_albums(
        qdrant_client=db_client.qdrant,
        collection_name=collection_name,
        sort=sort,
    )


# ── Liked songs ───────────────────────────────────────────────────────────────

@router.get("/liked-songs", response_model=LikedSongsResponse)
async def get_library_liked_songs(
    request: Request,
    collection_name: str = Query(..., description="Collection name (required)"),
) -> LikedSongsResponse:
    db_client = request.app.state.db_client
    if db_client is None or db_client.qdrant is None:
        return LikedSongsResponse(tracks=[], collection_name=collection_name)
    return LibraryService.get_liked_songs(
        qdrant_client=db_client.qdrant,
        collection_name=collection_name,
    )


# ── Library statistics ────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats(
    request: Request,
    collection_name: Optional[str] = Query(None, description="Collection to get stats for"),
) -> dict:
    """Library statistics: total tracks, top genres (from Qdrant payload).

    If collection_name is provided, stats are for that specific collection.
    Otherwise, picks the collection with the most points.

    Returns:
        {
          "total_tracks": int,
          "collection_name": str | None,
          "genres": [{"genre": str, "count": int, "pct": int}, ...],  # top-3
          "qdrant_available": bool,
        }
    """
    db_client = request.app.state.db_client
    empty_payload = {
        "total_tracks": 0,
        "collection_name": None,
        "unique_genres": 0,
        "unique_artists": 0,
        "genres": [],
        "duration_buckets": [],
        "top_artists": [],
        "year_range": None,
        "decades": [],
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

    # Use the requested collection; fall back to the default collection name
    # (same default as DbClient / LyricsDB) so that stats and search always
    # target the same collection when the frontend does not specify one.
    DEFAULT_COLLECTION = "music_explorer"
    existing = {c.name for c in cols}

    target_col: str | None = None
    target_count: int = 0

    pick = collection_name if collection_name else DEFAULT_COLLECTION
    if pick in existing:
        try:
            info = qdrant.get_collection(pick)
            target_col = pick
            target_count = info.points_count or 0
        except Exception:
            pass

    # If the default doesn't exist, fall back to any collection with points
    if not target_col:
        for col in cols:
            try:
                info = qdrant.get_collection(col.name)
                cnt = info.points_count or 0
                if cnt > target_count:
                    target_count = cnt
                    target_col = col.name
            except Exception:
                pass

    if not target_col:
        return {**empty_payload, "collection_name": collection_name, "qdrant_available": True}

    # Scroll through ALL points (payload only, no vectors — fast even on large collections)
    genre_counter: Counter = Counter()
    duration_counter: Counter = Counter()
    artist_counter: Counter = Counter()
    year_counter: Counter = Counter()
    offset = None

    try:
        while True:
            results, next_offset = qdrant.scroll(
                collection_name=target_col,
                offset=offset,
                limit=250,
                with_payload=["genre", "duration", "artist", "year"],
                with_vectors=False,
            )
            for point in results:
                pl = point.payload or {}
                genre = pl.get("genre")
                if genre and str(genre).strip():
                    genre_counter[str(genre).strip()] += 1
                dur = pl.get("duration")
                if dur and str(dur).strip():
                    duration_counter[str(dur).strip()] += 1
                artist = pl.get("artist")
                if artist and str(artist).strip():
                    artist_counter[str(artist).strip()] += 1
                year = pl.get("year")
                try:
                    yi = int(year) if year is not None else None
                except (TypeError, ValueError):
                    yi = None
                if yi and 1900 <= yi <= 2100:
                    year_counter[yi] += 1
            if next_offset is None or not results:
                break
            offset = next_offset
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
    top_artists = [
        {"artist": a, "count": c, "pct": round(c / total_artists * 100)}
        for a, c in artist_counter.most_common(5)
    ]

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
        "genres": top_genres,
        "duration_buckets": top_durations,
        "top_artists": top_artists,
        "year_range": year_range,
        "decades": decades,
        "qdrant_available": True,
    }


# ── Top similar/dissimilar pairs ──────────────────────────────────────────────

@router.get("/top-pairs")
async def get_top_pairs(
    request: Request,
    collection_name: Optional[str] = Query(None, description="Collection to get top pairs for"),
) -> dict:
    """Get cached top-similar and top-dissimilar track pairs.

    Returns cached data if available, otherwise empty lists.

    Returns:
        {
          "similar": [...],
          "dissimilar": [...],
          "collection_name": str | None,
          "computed_at": float | None,
        }
    """
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

    # Resolve target collection (same logic as /stats)
    DEFAULT_COLLECTION = "music_explorer"
    existing = {c.name for c in cols}

    target_col: str | None = None
    pick = collection_name if collection_name else DEFAULT_COLLECTION
    if pick in existing:
        target_col = pick

    if not target_col:
        # Fall back to collection with most points
        target_count = 0
        for col in cols:
            try:
                info = qdrant.get_collection(col.name)
                cnt = info.points_count or 0
                if cnt > target_count:
                    target_count = cnt
                    target_col = col.name
            except Exception:
                pass

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
        return {
            "similar": cached.get("similar", []),
            "dissimilar": cached.get("dissimilar", []),
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
async def index_folder(req: IndexRequest, request: Request) -> dict:
    """Index a folder with music files.

    Returns {"status": "completed", "count": N, "message": "..."}
    """
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable — is Qdrant running?")
    result = await service.index_folder(
        folder_path=req.folder_path,
        collection_name=req.collection_name,
        better_lyrics_quality=req.better_lyrics_quality,
        text_model=req.text_model,
        enhance_by_musicbrainz=req.enhance_by_musicbrainz,
    )
    return result


# ── Status / progress ─────────────────────────────────────────────────────────

@router.get("/status")
async def get_status(request: Request) -> dict:
    """Return current indexing status."""
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable — is Qdrant running?")
    return await service.get_status()


@router.get("/progress/{job_id}")
async def get_progress(job_id: str) -> IndexProgress:
    """Get indexing progress (not yet implemented)."""
    raise HTTPException(status_code=501, detail="Job tracking not yet implemented")


# ── Delete collection ─────────────────────────────────────────────────────────

@router.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str, request: Request):
    """Delete a Qdrant collection by name and clean up related cache files."""
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")

    # Collect track ids BEFORE dropping the collection so we can purge their
    # transcoded audio cache after the Qdrant delete succeeds.
    track_ids: list[str] = []
    try:
        qdrant = db_client.qdrant
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=collection_name,
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            track_ids.extend(str(p.id) for p in points)
            if offset is None:
                break
    except Exception:
        # Collection might be missing or unreadable — non-fatal, just skip cache purge.
        track_ids = []

    try:
        qdrant.delete_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

    # Clean up cached top-pairs if exists
    try:
        cache_file = Path(__file__).parent.parent.parent / "cache" / "top_pairs" / f"{collection_name}.json"
        if cache_file.exists():
            cache_file.unlink()
    except Exception:
        pass

    # Drop transcoded ALAC→FLAC cache for the deleted tracks.
    if track_ids:
        try:
            from app.services.audio_streaming import drop_transcoded_for_tracks
            drop_transcoded_for_tracks(track_ids)
        except Exception:
            pass

    return {"deleted": True, "collection_name": collection_name}


# ── Sonic Descriptor ──────────────────────────────────────────────────────────

@router.get("/sonic-descriptor/{track_slug}")
async def get_sonic_descriptor(track_slug: str) -> dict:
    """Return tags + sonic_class + audio_signature for a track. 404 if track unknown."""
    from app.resources.metadata_db import MetadataDB
    MetadataDB.init()
    desc = MetadataDB.get_sonic_descriptor(track_slug)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"Track {track_slug} not found")
    return {"track_id": track_slug, **desc}


@router.get("/sonic-facets")
async def get_sonic_facets(top_k: int = 50) -> dict:
    """Return aggregate counts of top-K sonic_tags across the library.

    Powers the SonicFiltersChips UI in SearchSectionV2 — the chip set is
    derived from real library data, not a hardcoded vocabulary."""
    from app.resources.metadata_db import MetadataDB
    MetadataDB.init()
    return MetadataDB.get_sonic_facets(top_k=top_k)


@router.get("/year-facets")
async def get_year_facets(
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
        return {"version": 0, "groups": {}}
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


# ── Cluster discovery (HDBSCAN job) ───────────────────────────────────────────

_CLUSTER_JOBS: dict[str, dict] = {}
_APP_STATE: dict = {}


def register_app_state(app) -> None:
    """Called from main.py startup to expose services to module-level helpers."""
    _APP_STATE["sonic_descriptor_service"] = app.state.sonic_descriptor_service
    _APP_STATE["db_client"] = app.state.db_client


async def _run_cluster_discovery_job(collection: str) -> dict:
    """Run HDBSCAN clustering for ``collection`` in a thread (CPU-bound)."""
    import functools
    svc = _APP_STATE.get("sonic_descriptor_service")
    db = _APP_STATE.get("db_client")
    if svc is None or db is None:
        raise RuntimeError("services unavailable")
    loop = asyncio.get_running_loop()
    fn = functools.partial(svc.cluster_library, db.qdrant, collection)
    return await loop.run_in_executor(None, fn)


@router.post("/cluster-discovery", status_code=202)
async def post_cluster_discovery(
    request: Request,
    collection: str = Query(..., description="Collection to cluster"),
) -> dict:
    """Trigger HDBSCAN clustering. Returns immediately with a job_id."""
    job_id = uuid.uuid4().hex[:12]
    _CLUSTER_JOBS[job_id] = {"status": "running", "result": None, "error": None}

    async def _runner():
        try:
            from app.api.routes import library as _self_mod
            result = await _self_mod._run_cluster_discovery_job(collection)
            _CLUSTER_JOBS[job_id] = {"status": "completed", "result": result, "error": None}
        except Exception as e:
            logger.exception("[Cluster Discovery] job %s failed", job_id)
            _CLUSTER_JOBS[job_id] = {"status": "failed", "result": None, "error": str(e)}

    asyncio.create_task(_runner())
    return {"job_id": job_id, "status": "pending"}


@router.get("/cluster-jobs/{job_id}")
async def get_cluster_job(job_id: str) -> dict:
    """Return the current state of a clustering job."""
    job = _CLUSTER_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, **job}


@router.get("/clusters/representatives")
async def get_cluster_representatives(
    request: Request,
    collection: str = Query(..., description="Collection name"),
) -> list[dict]:
    """Return the cluster grid for the curator UI: id, size, representatives, current label."""
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    return svc.get_cluster_representatives(collection=collection)


@router.post("/clusters/labels")
async def post_cluster_labels(payload: dict, request: Request) -> dict:
    """Persist user-assigned cluster labels.

    Body: ``{"collection": str, "labels": {"<cluster_id_int>": "Label name", ...}}``.
    """
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    collection = payload.get("collection")
    raw_labels = payload.get("labels", {})
    if not collection or not isinstance(raw_labels, dict):
        raise HTTPException(status_code=400, detail="Body must include 'collection' and 'labels' dict")
    try:
        labels = {int(k): v for k, v in raw_labels.items()}
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Label keys must be integer cluster ids")
    svc.save_cluster_labels(collection=collection, labels=labels)
    return {"ok": True, "n_labels": len(labels)}


@router.get("/sonic-clusters")
async def get_sonic_clusters(
    request: Request,
    collection: str = Query(..., description="Collection name"),
) -> list[dict]:
    """Return labeled clusters for Sonic Map overlay. Empty list if curator not run yet."""
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    labels = svc.load_cluster_labels(collection=collection)
    assignments = svc.load_cluster_assignments(collection=collection)
    if not labels or not assignments:
        return []

    by_id: dict[int, list[str]] = {}
    for slug, cid in assignments.items():
        if cid == -1:
            continue
        by_id.setdefault(cid, []).append(slug)
    return [
        {
            "cluster_id": cid,
            "label": labels.get(cid, f"Cluster {cid}"),
            "size": len(slugs),
            "member_slugs": slugs,
        }
        for cid, slugs in by_id.items()
    ]


# ── Sonic classifier (MLP training job) ───────────────────────────────────────

_TRAINING_JOBS: dict[str, dict] = {}


async def _run_classifier_training_job(collection: str) -> dict:
    """Run MLP classifier training in a thread."""
    import functools
    svc = _APP_STATE.get("sonic_descriptor_service")
    db = _APP_STATE.get("db_client")
    if svc is None or db is None:
        raise RuntimeError("services unavailable")
    loop = asyncio.get_running_loop()
    fn = functools.partial(svc.train_classifier, db.qdrant, collection)
    return await loop.run_in_executor(None, fn)


@router.post("/sonic-classifier/train", status_code=202)
async def post_train_classifier(
    request: Request,
    collection: str = Query(..., description="Collection to train classifier on"),
) -> dict:
    """Train MLP classifier on labeled clusters. Returns job_id immediately."""
    job_id = uuid.uuid4().hex[:12]
    _TRAINING_JOBS[job_id] = {"status": "running", "result": None, "error": None}

    async def _runner():
        try:
            from app.api.routes import library as _self_mod
            result = await _self_mod._run_classifier_training_job(collection)
            _TRAINING_JOBS[job_id] = {"status": "completed", "result": result, "error": None}
        except Exception as e:
            logger.exception("[Classifier Training] job %s failed", job_id)
            _TRAINING_JOBS[job_id] = {"status": "failed", "result": None, "error": str(e)}

    asyncio.create_task(_runner())
    return {"job_id": job_id, "status": "pending"}


@router.get("/sonic-classifier/status")
async def get_sonic_classifier_status(
    request: Request,
    collection: str = Query(..., description="Collection name"),
) -> dict:
    """Return readiness of the classifier."""
    svc = request.app.state.sonic_descriptor_service
    if svc is None:
        raise HTTPException(status_code=503, detail="Sonic Descriptor Service unavailable")
    return svc.get_classifier_status(collection=collection)


@router.get("/sonic-classifier/jobs/{job_id}")
async def get_training_job(job_id: str) -> dict:
    """Inspect the state of a training job."""
    job = _TRAINING_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    return {"job_id": job_id, **job}
