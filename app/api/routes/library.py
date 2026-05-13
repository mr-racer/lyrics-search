"""Library endpoints."""

import asyncio
import heapq
import logging
from collections import Counter
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

from app.domain.models import IndexRequest, IndexProgress
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

    result = []
    for col in cols:
        try:
            info = qdrant.get_collection(col.name)
            count = info.points_count or 0
        except Exception:
            count = 0
        result.append({"name": col.name, "count": count})

    return {
        "collections": result,
        "total_points": sum(c["count"] for c in result),
        "qdrant_available": True,
    }


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

    try:
        qdrant = db_client.qdrant
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
