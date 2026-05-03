"""Library endpoints."""

import asyncio
from collections import Counter
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request

from app.domain.models import IndexRequest, IndexProgress
from app.services.library_service import LibraryService
from app.services.similarity_service import load_top_pairs

router = APIRouter(prefix="/library", tags=["Library"])


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
    if db_client is None:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "duration_buckets": [], "qdrant_available": False}

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "duration_buckets": [], "qdrant_available": False}

    if not cols:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "duration_buckets": [], "qdrant_available": True}

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
        return {"total_tracks": 0, "collection_name": collection_name, "genres": [], "duration_buckets": [], "qdrant_available": True}

    # Sample up to 1 000 points; collect genre + duration buckets from payload
    genre_counter: Counter = Counter()
    duration_counter: Counter = Counter()
    offset = None
    sampled = 0
    SAMPLE_LIMIT = 1000

    try:
        while sampled < SAMPLE_LIMIT:
            results, next_offset = qdrant.scroll(
                collection_name=target_col,
                offset=offset,
                limit=min(100, SAMPLE_LIMIT - sampled),
                with_payload=["genre", "duration"],
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
            sampled += len(results)
            if next_offset is None or not results:
                break
            offset = next_offset
    except Exception:
        pass  # data unavailable — still return total count

    total_sampled = sum(genre_counter.values()) or 1
    top_genres = [
        {"genre": g, "count": c, "pct": round(c / total_sampled * 100)}
        for g, c in genre_counter.most_common(3)
    ]

    total_dur = sum(duration_counter.values()) or 1
    top_durations = [
        {"range": r, "count": c, "pct": round(c / total_dur * 100)}
        for r, c in duration_counter.most_common(6)
    ]

    return {
        "total_tracks": target_count,
        "collection_name": target_col,
        "genres": top_genres,
        "duration_buckets": top_durations,
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
