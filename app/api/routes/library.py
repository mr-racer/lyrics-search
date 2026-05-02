"""Library endpoints."""

import asyncio
from collections import Counter
from fastapi import APIRouter, HTTPException, Request

from app.domain.models import IndexRequest, IndexProgress
from app.services.library_service import LibraryService

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
async def get_stats(request: Request) -> dict:
    """Library statistics: total tracks, top genres (from Qdrant payload).

    Scrolls up to 1 000 points to sample genre distribution — fast enough
    for a sidebar/tab load while still covering most small-to-mid libraries.

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
        return {"total_tracks": 0, "collection_name": None, "genres": [], "qdrant_available": False}

    try:
        qdrant = db_client.qdrant
        cols = qdrant.get_collections().collections
    except Exception:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "qdrant_available": False}

    if not cols:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "qdrant_available": True}

    # Pick the collection with the most points
    best_col: str | None = None
    best_count: int = 0
    for col in cols:
        try:
            info = qdrant.get_collection(col.name)
            cnt = info.points_count or 0
            if cnt > best_count:
                best_count = cnt
                best_col = col.name
        except Exception:
            pass

    if not best_col:
        return {"total_tracks": 0, "collection_name": None, "genres": [], "qdrant_available": True}

    # Sample up to 1 000 points; collect genre tags from payload
    genre_counter: Counter = Counter()
    offset = None
    sampled = 0
    SAMPLE_LIMIT = 1000

    try:
        while sampled < SAMPLE_LIMIT:
            results, next_offset = qdrant.scroll(
                collection_name=best_col,
                offset=offset,
                limit=min(100, SAMPLE_LIMIT - sampled),
                with_payload=["genre"],
                with_vectors=False,
            )
            for point in results:
                genre = (point.payload or {}).get("genre")
                if genre and str(genre).strip():
                    genre_counter[str(genre).strip()] += 1
            sampled += len(results)
            if next_offset is None or not results:
                break
            offset = next_offset
    except Exception:
        pass  # genre data unavailable — still return total count

    total_sampled = sum(genre_counter.values()) or 1
    top_genres = [
        {"genre": g, "count": c, "pct": round(c / total_sampled * 100)}
        for g, c in genre_counter.most_common(3)
    ]

    return {
        "total_tracks": best_count,
        "collection_name": best_col,
        "genres": top_genres,
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
