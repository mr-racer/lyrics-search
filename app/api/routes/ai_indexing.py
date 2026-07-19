"""User-triggered AI indexing endpoints.

POST   /library/ai-index/{task_type}        — start a job
GET    /library/ai-index/status             — status per task type
DELETE /library/ai-index/{task_type}/cache  — wipe cache rows for the task
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.domain.models import AIJobStatus, User
from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service

router = APIRouter(prefix="/library/ai-index", tags=["AI Indexing"])

_TASK_TYPES = {"sonic_vibe", "refined_facts", "artist_bio"}


class StartJobRequest(BaseModel):
    lang: str  # "ru" | "en" (free-form for forward-compat)
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    bio_source: Optional[str] = "facts"  # "facts" | "web" — only used by artist_bio


class StartJobResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    sonic_vibe: Optional[AIJobStatus] = None
    refined_facts: Optional[AIJobStatus] = None
    artist_bio: Optional[AIJobStatus] = None


class CacheResetResponse(BaseModel):
    deleted_rows: int


def _count_eligible(db_client, collection_name: str, task_type: str) -> int:
    """Return n_total for the job — how many items the task will process.

    The count depends on the task type:

    - **sonic_vibe**: number of tracks in the collection (one vibe per track).
    - **artist_bio**: number of distinct artists in the collection.
    - **refined_facts**: total number of raw facts (song + artist) that will be
      fed into the LLM.
    """
    if task_type == "sonic_vibe":
        try:
            info = db_client.qdrant.count(
                collection_name=collection_name, exact=True
            )
            return int(info.count)
        except Exception:
            return 0

    if task_type == "artist_bio":
        # Distinct artists from the SQLite mirror (indexed query, instant).
        # 0 means the mirror is empty (pre-backfill) — fall back to scanning
        # Qdrant payloads with the same logic artist_bio.run uses.
        try:
            n = MetadataDB.count_distinct_artist_slugs(collection_name)
            if n:
                return n
        except Exception:
            pass
        try:
            from app.services.artist_split import artist_slugs
            seen: set[str] = set()
            offset = None
            while True:
                points, offset = db_client.qdrant.scroll(
                    collection_name=collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["artist", "artist_slugs"],
                    with_vectors=False,
                )
                if not points:
                    break
                for pt in points:
                    p = pt.payload or {}
                    slugs = p.get("artist_slugs") or []
                    if not slugs:
                        raw = (p.get("artist") or "").strip()
                        if raw:
                            slugs = artist_slugs(raw)
                    seen.update(slugs)
                if offset is None:
                    break
            return len(seen)
        except Exception:
            return 0

    if task_type == "refined_facts":
        try:
            return MetadataDB.count_facts_for_collection(collection_name)
        except Exception:
            return 0

    # Fallback for unknown task types — count tracks
    try:
        info = db_client.qdrant.count(collection_name=collection_name, exact=True)
        return int(info.count)
    except Exception:
        return 0


@router.post("/{task_type}", response_model=StartJobResponse)
async def start_job(
    task_type: str,
    req: StartJobRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StartJobResponse:
    if task_type not in _TASK_TYPES:
        raise HTTPException(status_code=404, detail=f"unknown task_type: {task_type}")
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    derived = derive_collection_for_user(current_user)

    # Off the event loop: the artist_bio fallback path scans the whole
    # collection — run in a worker thread so a slow count can't freeze the app.
    n_total = await asyncio.to_thread(_count_eligible, db_client, derived, task_type)
    try:
        job_id = ai_indexing_service.start_job(
            task_type=task_type,
            collection_name=derived,
            lang=req.lang,
            db_client=db_client,
            llm_client=None,
            n_total=n_total,
            llm_base_url=(req.llm_base_url or "").strip() or None,
            llm_model=(req.llm_model or "").strip() or None,
            bio_source=(req.bio_source or "facts").strip(),
        )
    except ValueError as e:
        # Distinguish "unknown task_type" (404 above) from concurrency conflict.
        msg = str(e).lower()
        if "already running" in msg:
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    return StartJobResponse(job_id=job_id)


@router.get("/status", response_model=StatusResponse)
def status(
    current_user: User = Depends(get_current_user),
) -> StatusResponse:
    derived = derive_collection_for_user(current_user)
    out = StatusResponse()
    for tt in _TASK_TYPES:
        row = MetadataDB.get_latest_ai_job(derived, tt)
        if row:
            setattr(out, tt, AIJobStatus(**{
                **row,
                "started_at":  row["started_at"]  and str(row["started_at"]),
                "finished_at": row["finished_at"] and str(row["finished_at"]),
            }))
    return out


@router.delete("/{task_type}/cache", response_model=CacheResetResponse)
def reset_cache(
    task_type: str,
    current_user: User = Depends(get_current_user),
) -> CacheResetResponse:
    """Drop all cached output rows for the given task type + collection.

    Accessors delete_sonic_vibes / delete_refined_facts are implemented
    by the respective task modules in T14/T15 and become available at
    runtime via MetadataDB.
    """
    derived = derive_collection_for_user(current_user)
    if task_type == "sonic_vibe":
        if not hasattr(MetadataDB, "delete_sonic_vibes"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        n = MetadataDB.delete_sonic_vibes(derived)
    elif task_type == "refined_facts":
        if not hasattr(MetadataDB, "delete_refined_facts"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        # refined_facts is keyed by scope_key (shared pool), not by collection —
        # build this library's song keys with the same slug function the rows
        # were written with, so the reset actually hits them.
        from app.services.song_facts_service import get_song_facts_key
        song_keys = list({
            get_song_facts_key(artist, title)
            for artist, title in MetadataDB.get_artist_title_pairs_for_collection(derived)
            if artist and title
        })
        n = MetadataDB.delete_refined_facts(derived, song_keys=song_keys)
    elif task_type == "artist_bio":
        if not hasattr(MetadataDB, "delete_artist_bios"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        n = MetadataDB.delete_artist_bios(derived)
    else:
        raise HTTPException(status_code=404, detail=f"unknown task_type: {task_type}")
    return CacheResetResponse(deleted_rows=int(n))
