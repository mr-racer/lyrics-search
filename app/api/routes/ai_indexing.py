"""User-triggered AI indexing endpoints.

POST   /library/ai-index/{task_type}        — start a job
GET    /library/ai-index/status?collection= — status per task type
DELETE /library/ai-index/{task_type}/cache  — wipe cache rows for the task
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app.domain.models import AIJobStatus
from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service

router = APIRouter(prefix="/library/ai-index", tags=["AI Indexing"])

_TASK_TYPES = {"sonic_vibe", "refined_facts", "artist_bio"}


class StartJobRequest(BaseModel):
    collection_name: str
    lang: str  # "ru" | "en" (free-form for forward-compat)


class StartJobResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    sonic_vibe: Optional[AIJobStatus] = None
    refined_facts: Optional[AIJobStatus] = None
    artist_bio: Optional[AIJobStatus] = None


class CacheResetResponse(BaseModel):
    deleted_rows: int


def _count_eligible(db_client, collection_name: str) -> int:
    """Return n_total for the job — how many tracks the task will scan.

    Counts ALL points in the collection. Individual tasks (sonic_vibe,
    refined_facts) may further skip tracks at run time based on their own
    eligibility rules.
    """
    try:
        info = db_client.qdrant.count(collection_name=collection_name, exact=True)
        return int(info.count)
    except Exception:
        return 0


@router.post("/{task_type}", response_model=StartJobResponse)
async def start_job(task_type: str, req: StartJobRequest, request: Request) -> StartJobResponse:
    if task_type not in _TASK_TYPES:
        raise HTTPException(status_code=404, detail=f"unknown task_type: {task_type}")
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    n_total = _count_eligible(db_client, req.collection_name)
    try:
        job_id = ai_indexing_service.start_job(
            task_type=task_type,
            collection_name=req.collection_name,
            lang=req.lang,
            db_client=db_client,
            llm_client=None,  # tasks resolve the llm client themselves at run time
            n_total=n_total,
        )
    except ValueError as e:
        # Distinguish "unknown task_type" (404 above) from concurrency conflict.
        msg = str(e).lower()
        if "already running" in msg:
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    return StartJobResponse(job_id=job_id)


@router.get("/status", response_model=StatusResponse)
def status(collection: str = Query(...)) -> StatusResponse:
    out = StatusResponse()
    for tt in _TASK_TYPES:
        row = MetadataDB.get_latest_ai_job(collection, tt)
        if row:
            setattr(out, tt, AIJobStatus(**{
                **row,
                "started_at":  row["started_at"]  and str(row["started_at"]),
                "finished_at": row["finished_at"] and str(row["finished_at"]),
            }))
    return out


@router.delete("/{task_type}/cache", response_model=CacheResetResponse)
def reset_cache(task_type: str, collection: str = Query(...)) -> CacheResetResponse:
    """Drop all cached output rows for the given task type + collection.

    Accessors delete_sonic_vibes / delete_refined_facts are implemented
    by the respective task modules in T14/T15 and become available at
    runtime via MetadataDB.
    """
    if task_type == "sonic_vibe":
        if not hasattr(MetadataDB, "delete_sonic_vibes"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        n = MetadataDB.delete_sonic_vibes(collection)
    elif task_type == "refined_facts":
        if not hasattr(MetadataDB, "delete_refined_facts"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        n = MetadataDB.delete_refined_facts(collection)
    elif task_type == "artist_bio":
        if not hasattr(MetadataDB, "delete_artist_bios"):
            raise HTTPException(status_code=501, detail="cache reset not yet implemented")
        n = MetadataDB.delete_artist_bios(collection)
    else:
        raise HTTPException(status_code=404, detail=f"unknown task_type: {task_type}")
    return CacheResetResponse(deleted_rows=int(n))
