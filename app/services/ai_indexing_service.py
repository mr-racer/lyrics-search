"""Generic job runner for user-triggered AI indexing tasks.

The framework is task-agnostic: each task type registers a coroutine of
signature ``async def run(job: JobState, db_client, llm_client) -> None``.
``start_job`` creates a JobState, persists to ``ai_indexing_jobs``, and
spawns an asyncio task that drives state transitions (queued -> running ->
done/failed/cancelled).

Concurrency policy: at most one active job per (collection_name, task_type).
Different collections OR different task types can run concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable

from app.resources.metadata_db import MetadataDB

logger = logging.getLogger(__name__)

TaskFunc = Callable[["JobState", object, object], Awaitable[None]]

# Module-level registries — single-process job state.
_registry: dict[str, TaskFunc] = {}
_active: dict[tuple[str, str], "JobState"] = {}
_running_tasks: dict[str, asyncio.Task] = {}


@dataclass
class JobState:
    job_id: str
    task_type: str
    collection_name: str
    lang: str
    n_total: int
    llm_base_url: str | None = None
    llm_model: str | None = None
    bio_source: str = "facts"  # "facts" | "web" — used by artist_bio task
    # Three states, and the difference between the last two is the whole point:
    #   None    -> process the WHOLE collection (manual
    #              /library/ai-index/{task_type}, a deliberate backfill);
    #   (a, b)  -> process exactly these tracks (the batch just indexed);
    #   ()      -> process NOTHING — this run touched no track.
    # An empty tuple used to be falsy-tested together with None, so a rescan
    # that added nothing re-enriched all 5,600 tracks: the single most common
    # append outcome triggered the most expensive possible job. Test with
    # ``is not None``, never for truthiness.
    new_track_ids: tuple[str, ...] | None = None
    n_done: int = 0       # actually processed (LLM called OR served from cache)
    n_failed: int = 0     # LLM call / validation / persistence error
    n_skipped: int = 0    # eligible track but task had nothing to feed the LLM
    status: str = "queued"
    # tz-aware now() — datetime.utcnow() is deprecated in Python 3.12+
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def register_task(task_type: str, fn: TaskFunc) -> None:
    """Register a task implementation. Idempotent — last registration wins."""
    _registry[task_type] = fn


def start_job(
    *,
    task_type: str,
    collection_name: str,
    lang: str,
    db_client,
    llm_client,
    n_total: int,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    bio_source: str = "facts",
    new_track_ids: tuple[str, ...] | None = None,
) -> str:
    """Start a new AI indexing job.

    Returns the new job_id. Raises ValueError if (a) task_type is not
    registered or (b) a job is already running for this (collection, task_type).

    ``new_track_ids`` restricts the job to the tracks just indexed (auto
    AI-indexing from an append/upload run). ``None`` keeps the legacy
    whole-collection behaviour for the manual user-triggered entry point.
    """
    if task_type not in _registry:
        raise ValueError(f"unknown task_type: {task_type}")

    key = (collection_name, task_type)
    existing = _active.get(key)
    if existing is not None and existing.status in {"queued", "running"}:
        raise ValueError(
            f"job for {task_type} on {collection_name} is already running "
            f"(job_id={existing.job_id})"
        )

    job_id = str(uuid.uuid4())
    job = JobState(
        job_id=job_id, task_type=task_type,
        collection_name=collection_name, lang=lang, n_total=n_total,
        llm_base_url=llm_base_url, llm_model=llm_model,
        bio_source=bio_source,
        new_track_ids=new_track_ids,
    )
    _active[key] = job
    MetadataDB.record_ai_job(job_id, task_type, collection_name, lang, n_total)

    async def _run():
        job.status = "running"
        MetadataDB.update_ai_job(job_id=job_id, status="running")
        try:
            await _registry[task_type](job, db_client, llm_client)
            job.status = "done"
            MetadataDB.update_ai_job(
                job_id=job_id, status="done", finished=True,
            )
        except asyncio.CancelledError:
            job.status = "cancelled"
            MetadataDB.update_ai_job(
                job_id=job_id, status="cancelled", finished=True,
            )
            raise
        except Exception as e:
            logger.exception(
                "[ai-indexing] task %s job %s failed", task_type, job_id,
            )
            job.status = "failed"
            MetadataDB.update_ai_job(
                job_id=job_id, status="failed",
                error=str(e)[:500], finished=True,
            )
        finally:
            # Free both module-level dicts so finished jobs don't accumulate
            # over the server's lifetime. Subsequent start_job() calls for the
            # same (collection, task_type) will re-create entries fresh.
            _active.pop(key, None)
            _running_tasks.pop(job_id, None)

    _running_tasks[job_id] = asyncio.create_task(_run())
    return job_id


async def _wait_for_job(job_id: str) -> None:
    """Test helper — block until the job task is finished (or cancelled)."""
    task = _running_tasks.get(job_id)
    if task is not None:
        try:
            await task
        except Exception:
            # Failures are recorded in DB by _run's except handler; the
            # task itself may re-raise from CancelledError. Swallow here so
            # tests that wait for completion don't blow up on cancel.
            pass


async def wait_for_job(job_id: str) -> None:
    """Await a started job to completion (or failure/cancel).

    Public wrapper over ``_wait_for_job`` for callers (e.g. the server-mode
    upload runner) that start a job and must block until it finishes before
    proceeding. Safe to call after the job already finished (no-op).
    """
    await _wait_for_job(job_id)


def cancel_job(job_id: str) -> bool:
    """Cancel a running job. Returns True if the job was running."""
    task = _running_tasks.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        return True
    return False
