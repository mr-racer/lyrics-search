"""Tests for AI Indexing — schema, accessors, and (later in T12) the job runner."""

import asyncio

import pytest
from unittest.mock import MagicMock

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_record_ai_job_writes_initial_row():
    MetadataDB.record_ai_job(
        job_id="job-1", task_type="sonic_vibe",
        collection_name="music", lang="en", n_total=100,
    )
    row = MetadataDB._connect().execute(
        "SELECT job_id, task_type, collection_name, lang, status, n_total, n_done, n_failed "
        "FROM ai_indexing_jobs"
    ).fetchone()
    assert row == ("job-1", "sonic_vibe", "music", "en", "queued", 100, 0, 0)


def test_update_ai_job_progress():
    MetadataDB.record_ai_job(
        job_id="job-1", task_type="sonic_vibe",
        collection_name="music", lang="en", n_total=100,
    )
    MetadataDB.update_ai_job(job_id="job-1", status="running", n_done=42, n_failed=3)
    row = MetadataDB._connect().execute(
        "SELECT status, n_done, n_failed FROM ai_indexing_jobs WHERE job_id=?", ("job-1",)
    ).fetchone()
    assert row == ("running", 42, 3)


def test_update_ai_job_marks_finished():
    MetadataDB.record_ai_job(
        job_id="job-1", task_type="sonic_vibe",
        collection_name="music", lang="en", n_total=10,
    )
    MetadataDB.update_ai_job(job_id="job-1", status="done", finished=True)
    row = MetadataDB._connect().execute(
        "SELECT status, finished_at FROM ai_indexing_jobs WHERE job_id=?", ("job-1",)
    ).fetchone()
    assert row[0] == "done"
    assert row[1] is not None  # finished_at populated


def test_update_ai_job_records_error():
    MetadataDB.record_ai_job(
        job_id="job-1", task_type="sonic_vibe",
        collection_name="music", lang="en", n_total=10,
    )
    MetadataDB.update_ai_job(job_id="job-1", status="failed", error="LLM timeout")
    row = MetadataDB._connect().execute(
        "SELECT status, error FROM ai_indexing_jobs WHERE job_id=?", ("job-1",)
    ).fetchone()
    assert row == ("failed", "LLM timeout")


def test_get_latest_ai_job_returns_most_recent_per_task():
    import time
    MetadataDB.record_ai_job("job-a", "sonic_vibe", "music", "en", 10)
    time.sleep(0.01)
    MetadataDB.record_ai_job("job-b", "sonic_vibe", "music", "en", 10)
    latest = MetadataDB.get_latest_ai_job("music", "sonic_vibe")
    assert latest["job_id"] == "job-b"


def test_get_latest_ai_job_returns_none_when_missing():
    assert MetadataDB.get_latest_ai_job("music", "sonic_vibe") is None


def test_get_latest_ai_job_distinct_per_task_type():
    """Different task_types in the same collection don't collide."""
    MetadataDB.record_ai_job("job-a", "sonic_vibe", "music", "en", 10)
    MetadataDB.record_ai_job("job-b", "refined_facts", "music", "en", 10)
    a = MetadataDB.get_latest_ai_job("music", "sonic_vibe")
    b = MetadataDB.get_latest_ai_job("music", "refined_facts")
    assert a["job_id"] == "job-a"
    assert b["job_id"] == "job-b"


# ── T12: Job runner framework tests ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_service_state():
    """Clear module-level job runner state between tests to avoid cross-test pollution."""
    try:
        from app.services import ai_indexing_service
        ai_indexing_service._registry.clear()
        ai_indexing_service._active.clear()
        ai_indexing_service._running_tasks.clear()
    except ImportError:
        pass  # Module doesn't exist yet during Step 1 (failing tests phase).
    yield
    try:
        from app.services import ai_indexing_service
        ai_indexing_service._registry.clear()
        ai_indexing_service._active.clear()
        ai_indexing_service._running_tasks.clear()
    except ImportError:
        pass


async def _noop_task(job, db_client, llm_client):
    """Test task: marks all 5 items done."""
    for i in range(5):
        MetadataDB.update_ai_job(job_id=job.job_id, n_done=i + 1)


def test_register_task_and_start_job():
    """Register a task, start a job, wait for completion, verify DB state."""
    from app.services import ai_indexing_service

    ai_indexing_service.register_task("test_noop", _noop_task)

    async def _scenario():
        job_id = ai_indexing_service.start_job(
            task_type="test_noop",
            collection_name="music",
            lang="en",
            db_client=MagicMock(),
            llm_client=MagicMock(),
            n_total=5,
        )
        assert job_id
        await ai_indexing_service._wait_for_job(job_id)
        return job_id

    job_id = asyncio.run(_scenario())
    job = MetadataDB.get_latest_ai_job("music", "test_noop")
    assert job["status"] == "done"
    assert job["n_done"] == 5
    assert job["finished_at"] is not None


def test_unknown_task_type_raises():
    from app.services import ai_indexing_service

    with pytest.raises(ValueError) as excinfo:
        ai_indexing_service.start_job(
            task_type="totally_made_up",
            collection_name="music",
            lang="en",
            db_client=MagicMock(),
            llm_client=MagicMock(),
            n_total=10,
        )
    assert "unknown task_type" in str(excinfo.value).lower()


def test_concurrent_job_rejected():
    """Starting a second job for the same (collection, task_type) while one is
    already active must raise ValueError."""
    from app.services import ai_indexing_service

    async def _blocking_task(job, db_client, llm_client):
        # Long-running stub — sleeps until cancelled.
        await asyncio.sleep(10)

    ai_indexing_service.register_task("test_blocker", _blocking_task)

    async def _scenario():
        job_id = ai_indexing_service.start_job(
            task_type="test_blocker", collection_name="music", lang="en",
            db_client=MagicMock(), llm_client=MagicMock(), n_total=1000,
        )
        # Immediately try to start another for the same (collection, task_type).
        with pytest.raises(ValueError) as excinfo:
            ai_indexing_service.start_job(
                task_type="test_blocker", collection_name="music", lang="en",
                db_client=MagicMock(), llm_client=MagicMock(), n_total=10,
            )
        assert "already running" in str(excinfo.value).lower()
        # Cleanup — cancel so the test doesn't hold a pending task.
        ai_indexing_service.cancel_job(job_id)
        try:
            await ai_indexing_service._wait_for_job(job_id)
        except asyncio.CancelledError:
            pass

    asyncio.run(_scenario())


def test_failed_task_marks_job_failed():
    from app.services import ai_indexing_service

    async def _failing_task(job, db_client, llm_client):
        raise RuntimeError("simulated LLM error")

    ai_indexing_service.register_task("test_failer", _failing_task)

    async def _scenario():
        job_id = ai_indexing_service.start_job(
            task_type="test_failer", collection_name="music", lang="en",
            db_client=MagicMock(), llm_client=MagicMock(), n_total=1,
        )
        await ai_indexing_service._wait_for_job(job_id)

    asyncio.run(_scenario())
    job = MetadataDB.get_latest_ai_job("music", "test_failer")
    assert job["status"] == "failed"
    assert "simulated LLM error" in (job["error"] or "")
    assert job["finished_at"] is not None


def test_different_collection_or_task_type_allowed_concurrent():
    """Same task_type but different collection — should be allowed concurrently.
    Same collection but different task_type — should also be allowed."""
    from app.services import ai_indexing_service

    async def _slow_task(job, db_client, llm_client):
        await asyncio.sleep(0.05)

    ai_indexing_service.register_task("test_slow_a", _slow_task)
    ai_indexing_service.register_task("test_slow_b", _slow_task)

    async def _scenario():
        a = ai_indexing_service.start_job(
            task_type="test_slow_a", collection_name="music_x", lang="en",
            db_client=MagicMock(), llm_client=MagicMock(), n_total=1,
        )
        # Different collection — must be accepted.
        b = ai_indexing_service.start_job(
            task_type="test_slow_a", collection_name="music_y", lang="en",
            db_client=MagicMock(), llm_client=MagicMock(), n_total=1,
        )
        # Same collection but different task type — must be accepted.
        c = ai_indexing_service.start_job(
            task_type="test_slow_b", collection_name="music_x", lang="en",
            db_client=MagicMock(), llm_client=MagicMock(), n_total=1,
        )
        await ai_indexing_service._wait_for_job(a)
        await ai_indexing_service._wait_for_job(b)
        await ai_indexing_service._wait_for_job(c)
        return a, b, c

    a, b, c = asyncio.run(_scenario())
    assert a != b != c
