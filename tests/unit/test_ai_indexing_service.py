"""Tests for AI Indexing — schema, accessors, and (later in T12) the job runner."""

import pytest

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
