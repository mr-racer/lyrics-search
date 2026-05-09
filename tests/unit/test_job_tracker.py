"""Tests for app.services.job_tracker."""

import asyncio
import time

from app.services.job_tracker import (
    IndexJob,
    IndexStage,
    IndexStatus,
    JobTracker,
)


class TestIndexJob:
    def test_creates_with_all_stages_pending(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        for stage in IndexStage:
            assert stage in job.stages
            assert job.stages[stage].status == IndexStatus.PENDING

    def test_add_subscriber(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        q = job.add_subscriber()
        assert q in job._subscribers

    def test_remove_subscriber(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        q = job.add_subscriber()
        job.remove_subscriber(q)
        assert q not in job._subscribers

    def test_remove_missing_subscriber(self):
        """Removing a non-existent queue should not raise."""
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        job.remove_subscriber(asyncio.Queue())

    async def test_notify_subscribers(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        q = job.add_subscriber()
        await job.notify_subscribers({"extra": "data"})
        payload = q.get_nowait()
        assert "extra" in payload
        assert payload["extra"] == "data"


class TestCalculateETA:
    def test_no_started_at_returns_none(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        assert job.calculate_eta_seconds(IndexStage.METADATA) is None

    def test_zero_current_returns_none(self):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        job.stages[IndexStage.METADATA].started_at = time.time()
        job.stages[IndexStage.METADATA].total = 100
        assert job.calculate_eta_seconds(IndexStage.METADATA) is None

    def test_correct_eta(self, monkeypatch):
        job = IndexJob(
            job_id="j1", folder_path="/music", collection_name="test"
        )
        stage = job.stages[IndexStage.METADATA]
        stage.started_at = time.time() - 10  # 10 seconds ago
        stage.current = 10
        stage.total = 100
        eta = job.calculate_eta_seconds(IndexStage.METADATA)
        assert eta is not None
        assert eta == 90  # 90 remaining at 1/s


class TestJobTracker:
    def test_singleton(self):
        a = JobTracker()
        b = JobTracker()
        assert a is b

    def test_create_and_get_job(self):
        jt = JobTracker()
        job = jt.create_job("/music", "col")
        found = jt.get_job(job.job_id)
        assert found is job

    def test_get_nonexistent_job(self):
        jt = JobTracker()
        assert jt.get_job("nonexistent") is None

    def test_get_all_jobs(self):
        jt = JobTracker()
        jt.create_job("/a", "c1")
        jt.create_job("/b", "c2")
        assert len(jt.get_all_jobs()) >= 2

    def test_remove_completed_job(self):
        jt = JobTracker()
        job = jt.create_job("/music", "col")
        jt.remove_completed_job(job.job_id)
        assert jt.get_job(job.job_id) is None


class TestProgressSummary:
    def test_summary_structure(self):
        jt = JobTracker()
        job = jt.create_job("/music", "col")
        summary = jt.get_progress_summary(job)
        assert "job_id" in summary
        assert "overall_status" in summary
        assert "overall_percent" in summary
        assert "stages" in summary

    def test_completed_stage_contributes_fully(self):
        jt = JobTracker()
        job = jt.create_job("/music", "col")
        job.stages[IndexStage.LYRICS].status = IndexStatus.COMPLETED
        summary = jt.get_progress_summary(job)
        # LYRICS is 25% weight
        assert summary["overall_percent"] == 25

    def test_running_stage_partial_contribution(self):
        jt = JobTracker()
        job = jt.create_job("/music", "col")
        stage = job.stages[IndexStage.LYRICS]
        stage.status = IndexStatus.RUNNING
        stage.current = 50
        stage.total = 100
        summary = jt.get_progress_summary(job)
        # LYRICS 25% weight * 50% progress = 12.5% → int = 12
        assert summary["overall_percent"] == 12
