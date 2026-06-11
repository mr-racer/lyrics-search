"""Integration tests for /library/ai-index routes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import ai_indexing as ai_route
from app.resources.metadata_db import MetadataDB
from app.services import ai_indexing_service
from ._auth_helper import authenticate_test_client

# Fixed user used for tests that need to verify collection-name derivation.
_FIXED_USER = SimpleNamespace(id="user-A", email="a@x", role="owner", created_at=0.0)
_FIXED_COLLECTION = "acct_user-A"


@pytest.fixture(autouse=True)
def _reset_service_state():
    ai_indexing_service._registry.clear()
    ai_indexing_service._active.clear()
    ai_indexing_service._running_tasks.clear()
    yield
    ai_indexing_service._registry.clear()
    ai_indexing_service._active.clear()
    ai_indexing_service._running_tasks.clear()


async def _instant_task(job, db_client, llm_client):
    """Stub task that finishes immediately."""
    MetadataDB.update_ai_job(job_id=job.job_id, n_done=1)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    # Register the two real task types under stub implementations so the
    # route's task_type validation passes without invoking real LLM logic.
    ai_indexing_service.register_task("sonic_vibe", _instant_task)
    ai_indexing_service.register_task("refined_facts", _instant_task)

    qdrant = MagicMock()
    qdrant.count.return_value = MagicMock(count=42)
    db_stub = MagicMock()
    db_stub.qdrant = qdrant
    app.state.db_client = db_stub

    # Override get_current_user so collection derivation is deterministic.
    app.dependency_overrides[ai_route.get_current_user] = lambda: _FIXED_USER

    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c

    app.dependency_overrides.pop(ai_route.get_current_user, None)
    MetadataDB._reset_for_tests()


def test_post_ai_index_starts_job(client):
    resp = client.post(
        "/api/v1/library/ai-index/sonic_vibe",
        json={"collection_name": "acct_BAD", "lang": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    # Server derives collection from JWT user → _FIXED_COLLECTION, ignores supplied value.
    # The seeded n_total comes from qdrant.count → 42.
    row = MetadataDB.get_latest_ai_job(_FIXED_COLLECTION, "sonic_vibe")
    assert row["n_total"] == 42


def test_post_ai_index_unknown_task_type(client):
    resp = client.post(
        "/api/v1/library/ai-index/totally_not_real",
        json={"collection_name": "music", "lang": "en"},
    )
    assert resp.status_code == 404


def test_post_ai_index_db_unavailable_returns_503(client):
    app.state.db_client = None
    resp = client.post(
        "/api/v1/library/ai-index/sonic_vibe",
        json={"collection_name": "music", "lang": "en"},
    )
    assert resp.status_code == 503


def test_status_returns_nulls_when_no_jobs(client):
    resp = client.get(
        "/api/v1/library/ai-index/status",
        params={"collection": "music"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sonic_vibe"] is None
    assert body["refined_facts"] is None


def test_status_returns_latest_job_per_task(client):
    # Seed a manual job row using the derived collection (_FIXED_COLLECTION).
    # Server ignores the ?collection= param and always reads from _FIXED_COLLECTION.
    MetadataDB.record_ai_job("seeded-job", "sonic_vibe", _FIXED_COLLECTION, "en", 100)
    MetadataDB.update_ai_job(job_id="seeded-job", status="done", n_done=100, finished=True)
    resp = client.get(
        "/api/v1/library/ai-index/status",
        params={"collection": "music"},  # ignored by server
    )
    body = resp.json()
    assert body["sonic_vibe"] is not None
    assert body["sonic_vibe"]["job_id"] == "seeded-job"
    assert body["sonic_vibe"]["status"] == "done"
    assert body["sonic_vibe"]["n_total"] == 100
    assert body["refined_facts"] is None


def test_delete_cache_unknown_task_type(client):
    resp = client.delete(
        "/api/v1/library/ai-index/totally_not_real/cache",
        params={"collection": "music"},
    )
    assert resp.status_code == 404


def test_artist_bio_task_type_accepted(client):
    """POST /library/ai-index/artist_bio should be 200 (or 400 for already-running)."""
    r = client.post(
        "/api/v1/library/ai-index/artist_bio",
        json={"collection_name": "test_collection", "lang": "en"},
    )
    assert r.status_code in (200, 400, 409, 503)  # not 404 — task_type is recognised


def test_status_response_includes_artist_bio_field(client):
    r = client.get("/api/v1/library/ai-index/status?collection=test_collection")
    assert r.status_code == 200
    body = r.json()
    assert "artist_bio" in body


def test_artist_bio_cache_reset_works(client):
    r = client.delete(
        "/api/v1/library/ai-index/artist_bio/cache?collection=test_collection"
    )
    assert r.status_code == 200
    assert "deleted_rows" in r.json()


def test_start_job_ignores_supplied_collection():
    """D-soft: server derives collection from JWT even when client sends a different one."""
    from app.services import ai_indexing_service
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x", role="owner", created_at=0.0)
    app_under_test = create_app()
    app_under_test.dependency_overrides[ai_route.get_current_user] = lambda: fixed
    with patch.object(ai_indexing_service, "start_job", return_value="job-1") as s:
        with TestClient(app_under_test) as c:
            c.app.state.db_client = MagicMock()
            c.app.state.db_client.qdrant.count.return_value = MagicMock(count=0)
            c.post(
                "/api/v1/library/ai-index/sonic_vibe",
                json={"collection_name": "acct_BAD", "lang": "en"},
            )
        assert s.call_args.kwargs["collection_name"] == "acct_user-A"
