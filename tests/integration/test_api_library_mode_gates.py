"""Mode-gate matrix: which endpoints exist in sharing vs server mode (Phase C)."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User


def _setup_mode(mode: str) -> None:
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode=mode, created_at=1700000000.0)


def _login(app, uid: str = "acct_alice") -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


@pytest.fixture(autouse=True)
def _db(clean_metadata_db):
    # Isolated tmp DB per test (instance_config set per-test via _setup_mode).
    yield


class TestIndexModeGate:
    def test_index_allowed_in_sharing_mode(self, tmp_path):
        _setup_mode("sharing")
        app = create_app()
        _login(app)
        with TestClient(app) as c:
            # Real folder so the folder-exists sanity check passes; we only assert
            # the mode gate let us through (200 started or 503 service-down).
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code in (200, 503), resp.text

    def test_index_404s_in_server_mode(self, tmp_path):
        _setup_mode("server")
        app = create_app()
        _login(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code == 404

    def test_upload_404s_in_sharing_mode(self):
        _setup_mode("sharing")
        app = create_app()
        _login(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", b"fake", "audio/flac")},
            )
            assert resp.status_code == 404

    def test_batch_commit_404s_in_sharing_mode(self):
        _setup_mode("sharing")
        app = create_app()
        _login(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload/batch-commit",
                json={"upload_ids": ["x"]},
            )
            assert resp.status_code == 404


class TestSharingIndexSanityCheck:
    def test_missing_folder_returns_400(self):
        _setup_mode("sharing")
        app = create_app()
        _login(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": "/nonexistent/path/xyz", "collection_name": "x"},
            )
            assert resp.status_code == 400
            assert "does not exist" in resp.json()["detail"].lower()
