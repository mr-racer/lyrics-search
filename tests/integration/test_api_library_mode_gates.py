"""Mode-gate matrix: which endpoints exist in sharing vs server mode (Phase C)."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User


def _setup_mode(mode: str) -> None:
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode=mode, created_at=1700000000.0)


def _login(app, uid: str = "acct_alice", role: str = "member") -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role=role, created_at=0.0,
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

    def test_index_member_403_in_server_mode(self, tmp_path):
        """Members must not point the indexer at arbitrary host folders —
        they upload via POST /library/upload instead."""
        _setup_mode("server")
        app = create_app()
        _login(app, role="member")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code == 403

    def test_index_owner_allowed_in_server_mode(self, tmp_path):
        """The owner runs the instance host, so folder indexing (wizard +
        settings) must work in server mode too — not only via uploads."""
        _setup_mode("server")
        app = create_app()
        _login(app, uid="acct_owner", role="owner")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code in (200, 503), resp.text

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


class TestModeGateMatrix:
    """Pin the surface: each endpoint reachable in exactly one mode."""

    SERVER_ONLY = [
        ("POST", "/api/v1/library/upload", "file"),
        ("GET", "/api/v1/library/upload/some-id", None),
        ("POST", "/api/v1/library/upload/batch-commit", {"upload_ids": ["x"]}),
        ("DELETE", "/api/v1/library/tracks/some-track", None),
    ]
    # NOTE: /library/index is no longer sharing-only — in server mode it is
    # owner-only (see TestIndexModeGate) so the wizard can index a host folder.

    @pytest.mark.parametrize("method,path,body", SERVER_ONLY)
    def test_server_only_404s_in_sharing(self, method, path, body):
        _setup_mode("sharing")
        app = create_app()
        _login(app)  # bypass auth so the 404 is unambiguously from the mode gate
        with TestClient(app) as c:
            kwargs = {}
            if body == "file":
                kwargs["files"] = {"file": ("x.flac", b"x", "audio/flac")}
            elif isinstance(body, dict):
                kwargs["json"] = body
            resp = c.request(method, path, **kwargs)
            assert resp.status_code == 404, f"{method} {path}: {resp.status_code} {resp.text}"
