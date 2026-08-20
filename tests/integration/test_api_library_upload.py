"""POST /library/upload — multipart upload → quarantine → atomic promote → row insert.

Also covers POST /library/upload/batch-commit → LibraryService.enqueue_upload_indexing
(merged from test_api_library_upload_batch_commit.py; see ``class TestBatchCommit``).
"""

import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User
from app.services.library_service import LibraryService


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def _server_mode(clean_metadata_db, tmp_path, monkeypatch):
    """Instance mode = 'server', seed accounts, redirect media dirs to tmp.

    Relies on the autouse ``clean_metadata_db`` fixture (tests/integration/conftest)
    for an isolated, thread-safe tmp DB.
    """
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    for uid, email in [("acct_alice", "alice@example.com"), ("acct_bob", "bob@example.com")]:
        MetadataDB.create_user(
            user_id=uid, email=email, password_hash="h", role="member",
            created_at=1700000000.0,
        )
    media_root = tmp_path / "media"
    monkeypatch.setattr("app.services.uploads_service.media_root_default", lambda: media_root)
    monkeypatch.setattr(
        "app.services.uploads_service.quarantine_root_default", lambda: media_root / "_quarantine",
    )
    return media_root


def _login(app, uid: str, email: str = "x@example.com") -> None:
    """Override the auth dependency (router gate + route param share the key)."""
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=email, role="member", created_at=0.0,
    )


from contextlib import contextmanager


@contextmanager
def _client_with_library(app):
    """TestClient whose app.state carries a LibraryService.

    The lifespan leaves ``library_service`` None whenever Qdrant is unreachable
    — which it always is under test — and batch-commit 503s on that check before
    it ever reaches the service. These tests are about the route's own ownership
    filtering and hand-off, so give it something to hand off to. Installed after
    __enter__ because the lifespan would otherwise overwrite it.
    """
    with TestClient(app) as c:
        c.app.state.library_service = LibraryService()
        yield c


# ---- batch-commit helpers (merged from test_api_library_upload_batch_commit.py) ----
# These keep distinct names from the upload helpers above because their bodies
# differ (no media-dir redirect; per-uid emails); the batch tests reference them.


@pytest.fixture
def _server_mode_batch(clean_metadata_db, tmp_path):
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    for uid in ("acct_alice", "acct_bob"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h", role="member",
            created_at=1700000000.0,
        )
    return tmp_path


def _login_batch(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


def _mk(account_id: str, sha: str, name: str) -> str:
    from app.resources.metadata_db import MetadataDB
    return MetadataDB.create_pending_upload(
        account_id=account_id, sha256=sha, original_filename=name,
        size_bytes=10, storage_path="/dev/null",
    )


class TestUploadHappyPath:
    def test_flac_upload_returns_queued(self, _server_mode, audio_bytes):
        app = create_app()
        _login(app, "acct_alice", "alice@example.com")
        with TestClient(app) as c:
            data = audio_bytes("tiny.flac")
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", data, "audio/flac")},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "upload_id" in body
            assert body["sha256"] == _sha(data)
            assert body["size"] == len(data)
            assert body["status"] == "uploaded"
            dest = _server_mode / "acct_alice" / "audio" / f"{body['sha256']}.flac"
            assert dest.exists()
            assert dest.read_bytes() == data

    def test_mp3_upload(self, _server_mode, audio_bytes):
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            data = audio_bytes("tiny.mp3")
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.mp3", data, "audio/mpeg")},
            )
            assert resp.status_code == 200, resp.text
            assert (_server_mode / "acct_alice" / "audio" / f"{resp.json()['sha256']}.mp3").exists()


class TestUploadIdempotent:
    def test_second_upload_same_sha_returns_existing(self, _server_mode, audio_bytes):
        from app.resources.metadata_db import MetadataDB
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            data = audio_bytes("tiny.flac")
            first = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", data, "audio/flac")},
            ).json()
            MetadataDB.update_pending_upload_status(
                first["upload_id"], status="done", track_id="t_fake",
            )
            second = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", data, "audio/flac")},
            ).json()
            assert second["status"] == "done"
            assert second["sha256"] == first["sha256"]
            assert second.get("track_id") == "t_fake"


class TestUploadRejected:
    def test_oversize_returns_400(self, _server_mode, monkeypatch):
        monkeypatch.setattr("app.services.uploads_service.MAX_UPLOAD_BYTES", 100)
        big = b"x" * 1024
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", big, "audio/flac")},
            )
            assert resp.status_code == 400
            assert "too large" in resp.json()["detail"].lower() or \
                   "exceeds" in resp.json()["detail"].lower()

    def test_non_audio_returns_400(self, _server_mode, monkeypatch):
        # Stub the sniff rather than depend on libmagic being installed: what
        # this test pins is the ROUTE's rule — a DEFINITE non-audio verdict
        # rejects even when the filename claims .flac. Without the stub, a host
        # without libmagic returns application/octet-stream, the documented
        # inconclusive case, where the extension whitelist is trusted on purpose
        # and the upload is accepted.
        monkeypatch.setattr(
            "app.api.routes.library.sniff_audio_mime", lambda head: "text/x-shellscript",
        )
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", b"#!/bin/sh\necho hi\n", "text/plain")},
            )
            # libmagic gives a definite non-audio sniff → reject despite .flac name.
            assert resp.status_code == 400


class TestCrossAccount:
    def test_same_sha_two_accounts_two_files(self, _server_mode, audio_bytes):
        data = audio_bytes("tiny.flac")
        for uid in ("acct_alice", "acct_bob"):
            app = create_app()
            _login(app, uid)
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/library/upload",
                    files={"file": ("song.flac", data, "audio/flac")},
                )
                assert resp.status_code == 200, resp.text
        sha = _sha(data)
        assert (_server_mode / "acct_alice" / "audio" / f"{sha}.flac").exists()
        assert (_server_mode / "acct_bob" / "audio" / f"{sha}.flac").exists()


class TestUploadStatus:
    def test_owner_can_read_own_upload(self, _server_mode, audio_bytes):
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            data = audio_bytes("tiny.flac")
            created = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", data, "audio/flac")},
            ).json()
            resp = c.get(f"/api/v1/library/upload/{created['upload_id']}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "uploaded"
            assert body["track_id"] is None
            assert body["error"] is None

    def test_cross_account_returns_404(self, _server_mode, audio_bytes):
        data = audio_bytes("tiny.flac")
        # Alice uploads.
        app_a = create_app()
        _login(app_a, "acct_alice")
        with TestClient(app_a) as c:
            up = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", data, "audio/flac")},
            ).json()
        # Bob tries to peek at Alice's upload.
        app_b = create_app()
        _login(app_b, "acct_bob")
        with TestClient(app_b) as c:
            resp = c.get(f"/api/v1/library/upload/{up['upload_id']}")
            assert resp.status_code == 404

    def test_unknown_upload_returns_404(self, _server_mode):
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/upload/no-such-id")
            assert resp.status_code == 404


class TestBatchCommit:
    @pytest.fixture(autouse=True)
    def _clean_embed_env(self, monkeypatch):
        """The embedding model now resolves via SettingsService (DB > env > default).
        Strip EMBED_MODEL so a dev shell value can't make these assertions flaky."""
        monkeypatch.delenv("EMBED_MODEL", raising=False)
        yield

    def test_returns_job_id(self, _server_mode_batch):
        u1 = _mk("acct_alice", "a" * 64, "x.flac")
        u2 = _mk("acct_alice", "b" * 64, "y.flac")
        app = create_app()
        _login_batch(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_abc"
            with _client_with_library(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    json={"upload_ids": [u1, u2]},
                )
                assert resp.status_code == 200, resp.text
                assert resp.json() == {"job_id": "job_abc"}
            enq.assert_called_once()
            kwargs = enq.call_args.kwargs
            assert kwargs.get("account_id") == "acct_alice"
            assert set(kwargs.get("upload_ids")) == {u1, u2}

    def test_cross_account_ids_filtered(self, _server_mode_batch):
        my = _mk("acct_alice", "a" * 64, "x.flac")
        not_mine = _mk("acct_bob", "b" * 64, "y.flac")
        app = create_app()
        _login_batch(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_x"
            with _client_with_library(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    json={"upload_ids": [my, not_mine]},
                )
                assert resp.status_code == 200, resp.text
            ids = set(enq.call_args.kwargs.get("upload_ids"))
            assert not_mine not in ids
            assert my in ids

    def test_empty_list_returns_400(self, _server_mode_batch):
        app = create_app()
        _login_batch(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/upload/batch-commit", json={"upload_ids": []})
            assert resp.status_code == 400

    def test_only_foreign_ids_returns_400(self, _server_mode_batch):
        not_mine = _mk("acct_bob", "c" * 64, "z.flac")
        app = create_app()
        _login_batch(app, "acct_alice")
        with _client_with_library(app) as c:
            resp = c.post(
                "/api/v1/library/upload/batch-commit",
                json={"upload_ids": [not_mine]},
            )
            assert resp.status_code == 400

    def test_body_text_model_is_accepted_and_ignored(self, _server_mode_batch):
        """One embedding model app-wide. A stale client may still send the
        field — it must not 422, and it must not reach the service."""
        u1 = _mk("acct_alice", "d" * 64, "x.flac")
        app = create_app()
        _login_batch(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_tm"
            with _client_with_library(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    json={"upload_ids": [u1],
                          "text_model": "Qwen/Qwen3-Embedding-0.6B"},
                )
                assert resp.status_code == 200, resp.text
            assert "text_model" not in enq.call_args.kwargs


class TestFolderIndexEmbedModel:
    """POST /library/index (server-mode member, MEMBER_INDEX_ROOT opt-in).

    There is one embedding model app-wide, so the body can no longer select one.
    The field is still accepted so a cached client does not 422.
    """

    def test_server_member_index_ignores_body_text_model(self, _server_mode_batch, monkeypatch):
        from unittest.mock import AsyncMock

        root = _server_mode_batch  # tmp_path — a real dir, passes the is_dir() check
        monkeypatch.setenv("MEMBER_INDEX_ROOT", str(root))
        app = create_app()
        _login_batch(app, "acct_alice")
        with patch(
            "app.services.library_service.LibraryService.index_folder",
            new_callable=AsyncMock,
        ) as idx:
            idx.return_value = {"status": "completed", "count": 0, "message": "ok"}
            with TestClient(app) as c:
                # This test exercises route-level model resolution, not Qdrant.
                # In limited mode (no live Qdrant) the lifespan leaves
                # library_service=None; inject a bare instance so the request
                # reaches index_folder (which is mocked above).
                from app.services.library_service import LibraryService
                if app.state.library_service is None:
                    app.state.library_service = object.__new__(LibraryService)
                resp = c.post(
                    "/api/v1/library/index",
                    json={"folder_path": str(root),
                          "text_model": "Qwen/Qwen3-Embedding-0.6B"},
                )
                assert resp.status_code == 200, resp.text
            assert "text_model" not in idx.call_args.kwargs
