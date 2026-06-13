"""POST /library/upload/batch-commit → LibraryService.enqueue_upload_indexing."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User


@pytest.fixture(autouse=True)
def _clean_embed_env(monkeypatch):
    """The embedding model now resolves via SettingsService (DB > env > default).
    Strip EMBED_MODEL so a dev shell value can't make these assertions flaky."""
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    yield


@pytest.fixture
def _server_mode(clean_metadata_db, tmp_path):
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    for uid in ("acct_alice", "acct_bob"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h", role="member",
            created_at=1700000000.0,
        )
    return tmp_path


def _login(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


def _mk(account_id: str, sha: str, name: str) -> str:
    from app.resources.metadata_db import MetadataDB
    return MetadataDB.create_pending_upload(
        account_id=account_id, sha256=sha, original_filename=name,
        size_bytes=10, storage_path="/dev/null",
    )


class TestBatchCommit:
    def test_returns_job_id(self, _server_mode):
        u1 = _mk("acct_alice", "a" * 64, "x.flac")
        u2 = _mk("acct_alice", "b" * 64, "y.flac")
        app = create_app()
        _login(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_abc"
            with TestClient(app) as c:
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

    def test_cross_account_ids_filtered(self, _server_mode):
        my = _mk("acct_alice", "a" * 64, "x.flac")
        not_mine = _mk("acct_bob", "b" * 64, "y.flac")
        app = create_app()
        _login(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_x"
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    json={"upload_ids": [my, not_mine]},
                )
                assert resp.status_code == 200, resp.text
            ids = set(enq.call_args.kwargs.get("upload_ids"))
            assert not_mine not in ids
            assert my in ids

    def test_empty_list_returns_400(self, _server_mode):
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/upload/batch-commit", json={"upload_ids": []})
            assert resp.status_code == 400

    def test_only_foreign_ids_returns_400(self, _server_mode):
        not_mine = _mk("acct_bob", "c" * 64, "z.flac")
        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload/batch-commit",
                json={"upload_ids": [not_mine]},
            )
            assert resp.status_code == 400

    def test_embedding_model_uses_instance_setting_ignoring_body(self, _server_mode):
        """D5: in server mode the embedding model is the admin's instance
        setting; the body text_model is IGNORED (member can't override)."""
        import time
        from app.resources.metadata_db import MetadataDB
        MetadataDB.set_instance_settings(
            {"EMBED_MODEL": "intfloat/multilingual-e5-base"}, time.time(),
        )
        u1 = _mk("acct_alice", "d" * 64, "x.flac")
        app = create_app()
        _login(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_tm"
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    # Body value must be ignored in favor of the instance model.
                    json={"upload_ids": [u1],
                          "text_model": "Qwen/Qwen3-Embedding-0.6B"},
                )
                assert resp.status_code == 200, resp.text
            assert enq.call_args.kwargs.get("text_model") == "intfloat/multilingual-e5-base"

    def test_text_model_optional_defaults_none(self, _server_mode):
        u1 = _mk("acct_alice", "e" * 64, "y.flac")
        app = create_app()
        _login(app, "acct_alice")
        with patch("app.services.library_service.LibraryService.enqueue_upload_indexing") as enq:
            enq.return_value = "job_tm2"
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/library/upload/batch-commit",
                    json={"upload_ids": [u1]},
                )
                assert resp.status_code == 200, resp.text
            assert enq.call_args.kwargs.get("text_model") is None
