"""Unit tests for pending_uploads CRUD on MetadataDB."""

import time

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point MetadataDB at a tmp file for every test in this module.

    pending_uploads.account_id is a FOREIGN KEY into users(id) and the DB runs
    with PRAGMA foreign_keys=ON, so we seed the accounts the tests reference
    (in production account_id always comes from an authenticated current_user).
    """
    db_file = tmp_path / "metadata.db"
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_file)
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    for uid in ("u1", "u2"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h",
            role="member", created_at=1700000000.0,
        )
    yield
    MetadataDB._reset_for_tests()


class TestPendingUploadsSchema:
    def test_table_exists(self):
        conn = MetadataDB.get()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_uploads)")}
        assert {"upload_id", "account_id", "sha256", "original_filename",
                "size_bytes", "storage_path", "status", "track_id",
                "error", "created_at"} <= cols

    def test_account_status_index_exists(self):
        conn = MetadataDB.get()
        idxs = {row[1] for row in conn.execute("PRAGMA index_list(pending_uploads)")}
        assert "idx_pending_uploads_account_status" in idxs


class TestPendingUploadsCrud:
    def _make(self, account_id="u1", sha="0" * 64, name="song.mp3"):
        upload_id = MetadataDB.create_pending_upload(
            account_id=account_id,
            sha256=sha,
            original_filename=name,
            size_bytes=1024,
            storage_path=f"media/{account_id}/audio/{sha}.mp3",
        )
        return upload_id

    def test_create_returns_uuid_like_id(self):
        upload_id = self._make()
        assert isinstance(upload_id, str)
        assert len(upload_id) >= 16

    def test_get_returns_full_row(self):
        upload_id = self._make()
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["upload_id"] == upload_id
        assert row["status"] == "uploaded"
        assert row["account_id"] == "u1"
        assert row["track_id"] is None
        assert row["error"] is None

    def test_get_missing_returns_none(self):
        assert MetadataDB.get_pending_upload("no-such-id") is None

    def test_update_status_to_indexing(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(upload_id, status="indexing")
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "indexing"

    def test_update_status_to_done_sets_track_id(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(
            upload_id, status="done", track_id="abc123",
        )
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "done"
        assert row["track_id"] == "abc123"

    def test_update_status_to_failed_sets_error(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(
            upload_id, status="failed", error="encoding failed",
        )
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "failed"
        assert row["error"] == "encoding failed"

    def test_list_by_account_scoped_to_account(self):
        self._make(account_id="u1", sha="a" * 64)
        self._make(account_id="u1", sha="b" * 64)
        self._make(account_id="u2", sha="c" * 64)
        rows_u1 = MetadataDB.list_pending_uploads_by_account("u1")
        rows_u2 = MetadataDB.list_pending_uploads_by_account("u2")
        assert {r["sha256"] for r in rows_u1} == {"a" * 64, "b" * 64}
        assert {r["sha256"] for r in rows_u2} == {"c" * 64}

    def test_list_by_account_filter_by_status(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        self._make(account_id="u1", sha="b" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        only_uploaded = MetadataDB.list_pending_uploads_by_account(
            "u1", status="uploaded",
        )
        assert len(only_uploaded) == 1
        assert only_uploaded[0]["sha256"] == "b" * 64

    def test_idempotent_lookup_by_sha(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        existing = MetadataDB.find_done_upload_by_sha(
            account_id="u1", sha256="a" * 64,
        )
        assert existing is not None
        assert existing["track_id"] == "t1"
        # Cross-account isolation: same sha, different account = no match.
        assert MetadataDB.find_done_upload_by_sha(
            account_id="u2", sha256="a" * 64,
        ) is None

    def test_purge_old_done_uploads(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        # Backdate to 8 days ago.
        conn = MetadataDB.get()
        conn.execute(
            "UPDATE pending_uploads SET created_at = ? WHERE upload_id = ?",
            (time.time() - 8 * 86400, u1),
        )
        conn.commit()
        # Younger row stays.
        u2 = self._make(account_id="u1", sha="b" * 64)
        MetadataDB.update_pending_upload_status(u2, status="done", track_id="t2")
        n = MetadataDB.purge_old_pending_uploads(older_than_seconds=7 * 86400)
        assert n == 1
        rows = MetadataDB.list_pending_uploads_by_account("u1")
        assert {r["upload_id"] for r in rows} == {u2}
