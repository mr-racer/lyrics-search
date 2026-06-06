"""Background cleanup: quarantine *.tmp older than 24h + pending_uploads 'done' older than 7d."""

import os
import time

from app.services.uploads_service import sweep_old_quarantine


class TestQuarantineSweep:
    def test_removes_old_tmp_files(self, tmp_path):
        old = tmp_path / "old.tmp"
        young = tmp_path / "young.tmp"
        old.write_bytes(b"x")
        young.write_bytes(b"y")
        # Backdate `old` to 48h ago.
        forty_eight_h_ago = time.time() - 48 * 3600
        os.utime(old, (forty_eight_h_ago, forty_eight_h_ago))
        n = sweep_old_quarantine(root=tmp_path, older_than_seconds=24 * 3600)
        assert n == 1
        assert not old.exists()
        assert young.exists()

    def test_returns_zero_when_root_missing(self, tmp_path):
        assert sweep_old_quarantine(root=tmp_path / "no-such-dir") == 0


class TestPendingUploadsPurge:
    def test_purge_only_done_older_than_cutoff(self, clean_metadata_db):
        from app.resources.metadata_db import MetadataDB
        MetadataDB.create_user(
            user_id="u1", email="u1@x.y", password_hash="h", role="member",
            created_at=1.0,
        )

        def _mk(sha):
            return MetadataDB.create_pending_upload(
                account_id="u1", sha256=sha, original_filename="x",
                size_bytes=1, storage_path="/x",
            )

        old_done = _mk("a" * 64)
        MetadataDB.update_pending_upload_status(old_done, status="done", track_id="t1")
        conn = MetadataDB.get()
        conn.execute(
            "UPDATE pending_uploads SET created_at = ? WHERE upload_id = ?",
            (time.time() - 10 * 86400, old_done),
        )
        conn.commit()

        young_done = _mk("b" * 64)
        MetadataDB.update_pending_upload_status(young_done, status="done", track_id="t2")

        old_uploaded = _mk("c" * 64)
        conn.execute(
            "UPDATE pending_uploads SET created_at = ? WHERE upload_id = ?",
            (time.time() - 10 * 86400, old_uploaded),
        )
        conn.commit()

        n = MetadataDB.purge_old_pending_uploads(older_than_seconds=7 * 86400)
        assert n == 1  # only old_done
        remaining = {r["upload_id"] for r in MetadataDB.list_pending_uploads_by_account("u1")}
        assert young_done in remaining
        assert old_uploaded in remaining  # NOT purged — wrong status
        assert old_done not in remaining
