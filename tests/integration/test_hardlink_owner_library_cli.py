"""Owner CLI: hardlink source folder into media/<owner>/audio/ + trigger indexing."""

import subprocess
import sys


def _seed_owner(uid: str = "acct_owner") -> None:
    from app.resources.metadata_db import MetadataDB
    MetadataDB.create_user(
        user_id=uid, email="owner@example.com", password_hash="h", role="owner",
        created_at=1700000000.0,
    )


class TestHardlinkCli:
    def test_invocation_prints_usage_with_no_args(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.hardlink_owner_library"],
            capture_output=True, text=True,
        )
        # argparse exits 2 (missing required args).
        assert result.returncode != 0
        out = (result.stdout + result.stderr).lower()
        assert "--source" in out and "--owner-email" in out

    def test_dry_run_lists_files_without_linking(self, tmp_path, audio_path, monkeypatch):
        src_dir = tmp_path / "music"
        src_dir.mkdir()
        (src_dir / "a.flac").write_bytes(audio_path("tiny.flac").read_bytes())
        (src_dir / "b.mp3").write_bytes(audio_path("tiny.mp3").read_bytes())
        (src_dir / "ignored.txt").write_text("not audio")

        media_root = tmp_path / "media"
        monkeypatch.setenv("MUSIX_MEDIA_ROOT", str(media_root))

        from scripts import hardlink_owner_library as cli
        n = cli.run(
            source=str(src_dir),
            owner_email="owner@example.com",
            owner_id_lookup=lambda email: "acct_owner",
            indexer=lambda *a, **k: None,
            dry_run=True,
        )
        assert n == 2  # flac + mp3, not txt
        assert not media_root.exists()

    def test_hardlinks_and_inserts_pending_uploads(self, clean_metadata_db, tmp_path, audio_path, monkeypatch):
        src_dir = tmp_path / "music"
        src_dir.mkdir()
        (src_dir / "a.flac").write_bytes(audio_path("tiny.flac").read_bytes())

        media_root = tmp_path / "media"
        monkeypatch.setenv("MUSIX_MEDIA_ROOT", str(media_root))
        _seed_owner()

        from app.resources.metadata_db import MetadataDB
        from scripts import hardlink_owner_library as cli
        calls = []
        n = cli.run(
            source=str(src_dir),
            owner_email="owner@example.com",
            owner_id_lookup=lambda email: "acct_owner",
            indexer=lambda **kwargs: calls.append(kwargs),
            dry_run=False,
        )
        assert n == 1
        audio_dir = media_root / "acct_owner" / "audio"
        files = list(audio_dir.glob("*.flac"))
        assert len(files) == 1
        assert files[0].stat().st_nlink >= 2 or \
            files[0].read_bytes() == (src_dir / "a.flac").read_bytes()
        rows = MetadataDB.list_pending_uploads_by_account("acct_owner")
        assert len(rows) == 1
        assert len(calls) == 1
        assert calls[0]["account_id"] == "acct_owner"

    def test_idempotent_on_existing_sha(self, clean_metadata_db, tmp_path, audio_path, monkeypatch):
        src_dir = tmp_path / "music"
        src_dir.mkdir()
        (src_dir / "a.flac").write_bytes(audio_path("tiny.flac").read_bytes())
        media_root = tmp_path / "media"
        monkeypatch.setenv("MUSIX_MEDIA_ROOT", str(media_root))
        _seed_owner()

        from app.resources.metadata_db import MetadataDB
        from scripts import hardlink_owner_library as cli
        for _ in range(2):
            cli.run(
                source=str(src_dir),
                owner_email="owner@example.com",
                owner_id_lookup=lambda email: "acct_owner",
                indexer=lambda **kwargs: None,
                dry_run=False,
            )
        rows = MetadataDB.list_pending_uploads_by_account("acct_owner")
        # Idempotent: second run sees the same SHA already in DB, no new row.
        assert len(rows) == 1
