"""Integration tests for owner-related CLI scripts.

Consolidated from:
  - test_create_owner_cli.py            -> class TestCreateOwnerCli
  - test_hardlink_owner_library_cli.py  -> class TestHardlinkCli
"""

import os
import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest


def _run(env, *args):
    """Invoke the create_owner CLI as a real subprocess to exercise argv parsing."""
    cmd = [sys.executable, "-m", "scripts.create_owner", *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    return proc


def _seed_owner(uid: str = "acct_owner") -> None:
    from app.resources.metadata_db import MetadataDB
    MetadataDB.create_user(
        user_id=uid, email="owner@example.com", password_hash="h", role="owner",
        created_at=1700000000.0,
    )


@pytest.fixture
def env(tmp_path):
    """Point MetadataDB at a tmp file via env override."""
    env = os.environ.copy()
    env["MUSIX_METADATA_DB"] = str(tmp_path / "owner_cli.db")
    env["MUSIX_JWT_SECRET"] = "x" * 32
    yield env


class TestCreateOwnerCli:
    """scripts.create_owner — atomicity + idempotency."""

    def test_create_owner_writes_user_and_instance_config(self, env):
        proc = _run(
            env, "--email", "owner@example.com", "--password", "ownerpass12345",
            "--mode", "sharing",
        )
        assert proc.returncode == 0, proc.stderr
        assert "owner created" in proc.stdout.lower()
        conn = sqlite3.connect(env["MUSIX_METADATA_DB"])
        u = conn.execute("SELECT email, role FROM users").fetchone()
        assert u == ("owner@example.com", "owner")
        cfg = conn.execute("SELECT mode FROM instance_config").fetchone()
        assert cfg == ("sharing",)

    def test_create_owner_idempotent_second_run_fails(self, env):
        _run(env, "--email", "o@x.y", "--password", "ownerpass12345", "--mode", "sharing")
        proc2 = _run(env, "--email", "o2@x.y", "--password", "ownerpass12345", "--mode", "sharing")
        assert proc2.returncode != 0
        assert "already" in (proc2.stderr + proc2.stdout).lower()

    def test_create_owner_server_mode(self, env):
        proc = _run(
            env, "--email", "srv@example.com", "--password", "srvownerpass1",
            "--mode", "server",
        )
        assert proc.returncode == 0
        conn = sqlite3.connect(env["MUSIX_METADATA_DB"])
        cfg = conn.execute("SELECT mode FROM instance_config").fetchone()
        assert cfg == ("server",)

    def test_create_owner_invalid_mode_rejected(self, env):
        proc = _run(
            env, "--email", "x@y.z", "--password", "abcdefgh1234", "--mode", "weird",
        )
        assert proc.returncode != 0

    def test_create_owner_short_password_rejected(self, env):
        proc = _run(env, "--email", "x@y.z", "--password", "short", "--mode", "sharing")
        assert proc.returncode != 0
        assert "password" in (proc.stderr + proc.stdout).lower()


class TestHardlinkCli:
    """Owner CLI: hardlink source folder into media/<owner>/audio/ + trigger indexing."""

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
