"""Integration test for scripts.create_owner — atomicity + idempotency."""
import os
import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest


def _run(env, *args):
    """Invoke the CLI as a real subprocess to exercise argv parsing."""
    cmd = [sys.executable, "-m", "scripts.create_owner", *args]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    return proc


@pytest.fixture
def env(tmp_path):
    """Point MetadataDB at a tmp file via env override."""
    env = os.environ.copy()
    env["MUSIX_METADATA_DB"] = str(tmp_path / "owner_cli.db")
    env["MUSIX_JWT_SECRET"] = "x" * 32
    yield env


def test_create_owner_writes_user_and_instance_config(env):
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


def test_create_owner_idempotent_second_run_fails(env):
    _run(env, "--email", "o@x.y", "--password", "ownerpass12345", "--mode", "sharing")
    proc2 = _run(env, "--email", "o2@x.y", "--password", "ownerpass12345", "--mode", "sharing")
    assert proc2.returncode != 0
    assert "already" in (proc2.stderr + proc2.stdout).lower()


def test_create_owner_server_mode(env):
    proc = _run(
        env, "--email", "srv@example.com", "--password", "srvownerpass1",
        "--mode", "server",
    )
    assert proc.returncode == 0
    conn = sqlite3.connect(env["MUSIX_METADATA_DB"])
    cfg = conn.execute("SELECT mode FROM instance_config").fetchone()
    assert cfg == ("server",)


def test_create_owner_invalid_mode_rejected(env):
    proc = _run(
        env, "--email", "x@y.z", "--password", "abcdefgh1234", "--mode", "weird",
    )
    assert proc.returncode != 0


def test_create_owner_short_password_rejected(env):
    proc = _run(env, "--email", "x@y.z", "--password", "short", "--mode", "sharing")
    assert proc.returncode != 0
    assert "password" in (proc.stderr + proc.stdout).lower()
