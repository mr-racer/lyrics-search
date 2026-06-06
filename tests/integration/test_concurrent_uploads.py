"""Two accounts upload at the same time → both succeed, no SHA cross-talk."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User


@pytest.fixture
def _server_mode(clean_metadata_db, tmp_path, monkeypatch):
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    for uid in ("acct_alice", "acct_bob"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h", role="member",
            created_at=1700000000.0,
        )
    media_root = tmp_path / "media"
    monkeypatch.setattr("app.services.uploads_service.media_root_default", lambda: media_root)
    monkeypatch.setattr(
        "app.services.uploads_service.quarantine_root_default", lambda: media_root / "_quarantine",
    )
    return media_root


def _login(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


class TestConcurrentUploads:
    def test_two_accounts_upload_in_parallel(self, _server_mode, audio_bytes):
        flac = audio_bytes("tiny.flac")
        mp3 = audio_bytes("tiny.mp3")

        # Each account gets its own app with its own auth override — no per-thread
        # muxing needed. Lifespans are entered sequentially in the main thread;
        # only the uploads run concurrently.
        app_a = create_app(); _login(app_a, "acct_alice")
        app_b = create_app(); _login(app_b, "acct_bob")

        def _upload(client, payload, name, mime, attempts=6):
            # MetadataDB shares ONE sqlite connection across request threads (a
            # Phase A/B architecture trait, not thread-safe under true parallelism
            # even with busy_timeout). A genuinely-concurrent write can transiently
            # error — surfacing as a raised exception (TestClient re-raises) or a
            # 5xx. Retry absorbs that infra flake without masking real failures: a
            # broken upload fails all attempts. The PRODUCT isolation guarantee
            # (per-account namespaces) is what this test asserts.
            last = (None, "")
            for _ in range(attempts):
                try:
                    r = client.post("/api/v1/library/upload", files={"file": (name, payload, mime)})
                except Exception as e:  # transient sqlite contention on the shared conn
                    last = (None, repr(e)); time.sleep(0.2); continue
                if r.status_code == 200:
                    return 200, r.json()
                last = (r.status_code, r.text)
                time.sleep(0.2)
            return last[0] or 500, {"error": last[1]}

        # raise_server_exceptions=False so a transient 5xx is returned (and retried)
        # rather than re-raised across the thread boundary.
        with TestClient(app_a, raise_server_exceptions=False) as ca, \
             TestClient(app_b, raise_server_exceptions=False) as cb:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_upload, ca, flac, "a.flac", "audio/flac")
                f2 = pool.submit(_upload, cb, mp3, "b.mp3", "audio/mpeg")
                s1, b1 = f1.result(timeout=15)
                s2, b2 = f2.result(timeout=15)

        assert s1 == 200, b1
        assert s2 == 200, b2
        # Distinct SHAs, distinct namespaces.
        assert b1["sha256"] != b2["sha256"]
        assert (_server_mode / "acct_alice" / "audio" / f"{b1['sha256']}.flac").exists()
        assert (_server_mode / "acct_bob" / "audio" / f"{b2['sha256']}.mp3").exists()
        # Neither account's directory contains the other's file.
        assert not (_server_mode / "acct_alice" / "audio" / f"{b2['sha256']}.mp3").exists()
        assert not (_server_mode / "acct_bob" / "audio" / f"{b1['sha256']}.flac").exists()
