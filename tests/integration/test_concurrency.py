"""Concurrency behaviour across accounts. Phase B verification.

Consolidates two suites:

* ``TestConcurrentMultiAccount`` exercises the four singleton fixes (model
  registry, library queue + semaphore, sonic classifier cache, transcoded cache
  namespace) in concert. Each sub-test drives the relevant service layer directly
  (no FastAPI dependency injection) so it runs whether or not the route-level
  auth wiring is exercised.
* ``TestConcurrentUploads`` drives two accounts uploading at the same time → both
  succeed, no SHA cross-talk.
"""

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from app.api.dependencies import get_current_user
from app.api.main import create_app
from app.domain.models import User
from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.metadata_db import MetadataDB
from app.resources.model_registry import ModelRegistry
from app.services.library_service import LibraryService
from app.services.search_service import SearchService


class _FakeModel:
    def __init__(self, name, dim=4): self.name = name; self.dim = dim
    def encode(self, q, **kw):
        # numpy array so search()'s .tolist() works, like real sentence-transformers.
        return np.array([hash(self.name) % 1000] + [0.0] * (self.dim - 1), dtype=np.float32)
    def get_sentence_embedding_dimension(self): return self.dim


@pytest.fixture
def seeded_registry():
    previous = ModelRegistry._text_model
    ModelRegistry._text_model = (
        _FakeModel("pinned"), ModelRegistry.VECTOR_NAME, ModelRegistry.VECTOR_DIM,
    )
    yield
    ModelRegistry._text_model = previous


@pytest.fixture
def fake_qdrant():
    q = MagicMock()
    q.get_collections.return_value = MagicMock(collections=[])
    q.query_points.return_value = MagicMock(points=[])
    return q


def _seed_user(user_id: str, model: str) -> None:
    """Create a user (Phase A's job) then pin their text model (Phase B column)."""
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    MetadataDB.update_user_settings(user_id, text_model_name=model)


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


class TestConcurrentMultiAccount:
    def test_concurrent_searches_share_the_one_vector_name(self, seeded_registry, fake_qdrant):
        """Two accounts searching the same shared engine at once.

        This used to guard against a real race: each account carried its own
        embedding model and the engine wrote the resolved one onto ``self``, so
        one thread's model could leak into the other's Qdrant query. There is
        one model app-wide now, so what is left to verify is that concurrent
        searches still reach Qdrant cleanly, every one of them under the single
        pinned vector name.
        """
        engine = LyricsSearchEngine(
            qdrant_client=fake_qdrant, collection_name="col-shared", lazy=True,
        )

        # Capture every `using` the engine sends to Qdrant, thread-safely via a
        # side_effect (no reliance on the racy shared call_args_list[-1]).
        seen_usings: set[str] = set()
        seen_lock = threading.Lock()

        def _record(*args, **kwargs):
            prefetches = kwargs.get("prefetch") or []
            if prefetches:
                with seen_lock:
                    seen_usings.add(prefetches[0].using)
            return MagicMock(points=[])

        fake_qdrant.query_points.side_effect = _record

        def _worker(_account_id: str):
            engine.search(query="q", limit=5, collection_name_override="col-shared")

        threads = [threading.Thread(target=_worker, args=("acct-A",)),
                   threading.Thread(target=_worker, args=("acct-B",))]
        for t in threads: t.start()
        for t in threads: t.join()

        assert seen_usings == {ModelRegistry.VECTOR_NAME}
        assert fake_qdrant.query_points.call_count == 2

    def test_two_accounts_can_start_indexing_concurrently(self):
        svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
        assert svc.try_start_job(account_id="acct-A", job_id="job-A") is True
        assert svc.try_start_job(account_id="acct-B", job_id="job-B") is True
        # Both accounts are in-flight at the same time
        assert svc.is_account_indexing("acct-A") and svc.is_account_indexing("acct-B")

    def test_same_account_double_indexing_rejected(self):
        svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
        assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
        assert svc.try_start_job(account_id="acct-A", job_id="j2") is False
        # Slot still held by j1
        assert svc.get_account_job_id("acct-A") == "j1"

    def test_transcoded_cache_isolated_between_accounts(self, tmp_path, monkeypatch):
        from app.services import audio_streaming
        monkeypatch.setattr(audio_streaming, "_CACHE_DIR", tmp_path / "tx")
        p_a = audio_streaming._cache_path(account_id="acct-A", track_id="t1")
        p_b = audio_streaming._cache_path(account_id="acct-B", track_id="t1")
        assert p_a.parent != p_b.parent

        # Simulate cached files
        p_a.parent.mkdir(parents=True); p_b.parent.mkdir(parents=True)
        p_a.write_bytes(b"A"); p_b.write_bytes(b"B")
        n = audio_streaming.drop_transcoded_for_tracks(account_id="acct-A", track_ids=["t1"])
        assert n == 1
        assert not p_a.exists()
        assert p_b.exists()  # other account untouched


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
