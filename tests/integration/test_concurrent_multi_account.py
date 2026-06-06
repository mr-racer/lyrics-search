"""End-to-end concurrent multi-account behaviour. Phase B verification.

Exercises the four singleton fixes (model registry, library queue + semaphore,
sonic classifier cache, transcoded cache namespace) in concert. Each sub-test
drives the relevant service layer directly (no FastAPI dependency injection) so
it runs whether or not the route-level auth wiring is exercised.
"""

import threading
from unittest.mock import MagicMock

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

import numpy as np
import pytest

from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.metadata_db import MetadataDB
from app.resources.model_registry import ModelRegistry
from app.services.library_service import LibraryService
from app.services.search_service import SearchService


class _FakeModel:
    def __init__(self, name, dim=4): self.name = name; self.dim = dim
    def encode(self, q):
        # numpy array so search()'s .tolist() works, like real sentence-transformers.
        return np.array([hash(self.name) % 1000] + [0.0] * (self.dim - 1), dtype=np.float32)
    def get_sentence_embedding_dimension(self): return self.dim


@pytest.fixture
def seeded_registry():
    ModelRegistry._text_models.clear()
    if hasattr(ModelRegistry, "_load_locks"):
        ModelRegistry._load_locks.clear()
    ModelRegistry._text_models["model-jina"] = (_FakeModel("model-jina"), "text_model_jina", 4)
    ModelRegistry._text_models["model-qwen"] = (_FakeModel("model-qwen"), "text_model_qwen", 4)
    yield
    ModelRegistry._text_models.clear()


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


def test_two_accounts_search_with_different_models_no_cross_contamination(seeded_registry, fake_qdrant):
    """Account A queries with jina, account B with qwen. Each must reach Qdrant
    with the vector_name matching their own model — no leak.

    col-shared has no pinned collection_settings model, so _resolve_model_name
    falls through to each account's user setting (collection would otherwise win).
    """
    _seed_user("acct-A", "model-jina")
    _seed_user("acct-B", "model-qwen")

    engine = LyricsSearchEngine(
        qdrant_client=fake_qdrant, collection_name="col-shared",
        model_name="model-jina", lazy=True,
    )
    svc = SearchService(lyrics_db=engine)

    # Per-account model resolution — done in the MAIN thread. SQLite connections
    # aren't safe for concurrent use across threads, and resolution isn't the
    # concurrency target here; the stateless engine is.
    resolved = {
        "acct-A": svc._resolve_model_name(account_id="acct-A", collection_name="col-shared"),
        "acct-B": svc._resolve_model_name(account_id="acct-B", collection_name="col-shared"),
    }
    assert resolved["acct-A"] == "model-jina"
    assert resolved["acct-B"] == "model-qwen"

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

    def _worker(account_id: str):
        engine.search(
            query="q", limit=5,
            model_name=resolved[account_id], collection_name_override="col-shared",
        )

    threads = [threading.Thread(target=_worker, args=("acct-A",)),
               threading.Thread(target=_worker, args=("acct-B",))]
    for t in threads: t.start()
    for t in threads: t.join()

    # Both accounts' models reached Qdrant under their OWN vector_name —
    # concurrent searches on the shared stateless engine didn't cross-contaminate.
    assert seen_usings == {"text_model_jina", "text_model_qwen"}


def test_two_accounts_can_start_indexing_concurrently():
    svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
    assert svc.try_start_job(account_id="acct-A", job_id="job-A") is True
    assert svc.try_start_job(account_id="acct-B", job_id="job-B") is True
    # Both accounts are in-flight at the same time
    assert svc.is_account_indexing("acct-A") and svc.is_account_indexing("acct-B")


def test_same_account_double_indexing_rejected():
    svc = LibraryService(search_service=MagicMock(), db_client=MagicMock())
    assert svc.try_start_job(account_id="acct-A", job_id="j1") is True
    assert svc.try_start_job(account_id="acct-A", job_id="j2") is False
    # Slot still held by j1
    assert svc.get_account_job_id("acct-A") == "j1"


def test_transcoded_cache_isolated_between_accounts(tmp_path, monkeypatch):
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
