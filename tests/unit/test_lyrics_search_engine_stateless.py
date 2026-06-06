"""Engine must be stateless: concurrent search(... model_name=...) calls
must NOT clobber each other's view of which model is in use."""

import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.model_registry import ModelRegistry


class _FakeModel:
    """Records every encode() call so the test can assert per-thread model use."""

    def __init__(self, name: str, dim: int = 4):
        self.name = name
        self.dim = dim

    def encode(self, query: str):
        # numpy array so search()'s .tolist() works, like real sentence-transformers.
        return np.array(
            [hash(self.name) % 1000] + [0.0] * (self.dim - 1), dtype=np.float32
        )

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


import pytest


@pytest.fixture(autouse=True)
def _seed_registry():
    ModelRegistry._text_models.clear()
    if hasattr(ModelRegistry, "_load_locks"):
        ModelRegistry._load_locks.clear()
    a = _FakeModel("model-A")
    b = _FakeModel("model-B")
    ModelRegistry._text_models["model-A"] = (a, "text_model_A", 4)
    ModelRegistry._text_models["model-B"] = (b, "text_model_B", 4)
    yield
    ModelRegistry._text_models.clear()


def _make_qdrant() -> Any:
    q = MagicMock()
    q.get_collections.return_value = MagicMock(collections=[])
    q.query_points.return_value = MagicMock(points=[])
    return q


def test_concurrent_searches_use_correct_per_call_model():
    """Two threads call search() with different model_name args. Each
    Qdrant query must be issued with the vector_name matching that model."""
    engine = LyricsSearchEngine(
        qdrant_client=_make_qdrant(),
        collection_name="col",
        model_name="model-A",   # initial / fallback only
        lazy=True,
    )

    using_per_thread: dict[str, str] = {}
    lock = threading.Lock()
    errors: list[BaseException] = []

    def _worker(model_name: str):
        try:
            engine.search(
                query="q",
                limit=5,
                model_name=model_name,
                collection_name_override="col",
            )
            call = engine.qdrant_client.query_points.call_args_list[-1]
            prefetches = call.kwargs.get("prefetch") or []
            using = prefetches[0].using if prefetches else None
            with lock:
                using_per_thread[model_name] = using
        except BaseException as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [
        threading.Thread(target=_worker, args=("model-A",)),
        threading.Thread(target=_worker, args=("model-B",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # The race-sensitive part is per-call vector resolution; since both threads
    # hit a shared MagicMock query_points, we assert each model_name resolved to
    # its own vector_name (proves search read model_name, not shared self state).
    assert using_per_thread.get("model-A") == "text_model_A"
    assert using_per_thread.get("model-B") == "text_model_B"


def test_search_resolves_per_call_model_not_self_state():
    """A single engine, two sequential searches with different model_name, must
    issue Qdrant queries with the matching vector_name each time."""
    engine = LyricsSearchEngine(
        qdrant_client=_make_qdrant(),
        collection_name="col",
        model_name="model-A",
        lazy=True,
    )
    engine.search(query="q", model_name="model-A", collection_name_override="col")
    using_a = engine.qdrant_client.query_points.call_args_list[-1].kwargs["prefetch"][0].using
    engine.search(query="q", model_name="model-B", collection_name_override="col")
    using_b = engine.qdrant_client.query_points.call_args_list[-1].kwargs["prefetch"][0].using
    assert using_a == "text_model_A"
    assert using_b == "text_model_B"
    # Engine must NOT have cached model-B onto self (stateless dispatch).
    assert engine.model_name == "model-A"
