"""The engine must hold no per-search model state.

This file used to guard a race: two accounts searching with two different
embedding models could clobber each other's ``engine.model_name``. There is one
model app-wide now, so that race is structurally impossible — what is still
worth pinning down is that the engine keeps nothing per-call on ``self`` and
always queries the one vector name the collections are built with.
"""

import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.resources.lyrics_search_engine import LyricsSearchEngine
from app.resources.model_registry import ModelRegistry


class _FakeModel:
    """Records every encode() call so the test can inspect what was sent."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list = []

    def encode(self, query, **kw):
        self.calls.append(query)
        # numpy array so search()'s .tolist() works, like real sentence-transformers.
        return np.zeros(self.dim, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


@pytest.fixture
def fake_model():
    model = _FakeModel()
    previous = ModelRegistry._text_model
    ModelRegistry._text_model = (model, ModelRegistry.VECTOR_NAME, model.dim)
    yield model
    ModelRegistry._text_model = previous


def _make_qdrant() -> Any:
    q = MagicMock()
    q.get_collections.return_value = MagicMock(collections=[])
    q.query_points.return_value = MagicMock(points=[])
    return q


def _engine(qdrant) -> LyricsSearchEngine:
    return LyricsSearchEngine(qdrant_client=qdrant, collection_name="col")


def test_search_queries_the_pinned_vector_name(fake_model):
    qdrant = _make_qdrant()
    _engine(qdrant).search(query="hello", limit=3)

    prefetch = qdrant.query_points.call_args.kwargs["prefetch"]
    dense = next(p for p in prefetch if p.using != "bm25")
    assert dense.using == ModelRegistry.VECTOR_NAME == "text"


def test_search_encodes_the_query_side_with_the_instruction_prefix(fake_model):
    """Qwen3-Embedding is asymmetric; the query side takes the prefix and the
    indexed documents do not."""
    _engine(_make_qdrant()).search(query="who produced this", limit=1)

    assert fake_model.calls == [ModelRegistry.QUERY_PREFIX + "who produced this"]


def test_concurrent_searches_leave_no_state_on_the_engine(fake_model):
    qdrant = _make_qdrant()
    engine = _engine(qdrant)
    before = dict(engine.__dict__)
    errors: list[BaseException] = []

    def _worker(q: str):
        try:
            engine.search(query=q, limit=1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(f"q{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert engine.__dict__ == before
    assert qdrant.query_points.call_count == 8


def test_vector_metadata_comes_from_the_registry(fake_model):
    engine = _engine(_make_qdrant())
    assert engine.vector_name == ModelRegistry.VECTOR_NAME
    assert engine.vector_dim == ModelRegistry.VECTOR_DIM
