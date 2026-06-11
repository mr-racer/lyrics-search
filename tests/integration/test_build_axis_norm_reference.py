"""build_axis_norm_reference: Qdrant scroll → axis stats JSON (fake Qdrant + CLAP)."""
from __future__ import annotations

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from scripts.build_axis_norm_reference import build_reference

DIM = 16


class FakeClapModel:
    def get_text_embedding(self, texts, use_tensor=False):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), DIM)).astype(np.float32)


class _Point:
    def __init__(self, vec):
        self.vector = {"clap": vec}


class FakeQdrant:
    """Two-page scroll to exercise the offset loop."""

    def __init__(self, vectors):
        self._pages = [vectors[: len(vectors) // 2], vectors[len(vectors) // 2:]]

    def scroll(self, collection_name, limit, with_payload, with_vectors, offset=None):
        if offset is None:
            return [_Point(v) for v in self._pages[0]], "page2"
        return [_Point(v) for v in self._pages[1]], None


def test_build_reference_stats_shape():
    rng = np.random.default_rng(0)
    vectors = [rng.normal(size=DIM).astype(np.float32).tolist() for _ in range(10)]

    stats = build_reference(FakeQdrant(vectors), "big_lib", clap_model=FakeClapModel())

    assert stats["n"] == 10
    assert stats["source_collection"] == "big_lib"
    assert len(stats["version"]) == 12
    assert set(stats["mean"]) == set(AXIS_NAMES)
    assert set(stats["std"]) == set(AXIS_NAMES)
    assert all(v > 0 for v in stats["std"].values())  # random vectors → nonzero spread


def test_build_reference_too_few_vectors_exits():
    rng = np.random.default_rng(0)
    vectors = [rng.normal(size=DIM).astype(np.float32).tolist()]
    with pytest.raises(SystemExit, match="CLAP vector"):
        build_reference(FakeQdrant(vectors), "tiny", clap_model=FakeClapModel())


def test_points_without_clap_vector_skipped():
    rng = np.random.default_rng(0)
    vectors = [rng.normal(size=DIM).astype(np.float32).tolist() for _ in range(4)]
    fake = FakeQdrant(vectors)
    # первый поинт второй страницы — без clap-вектора
    orig_pages = fake._pages
    fake._pages = [orig_pages[0], orig_pages[1]]

    class _NoClap:
        vector = {}

    real_scroll = fake.scroll

    def scroll_with_gap(collection_name, limit, with_payload, with_vectors, offset=None):
        pts, nxt = real_scroll(collection_name, limit, with_payload, with_vectors, offset)
        if offset is None:
            return pts + [_NoClap()], nxt
        return pts, nxt

    fake.scroll = scroll_with_gap
    stats = build_reference(fake, "lib", clap_model=FakeClapModel())
    assert stats["n"] == 4  # gap point ignored
