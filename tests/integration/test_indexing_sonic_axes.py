"""Indexing Stage 4 sonic axes: payload attachment + axis_norm_stats persistence.

CLAP model is faked (get_text_embedding returns a fixed matrix), Qdrant is a
recorder stub — no live services needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from app.resources.metadata_db import MetadataDB
from app.services.indexing_service import IndexingService

N_PROMPTS = len(AXIS_PROMPTS)
DIM = 16


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


class FakeClapModel:
    """get_text_embedding → deterministic (N_PROMPTS, DIM) matrix."""

    def get_text_embedding(self, texts, use_tensor=False):
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(texts), DIM)).astype(np.float32)


class FakeQdrant:
    def __init__(self):
        self.upserted = []

    def upsert(self, collection_name, points):
        self.upserted.extend(points)


class FakeEngine:
    def __init__(self):
        self.qdrant_client = FakeQdrant()
        self.collection_name = "test_col"
        self.vector_name = "text"
        self.vector_dim = 8
        self.model_clap = FakeClapModel()


def _song(artist: str, title: str) -> dict:
    return {
        "title": title, "artist": artist, "album": None, "year": None,
        "year_range": None, "genre": None, "duration": 100,
        "duration_range": None, "file_path": "/p", "cover_art_path": None,
        "producer": None, "label": None, "samples": None, "sampled_by": None,
        "bitrate_kbps": None, "lyrics": "la la la",
    }


def _clap_map(*keys):
    rng = np.random.default_rng(7)
    return {k: rng.normal(size=DIM).astype(np.float32) for k in keys}


class TestComputeSonicAxes:
    def test_returns_axis_dict_per_track(self):
        svc = IndexingService(FakeEngine())
        cmap = _clap_map(("artist a", "one"), ("artist b", "two"))

        axes_map = svc._compute_sonic_axes(cmap)

        assert set(axes_map) == set(cmap)
        for d in axes_map.values():
            assert tuple(d) == AXIS_NAMES
            assert all(isinstance(v, float) for v in d.values())

    def test_empty_clap_map_short_circuits(self):
        assert IndexingService(FakeEngine())._compute_sonic_axes({}) == {}

    def test_model_failure_returns_empty_not_raises(self):
        engine = FakeEngine()
        engine.model_clap = object()  # no get_text_embedding → AttributeError inside
        svc = IndexingService(engine)
        assert svc._compute_sonic_axes(_clap_map(("a", "b"))) == {}


class TestUpsertAttachesSonicAxes:
    def test_payload_gets_sonic_axes_for_matched_key(self):
        engine = FakeEngine()
        svc = IndexingService(engine)
        songs = [_song("Artist A", "One"), _song("Artist B", "Two")]
        text_vecs = np.zeros((2, 8), dtype=np.float32)
        axes = {a: 0.1 for a in AXIS_NAMES}
        sonic_axes_map = {("artist a", "one"): axes}  # only the first song

        svc._upsert_in_batches(songs, text_vecs, sonic_axes_map=sonic_axes_map)

        payloads = {p.payload["title"]: p.payload for p in engine.qdrant_client.upserted}
        assert payloads["One"]["sonic_axes"] == axes
        assert "sonic_axes" not in payloads["Two"]

    def test_no_axes_map_keeps_payload_clean(self):
        engine = FakeEngine()
        svc = IndexingService(engine)
        svc._upsert_in_batches([_song("A", "X")], np.zeros((1, 8), dtype=np.float32))
        assert "sonic_axes" not in engine.qdrant_client.upserted[0].payload


class TestPersistAxisNormStats:
    def test_writes_mean_std_n_version(self):
        svc = IndexingService(FakeEngine())
        sonic_axes_map = {
            ("a", "1"): {a: 1.0 for a in AXIS_NAMES},
            ("a", "2"): {a: 2.0 for a in AXIS_NAMES},
            ("a", "3"): {a: 3.0 for a in AXIS_NAMES},
        }

        svc._persist_axis_norm_stats(sonic_axes_map)

        stats = MetadataDB.get_axis_norm_stats("test_col")
        assert stats["n"] == 3
        assert stats["mean"]["energy"] == pytest.approx(2.0)
        assert stats["std"]["energy"] == pytest.approx(1.0)  # ddof=1: std([1,2,3])
        assert len(stats["version"]) == 12
        assert set(stats["mean"]) == set(AXIS_NAMES)

    def test_single_track_std_is_zero_not_nan(self):
        svc = IndexingService(FakeEngine())
        svc._persist_axis_norm_stats({("a", "1"): {a: 0.5 for a in AXIS_NAMES}})
        stats = MetadataDB.get_axis_norm_stats("test_col")
        assert stats["n"] == 1
        assert stats["std"]["energy"] == 0.0

    def test_empty_map_writes_nothing(self):
        svc = IndexingService(FakeEngine())
        svc._persist_axis_norm_stats({})
        assert MetadataDB.get_axis_norm_stats("test_col") is None
