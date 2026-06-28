"""Indexing pipeline integration tests (consolidated).

Covers three concerns, one class per former source file:
- TestDescriptorHook  — indexing triggers sonic-tag/class computation per track
  (from test_indexing_descriptor_hook.py).
- TestComputeSonicAxes / TestUpsertAttachesSonicAxes / TestPersistAxisNormStats —
  Stage 4 sonic axes: payload attachment + axis_norm_stats persistence, with a
  faked CLAP model and a recorder Qdrant stub (from test_indexing_sonic_axes.py).
- TestSonicPayload    — _build_payload_for_upsert pulls sonic_tags from SQLite when
  the row already has them (from test_indexing_sonic_payload.py).
"""
from __future__ import annotations

import sys
import types

# Stub laion_clap before any app import that may pull it in (guards the
# SonicDescriptorService import done inside TestDescriptorHook methods).
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from app.resources.metadata_db import MetadataDB
from app.services.indexing_service import IndexingService

N_PROMPTS = len(AXIS_PROMPTS)
DIM = 16


# --------------------------------------------------------------------------- #
# Fixtures / helpers (hoisted from sources)
# --------------------------------------------------------------------------- #
@pytest.fixture
def _isolated_db(tmp_path, monkeypatch):
    """Isolated MetadataDB on tmp paths. Applied per-class via usefixtures so it
    keeps its original (sonic-axes-only) scope rather than leaking module-wide."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
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


# --------------------------------------------------------------------------- #
# From test_indexing_descriptor_hook.py
# --------------------------------------------------------------------------- #
class TestDescriptorHook:
    """Verify that indexing triggers tag computation and class prediction per track."""

    def test_index_one_track_persists_descriptor(self, monkeypatch, tmp_path):
        """After indexing a track, MetadataDB should have its sonic_tags_json populated."""
        from app.services.sonic_descriptor_service import SonicDescriptorService

        # Stage a real service against tmp paths
        vocab_path = tmp_path / "prompts.json"
        vocab_path.write_text('{"version":1,"groups":{"e":["punchy","ambient"]}}')
        embs_path = tmp_path / "embs.npy"
        np.save(embs_path, np.eye(2, dtype=np.float32))
        svc = SonicDescriptorService(
            prompt_vocab_path=vocab_path,
            embeddings_path=embs_path,
            top_k_tags=2,
        )

        persisted: list[tuple] = []
        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.upsert_sonic_descriptor",
            lambda song_slug, tags=None, sonic_class=None, confidence=None, audio_signature=None: persisted.append((song_slug, tags, sonic_class)),
        )

        # Direct call to the hook — index_track_descriptor is what the indexing pipeline will invoke
        audio_vec = np.array([0.9, 0.1], dtype=np.float32)
        svc.index_track_descriptor(collection="my_col", slug="abc", audio_vector=audio_vec)

        assert len(persisted) == 1
        slug, tags, sclass = persisted[0]
        assert slug == "abc"
        assert tags[0]["tag"] == "punchy"  # closest to (0.9, 0.1)
        assert sclass is None  # no classifier trained → class is None

    def test_index_track_descriptor_persists_when_song_row_absent(self, monkeypatch, tmp_path):
        """Hook should ensure the songs row exists before persisting descriptors."""
        from app.services.sonic_descriptor_service import SonicDescriptorService
        from app.resources.metadata_db import MetadataDB

        # Patch DB to tmp
        db_path = tmp_path / "metadata_test.db"
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_path)
        monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
        MetadataDB._instance = None
        MetadataDB.init()

        vocab_path = tmp_path / "prompts.json"
        vocab_path.write_text('{"version":1,"groups":{"e":["punchy","ambient"]}}')
        embs_path = tmp_path / "embs.npy"
        np.save(embs_path, np.eye(2, dtype=np.float32))
        svc = SonicDescriptorService(
            prompt_vocab_path=vocab_path,
            embeddings_path=embs_path,
            top_k_tags=2,
        )

        # Simulate the bulk hook's per-track invocation: ensure_song first, then index
        MetadataDB.ensure_song(artist="Test Artist", title="Sample Song", collection_name="col1")
        audio_vec = np.array([0.9, 0.1], dtype=np.float32)
        svc.index_track_descriptor(
            collection="col1",
            slug="test-artist-sample-song",
            audio_vector=audio_vec,
        )

        desc = MetadataDB.get_sonic_descriptor("test-artist-sample-song")
        assert desc is not None
        assert len(desc["tags"]) == 2
        assert desc["tags"][0]["tag"] == "punchy"
        MetadataDB._instance = None


# --------------------------------------------------------------------------- #
# From test_indexing_sonic_axes.py (already class-based; scope _isolated_db here)
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_isolated_db")
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


@pytest.mark.usefixtures("_isolated_db")
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


@pytest.mark.usefixtures("_isolated_db")
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


# --------------------------------------------------------------------------- #
# From test_indexing_sonic_payload.py
# --------------------------------------------------------------------------- #
class TestSonicPayload:
    """Indexing payload includes sonic_tags when the SQLite row already has them
    (e.g. re-indexing a curated library)."""

    def test_payload_includes_sonic_tags_when_present(self, db):
        # slug = _slugify("Artist B") + "-" + _slugify("Title Y")
        #       = "artist-b" + "-" + "title-y" = "artist-b-title-y"
        song_slug = "artist-b-title-y"
        db.upsert_artist("artist-b", "Artist B", "test_col")
        db.upsert_song(song_slug, "Title Y", "artist-b", "test_col")
        db.upsert_sonic_descriptor(
            song_slug=song_slug,
            tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
        )

        from app.services.indexing_service import _build_payload_for_upsert
        song_info = {"title": "Title Y", "artist": "Artist B", "album": None,
                     "year": 2019, "year_range": "2015-2019", "genre": None,
                     "duration": 180, "file_path": "/p", "cover_art_path": None,
                     "producer": None, "label": None, "samples": None, "sampled_by": None,
                     "bitrate_kbps": None, "lyrics": ""}
        payload = _build_payload_for_upsert(song_info, slug=song_slug)
        assert sorted(payload["sonic_tags"]) == ["lo-fi", "melancholic"]

    def test_payload_omits_sonic_when_sqlite_empty(self, db):
        """Track whose SQLite row exists but has no sonic descriptor: payload has empty list."""
        from app.services.indexing_service import _build_payload_for_upsert
        song_info = {"title": "Z", "artist": "C", "album": None, "year": None,
                     "year_range": None, "genre": None, "duration": None,
                     "file_path": "/p", "cover_art_path": None, "producer": None,
                     "label": None, "samples": None, "sampled_by": None,
                     "bitrate_kbps": None, "lyrics": ""}
        payload = _build_payload_for_upsert(song_info, slug="unknown-z")
        assert payload["sonic_tags"] == []

    def test_payload_tolerates_no_slug(self, db):
        """When the caller can't derive a slug, payload stays clean (empty tags)."""
        from app.services.indexing_service import _build_payload_for_upsert
        song_info = {"title": "Z", "artist": "C", "album": None, "year": None,
                     "year_range": None, "genre": None, "duration": None,
                     "file_path": "/p", "cover_art_path": None, "producer": None,
                     "label": None, "samples": None, "sampled_by": None,
                     "bitrate_kbps": None, "lyrics": ""}
        payload = _build_payload_for_upsert(song_info, slug=None)
        assert payload["sonic_tags"] == []
