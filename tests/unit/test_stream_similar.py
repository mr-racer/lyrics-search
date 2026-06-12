"""Unit tests for similar_tracks(): CLAP neighbors re-ranked by axis closeness."""
import pytest

from app.resources.clap_features import AXIS_NAMES
from app.resources.metadata_db import MetadataDB
from app.services.stream_service import (
    SIMILAR_W_AXES,
    SIMILAR_W_CLAP,
    similar_tracks,
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


STATS = {
    "version": None,  # filled by fixture below
    "n": 200,
    "mean": {a: 0.0 for a in AXIS_NAMES},
    "std": {a: 1.0 for a in AXIS_NAMES},
}


@pytest.fixture
def axis_stats_in_db():
    from app.resources.clap_features import axis_version
    MetadataDB.set_axis_norm_stats("col", {**STATS, "version": axis_version()})


class _Point:
    def __init__(self, tid, vector=None, payload=None, score=None):
        self.id = tid
        self.vector = {"clap": vector} if vector is not None else {}
        self.payload = payload or {}
        self.score = score


def _axes(value):
    return {a: value for a in AXIS_NAMES}


class FakeQdrant:
    """retrieve() returns the seed; query_points() returns canned hits
    (modern qdrant-client interface — legacy .search() no longer exists)."""

    def __init__(self, seed_point, hits):
        self.seed_point = seed_point
        self.hits = hits

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self.seed_point] if self.seed_point and self.seed_point.id in ids else []

    def query_points(self, collection_name, query, using, limit, with_payload=True):
        from types import SimpleNamespace
        return SimpleNamespace(points=self.hits[:limit])


def _seed(axes_value=1.0):
    return _Point("seed", vector=[1.0, 0.0], payload={"sonic_axes": _axes(axes_value)})


def test_axis_closeness_rescores_equal_cosines(axis_stats_in_db):
    """Same CLAP cosine — the candidate whose axes match the seed wins."""
    hits = [
        _Point("far_axes", score=0.9, payload={"sonic_axes": _axes(-2.0)}),
        _Point("near_axes", score=0.9, payload={"sonic_axes": _axes(1.0)}),
    ]
    out = similar_tracks(
        qdrant_client=FakeQdrant(_seed(axes_value=1.0), hits),
        collection_name="col", seed_track_id="seed",
    )
    ids = [c.track_id for c in out["tracks"]]
    assert ids == ["near_axes", "far_axes"]
    near = out["tracks"][0]
    assert near.axis_match == pytest.approx(1.0)
    assert near.score == pytest.approx(SIMILAR_W_CLAP * 0.9 + SIMILAR_W_AXES * 1.0)


def test_no_axis_stats_degrades_to_pure_cosine():
    """Without usable stats the axis term is 0 — order = CLAP order."""
    hits = [
        _Point("second", score=0.8, payload={"sonic_axes": _axes(1.0)}),
        _Point("first", score=0.95, payload={"sonic_axes": _axes(-3.0)}),
    ]
    out = similar_tracks(
        qdrant_client=FakeQdrant(_seed(), hits),
        collection_name="col", seed_track_id="seed",
    )
    assert [c.track_id for c in out["tracks"]] == ["first", "second"]
    assert all(c.axis_match == 0.0 for c in out["tracks"])


def test_seed_itself_and_dislikes_filtered(axis_stats_in_db):
    MetadataDB.set_reaction("hated", "col", "dislike")
    hits = [
        _Point("seed", score=1.0, payload={"sonic_axes": _axes(1.0)}),
        _Point("hated", score=0.99, payload={"sonic_axes": _axes(1.0)}),
        _Point("ok", score=0.5, payload={"sonic_axes": _axes(1.0)}),
    ]
    out = similar_tracks(
        qdrant_client=FakeQdrant(_seed(), hits),
        collection_name="col", seed_track_id="seed",
    )
    assert [c.track_id for c in out["tracks"]] == ["ok"]


def test_exclude_ids_respected(axis_stats_in_db):
    hits = [_Point("a", score=0.9, payload={}), _Point("b", score=0.8, payload={})]
    out = similar_tracks(
        qdrant_client=FakeQdrant(_seed(), hits),
        collection_name="col", seed_track_id="seed", exclude_ids=["a"],
    )
    assert [c.track_id for c in out["tracks"]] == ["b"]


def test_seed_without_clap_vector_returns_empty():
    seed = _Point("seed", vector=None, payload={})
    out = similar_tracks(
        qdrant_client=FakeQdrant(seed, []),
        collection_name="col", seed_track_id="seed",
    )
    assert out["tracks"] == []


def test_limit_applied_after_rerank(axis_stats_in_db):
    hits = [_Point(f"t{i}", score=0.9 - i * 0.01, payload={"sonic_axes": _axes(1.0)})
            for i in range(20)]
    out = similar_tracks(
        qdrant_client=FakeQdrant(_seed(), hits),
        collection_name="col", seed_track_id="seed", limit=5,
    )
    assert len(out["tracks"]) == 5
