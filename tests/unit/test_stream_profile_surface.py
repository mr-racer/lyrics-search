"""Unit tests for long_term_profile() islands/axes and axis_playlist()."""
from datetime import datetime

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, axis_version
from app.resources.metadata_db import MetadataDB
from app.services.stream_service import (
    Anchor,
    axis_playlist,
    long_term_profile,
    merge_anchors,
)

NOW = datetime(2026, 6, 11, 12, 0, 0)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _stats():
    return {
        "version": axis_version(), "n": 200,
        "mean": {a: 0.0 for a in AXIS_NAMES},
        "std": {a: 1.0 for a in AXIS_NAMES},
    }


def _axes(value):
    return {a: value for a in AXIS_NAMES}


class _Point:
    def __init__(self, tid, vector=None, payload=None):
        self.id = tid
        self.vector = {"clap": vector} if vector is not None else {}
        self.payload = payload or {}
        self.score = None


class FakeQdrant:
    def __init__(self, points):
        self.points = {p.id: p for p in points}

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self.points[t] for t in ids if t in self.points]

    def scroll(self, collection_name, limit, with_payload=True, with_vectors=False, offset=None):
        return list(self.points.values()), None


class TestMergeAnchorsMembers:
    def test_members_recorded_on_merge(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        v_close = np.array([0.999, 0.04], dtype=np.float32)
        merged = merge_anchors(
            [Anchor("a", 2.0), Anchor("b", 1.0)], {"a": v, "b": v_close},
        )
        assert len(merged) == 1
        assert merged[0].members == ["a", "b"]

    def test_solo_anchor_members_is_itself(self):
        assert Anchor("x", 1.0).members == ["x"]


class TestLongTermProfile:
    def _seed_history(self, coll="acct_u"):
        MetadataDB.set_axis_norm_stats(coll, _stats())
        MetadataDB.set_reaction("t1", coll, "like")
        MetadataDB.set_reaction("t2", coll, "like")
        for tid in ("t1", "t2", "t3"):
            MetadataDB.record_playback_event(
                session_id="s", collection_name=coll, track_id=tid,
                played_sec=190.0, total_dur=200.0,
            )

    def test_profile_shape(self):
        self._seed_history()
        fake = FakeQdrant([
            _Point("t1", vector=[1.0, 0.0],
                   payload={"title": "T1", "artist": "A", "sonic_axes": _axes(1.0)}),
            _Point("t2", vector=[0.999, 0.04],
                   payload={"title": "T2", "artist": "A", "sonic_axes": _axes(1.0)}),
            _Point("t3", vector=[0.0, 1.0],
                   payload={"title": "T3", "artist": "B", "sonic_axes": _axes(-1.0)}),
        ])
        out = long_term_profile(qdrant_client=fake, collection_name="acct_u", now=NOW)

        assert out["n_signals"] == 5  # 3 events + 2 reactions
        assert out["confidence"] > 0.0
        # t1+t2 merge into one island (cos≈1), t3 separate
        assert len(out["islands"]) == 2
        top = out["islands"][0]
        assert {m["track_id"] for m in top["tracks"]} == {"t1", "t2"}
        # axes: likes dominate at +1z → positive prefs with level labels
        assert out["axes"]["energy"]["z"] > 0.3
        assert out["axes"]["energy"]["level"] in ("high", "very_high")

    def test_empty_history_returns_blank_profile(self):
        fake = FakeQdrant([])
        out = long_term_profile(qdrant_client=fake, collection_name="acct_u", now=NOW)
        assert out["islands"] == []
        assert out["axes"] is None
        assert out["n_signals"] == 0


class TestAxisPlaylist:
    def _qdrant(self):
        return FakeQdrant([
            _Point("calm", payload={"title": "C", "artist": "x", "sonic_axes": _axes(-1.0)}),
            _Point("mid", payload={"title": "M", "artist": "x", "sonic_axes": _axes(0.0)}),
            _Point("hot", payload={"title": "H", "artist": "x", "sonic_axes": _axes(1.0)}),
            _Point("no_axes", payload={"title": "N", "artist": "x"}),
        ])

    def test_ranks_by_target_closeness(self):
        MetadataDB.set_axis_norm_stats("col", _stats())
        out = axis_playlist(
            qdrant_client=self._qdrant(), collection_name="col",
            axis_targets=_axes(-1.0), limit=3,
        )
        assert [c.track_id for c in out["tracks"]] == ["calm", "mid", "hot"]
        assert out["diagnostics"]["skipped_no_axes"] == 1
        assert out["tracks"][0].pool == "axis"

    def test_dislikes_filtered(self):
        MetadataDB.set_axis_norm_stats("col", _stats())
        MetadataDB.set_reaction("calm", "col", "dislike")
        out = axis_playlist(
            qdrant_client=self._qdrant(), collection_name="col",
            axis_targets=_axes(-1.0), limit=3,
        )
        assert "calm" not in [c.track_id for c in out["tracks"]]

    def test_no_axis_stats_returns_empty_with_reason(self):
        out = axis_playlist(
            qdrant_client=self._qdrant(), collection_name="col",
            axis_targets=_axes(0.0), limit=3,
        )
        assert out["tracks"] == []
        assert out["diagnostics"]["reason"] == "no_axis_stats"

    def test_novelty_breaks_ties(self):
        MetadataDB.set_axis_norm_stats("col", _stats())
        # 'mid' played 10 times; a tie on axis match must prefer the unplayed one
        for _ in range(10):
            MetadataDB.record_playback_event(
                session_id="s", collection_name="col", track_id="mid",
                played_sec=190.0, total_dur=200.0,
            )
        fake = FakeQdrant([
            _Point("mid", payload={"title": "M", "artist": "x", "sonic_axes": _axes(0.0)}),
            _Point("fresh", payload={"title": "F", "artist": "x", "sonic_axes": _axes(0.0)}),
        ])
        out = axis_playlist(qdrant_client=fake, collection_name="col",
                            axis_targets=_axes(0.0), limit=2)
        assert [c.track_id for c in out["tracks"]] == ["fresh", "mid"]
