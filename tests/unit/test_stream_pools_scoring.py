"""Unit tests for candidate pools, scoring, and chunk assembly (design §5–6)."""
import random

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES
from app.services.stream_service import (
    AXIS_MATCH_DIST_SCALE,
    Anchor,
    LIKED_COOLDOWN_H,
    SCORE_W_ANCHOR,
    SCORE_W_NOVELTY,
    StreamCandidate,
    assemble_chunk,
    axis_match_score,
    pool_anchor_candidates,
    sample_liked_tracks,
    score_candidates,
    stratify_explore,
    z_scores_for_axes,
)

RNG = lambda: random.Random(42)  # noqa: E731

STATS = {
    "mean": {a: 0.0 for a in AXIS_NAMES},
    "std": {a: 1.0 for a in AXIS_NAMES},
}


def _cand(tid, artist="x", pool="anchor", cos=0.0, axes=None):
    payload = {"artist": artist, "title": tid}
    if axes is not None:
        payload["sonic_axes"] = axes
    return StreamCandidate(track_id=tid, payload=payload, pool=pool, max_anchor_cos=cos)


class TestZScores:
    def test_basic_z(self):
        raw = {a: 2.0 for a in AXIS_NAMES}
        z = z_scores_for_axes(raw, STATS, AXIS_NAMES)
        assert z["energy"] == pytest.approx(2.0)

    def test_zero_std_yields_zero_not_inf(self):
        stats = {"mean": {a: 0.0 for a in AXIS_NAMES}, "std": {a: 0.0 for a in AXIS_NAMES}}
        z = z_scores_for_axes({a: 5.0 for a in AXIS_NAMES}, stats, AXIS_NAMES)
        assert z["energy"] == 0.0

    def test_missing_axis_in_payload_unusable(self):
        raw = {"energy": 1.0}  # other axes absent
        assert z_scores_for_axes(raw, STATS, AXIS_NAMES) is None

    def test_none_inputs(self):
        assert z_scores_for_axes(None, STATS, AXIS_NAMES) is None
        assert z_scores_for_axes({a: 0.0 for a in AXIS_NAMES}, None, AXIS_NAMES) is None


class TestAxisMatch:
    def test_perfect_match_full_confidence(self):
        z = {a: 1.0 for a in AXIS_NAMES}
        assert axis_match_score(z, dict(z), 1.0, AXIS_NAMES) == pytest.approx(1.0)

    def test_distance_at_scale_gives_zero(self):
        z = {a: 0.0 for a in AXIS_NAMES}
        p = {a: AXIS_MATCH_DIST_SCALE for a in AXIS_NAMES}  # RMS dist == SCALE
        assert axis_match_score(z, p, 1.0, AXIS_NAMES) == pytest.approx(0.0)

    def test_confidence_scales_result(self):
        z = {a: 1.0 for a in AXIS_NAMES}
        assert axis_match_score(z, dict(z), 0.5, AXIS_NAMES) == pytest.approx(0.5)

    def test_missing_profile_or_axes_returns_zero(self):
        z = {a: 1.0 for a in AXIS_NAMES}
        assert axis_match_score(None, z, 1.0, AXIS_NAMES) == 0.0
        assert axis_match_score(z, None, 1.0, AXIS_NAMES) == 0.0
        assert axis_match_score(z, dict(z), 0.0, AXIS_NAMES) == 0.0


class _Hit:
    def __init__(self, tid, score, payload=None):
        self.id = tid
        self.score = score
        self.payload = payload or {"title": tid, "artist": "a"}


class _FakeQdrantSearch:
    """search() returns canned hits per anchor track_id (keyed by vector[0])."""

    def __init__(self, hits_by_marker):
        self.hits_by_marker = hits_by_marker

    def search(self, collection_name, query_vector, limit, with_payload):
        marker = query_vector[1][0]  # first vector component identifies the anchor
        return self.hits_by_marker[marker]


class TestPoolAnchor:
    def test_dedup_keeps_best_cosine_and_its_anchor(self):
        anchors = [
            Anchor("a1", 3.0, vector=[1.0, 0.0]),
            Anchor("a2", 2.0, vector=[2.0, 0.0]),
        ]
        fake = _FakeQdrantSearch({
            1.0: [_Hit("t1", 0.7), _Hit("t2", 0.9)],
            2.0: [_Hit("t1", 0.95)],
        })
        out = pool_anchor_candidates(fake, "col", anchors, excluded=set())
        assert out["t1"].max_anchor_cos == pytest.approx(0.95)
        assert out["t1"].anchor_track_id == "a2"
        assert out["t2"].anchor_track_id == "a1"

    def test_excluded_filtered(self):
        anchors = [Anchor("a1", 1.0, vector=[1.0, 0.0])]
        fake = _FakeQdrantSearch({1.0: [_Hit("bad", 0.99), _Hit("ok", 0.5)]})
        out = pool_anchor_candidates(fake, "col", anchors, excluded={"bad"})
        assert set(out) == {"ok"}

    def test_anchor_without_vector_skipped(self):
        out = pool_anchor_candidates(object(), "col", [Anchor("a", 1.0)], excluded=set())
        assert out == {}


class TestStratifyExplore:
    def test_round_robin_across_bins(self):
        def z(e, x):
            d = {a: 0.0 for a in AXIS_NAMES}
            d["energy"], d["experimental"] = e, x
            return d
        eligible = (
            [(f"low{i}", z(-2.0, 0.0)) for i in range(5)]
            + [(f"high{i}", z(2.0, 0.0)) for i in range(5)]
        )
        out = stratify_explore(eligible, pool_size=4, rng=RNG())
        lows = sum(1 for t in out if t.startswith("low"))
        highs = sum(1 for t in out if t.startswith("high"))
        assert lows == 2 and highs == 2  # perfectly balanced across 2 bins

    def test_tracks_without_axes_still_sampled(self):
        eligible = [(f"t{i}", None) for i in range(3)]
        assert len(stratify_explore(eligible, pool_size=2, rng=RNG())) == 2

    def test_drains_when_pool_smaller_than_request(self):
        eligible = [("only", None)]
        assert stratify_explore(eligible, pool_size=10, rng=RNG()) == ["only"]


class TestSampleLiked:
    def test_cooldown_respected_when_supply_allows(self):
        weights = {"fresh": 1.0, "cooling": 1.0}
        recency = {"cooling": 1.0}  # played 1h ago < 8h cooldown
        out = sample_liked_tracks(weights, recency, 1, RNG())
        assert out == ["fresh"]

    def test_topup_relaxes_cooldown(self):
        """Slider 100% + tiny liked list: rotation beats under-filling."""
        weights = {"a": 1.0, "b": 1.0}
        recency = {"a": 1.0, "b": 2.0}  # both inside cooldown
        out = sample_liked_tracks(weights, recency, 2, RNG())
        assert sorted(out) == ["a", "b"]

    def test_no_duplicates(self):
        weights = {f"t{i}": 1.0 for i in range(5)}
        out = sample_liked_tracks(weights, {}, 5, RNG())
        assert len(out) == len(set(out)) == 5

    def test_excluded_never_sampled(self):
        out = sample_liked_tracks({"a": 1.0, "b": 1.0}, {}, 2, RNG(), excluded={"a"})
        assert out == ["b"]

    def test_never_played_preferred_over_just_played(self):
        """Anti-repeat: a liked track played minutes ago is heavily demoted."""
        weights = {"just_played": 1.0, "never": 1.0}
        recency = {"just_played": 0.1}
        wins = sum(
            1 for i in range(50)
            if sample_liked_tracks(weights, recency, 1, random.Random(i),
                                   cooldown_h=0.0)[0] == "never"
        )
        assert wins > 35  # ~10:1 odds with 0.9 anti-repeat factor


class TestScoreCandidates:
    def test_anchor_cos_dominates(self):
        близкий = _cand("near", cos=0.9)
        далёкий = _cand("far", cos=0.1)
        score_candidates(
            [близкий, далёкий], p_final=None, confidence=0.0,
            play_counts={}, recency_hours={}, axis_stats=None, axis_names=AXIS_NAMES,
        )
        assert близкий.score > далёкий.score
        assert близкий.score == pytest.approx(SCORE_W_ANCHOR * 0.9 + SCORE_W_NOVELTY)

    def test_recent_play_penalised(self):
        a = _cand("recent", cos=0.5)
        b = _cand("stale", cos=0.5)
        score_candidates(
            [a, b], p_final=None, confidence=0.0,
            play_counts={}, recency_hours={"recent": 0.5},
            axis_stats=None, axis_names=AXIS_NAMES,
        )
        assert a.score < b.score

    def test_novelty_boosts_unplayed(self):
        a = _cand("unplayed", cos=0.5)
        b = _cand("worn", cos=0.5)
        score_candidates(
            [a, b], p_final=None, confidence=0.0,
            play_counts={"worn": 20}, recency_hours={},
            axis_stats=None, axis_names=AXIS_NAMES,
        )
        assert a.score > b.score

    def test_axis_match_term_applied(self):
        axes_match = {a: 1.0 for a in AXIS_NAMES}
        axes_off = {a: -3.0 for a in AXIS_NAMES}
        p = {a: 1.0 for a in AXIS_NAMES}
        близкий = _cand("fit", cos=0.5, axes=axes_match)
        чужой = _cand("misfit", cos=0.5, axes=axes_off)
        score_candidates(
            [близкий, чужой], p_final=p, confidence=1.0,
            play_counts={}, recency_hours={}, axis_stats=STATS, axis_names=AXIS_NAMES,
        )
        assert близкий.score > чужой.score
        assert близкий.axis_match == pytest.approx(1.0)


class TestAssembleChunk:
    def test_liked_quota_honored(self):
        main = [_cand(f"m{i}", artist=f"art{i}", cos=0.5) for i in range(5)]
        liked = [_cand(f"l{i}", artist=f"lart{i}", pool="liked") for i in range(5)]
        out = assemble_chunk(main, liked, n=3, liked_share=0.3, recent_artists=[])
        assert sum(1 for c in out if c.pool == "liked") == 1  # round(3·0.3) = 1
        assert len(out) == 3

    def test_slider_zero_no_liked(self):
        main = [_cand(f"m{i}", artist=f"a{i}", cos=0.5) for i in range(5)]
        liked = [_cand("l0", pool="liked")]
        out = assemble_chunk(main, liked, n=3, liked_share=0.0, recent_artists=[])
        assert all(c.pool != "liked" for c in out)

    def test_slider_full_all_liked(self):
        main = [_cand(f"m{i}", artist=f"a{i}", cos=0.9) for i in range(5)]
        liked = [_cand(f"l{i}", artist=f"la{i}", pool="liked") for i in range(5)]
        out = assemble_chunk(main, liked, n=3, liked_share=1.0, recent_artists=[])
        assert all(c.pool == "liked" for c in out)

    def test_main_sorted_by_score(self):
        c_low, c_hi = _cand("low", artist="a1", cos=0.1), _cand("hi", artist="a2", cos=0.9)
        score_candidates([c_low, c_hi], p_final=None, confidence=0.0, play_counts={},
                         recency_hours={}, axis_stats=None, axis_names=AXIS_NAMES)
        out = assemble_chunk([c_low, c_hi], [], n=2, liked_share=0.0, recent_artists=[])
        assert [c.track_id for c in out] == ["hi", "low"]

    def test_no_three_consecutive_same_artist(self):
        """Artist window seeds from session tail: two prior plays by 'X' block a third."""
        main = [_cand("x1", artist="X", cos=0.99), _cand("y1", artist="Y", cos=0.5)]
        out = assemble_chunk(main, [], n=2, liked_share=0.0,
                             recent_artists=["x", "x"])
        assert out[0].track_id == "y1"   # X demoted despite higher score

    def test_artist_rule_bends_when_no_alternative(self):
        main = [_cand(f"x{i}", artist="X", cos=0.9) for i in range(3)]
        out = assemble_chunk(main, [], n=3, liked_share=0.0, recent_artists=["x", "x"])
        assert len(out) == 3  # undersupply is worse — rule bent via topup pass

    def test_liked_topup_when_main_dry(self):
        liked = [_cand(f"l{i}", artist=f"a{i}", pool="liked") for i in range(5)]
        out = assemble_chunk([], liked, n=3, liked_share=0.3, recent_artists=[])
        assert len(out) == 3
        assert all(c.pool == "liked" for c in out)

    def test_both_dry_returns_short_chunk(self):
        out = assemble_chunk([], [], n=3, liked_share=0.5, recent_artists=[])
        assert out == []
