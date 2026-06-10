"""Unit tests for taste profiles: aggregation, anchors, merging, blending (§4-5)."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES
from app.services.stream_service import (
    ANCHOR_MERGE_THRESHOLD,
    Anchor,
    H_IMPLICIT_DAYS,
    PlaybackSignal,
    ReactionSignal,
    SESSION_ANCHOR_BOOST,
    W_FULL,
    aggregate_event_weights,
    aggregate_reaction_weights,
    axis_preferences,
    blend_axis_preferences,
    combine_weights,
    count_session_signals,
    merge_anchors,
    negative_track_ids,
    select_positive_anchors,
    session_blend_weight,
    union_anchor_weights,
)

NOW = datetime(2026, 6, 10, 12, 0, 0)


def _ev(track, played=230.0, dur=240.0, days_ago=0.0, session="s1", interacted=None):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=NOW - timedelta(days=days_ago), session_id=session,
        interacted=interacted,
    )


def _like(track, days_ago=0.0):
    return ReactionSignal(track_id=track, reaction="like",
                          updated_at=NOW - timedelta(days=days_ago))


def _dislike(track, days_ago=0.0):
    return ReactionSignal(track_id=track, reaction="dislike",
                          updated_at=NOW - timedelta(days=days_ago))


class TestAggregation:
    def test_fresh_full_listen_weight(self):
        w = aggregate_event_weights([_ev("a")], NOW)
        assert w["a"] == pytest.approx(W_FULL)

    def test_old_event_decays(self):
        w = aggregate_event_weights([_ev("a", days_ago=H_IMPLICIT_DAYS)], NOW)
        assert w["a"] == pytest.approx(W_FULL / 2.718281828, rel=1e-6)

    def test_sessions_weighted_independently(self):
        """Replay detection must not fire across different sessions."""
        events = [_ev("a", session="s1"), _ev("a", session="s2")]
        w = aggregate_event_weights(events, NOW)
        assert w["a"] == pytest.approx(2 * W_FULL)  # two full listens, no replay boost

    def test_reactions_decay_with_h_like(self):
        w = aggregate_reaction_weights([_like("a", days_ago=90.0)], NOW)
        assert w["a"] == pytest.approx(1.0 / 2.718281828, rel=1e-6)

    def test_combine_sums_across_sources(self):
        total = combine_weights({"a": 0.4, "b": -0.6}, {"a": 1.0})
        assert total == {"a": 1.4, "b": -0.6}


class TestNegativeSet:
    def test_dislike_is_hard_negative_regardless_of_age(self):
        neg = negative_track_ids([], [_dislike("d", days_ago=500.0)], NOW)
        assert "d" in neg

    def test_two_fresh_skips_cross_threshold(self):
        events = [_ev("s", played=5.0, days_ago=1.0), _ev("s", played=5.0, days_ago=0.5)]
        assert "s" in negative_track_ids(events, [], NOW)

    def test_one_skip_is_not_enough(self):
        assert negative_track_ids([_ev("s", played=5.0)], [], NOW) == set()

    def test_two_ancient_skips_decay_out(self):
        events = [_ev("s", played=5.0, days_ago=60.0), _ev("s", played=5.0, days_ago=50.0)]
        assert "s" not in negative_track_ids(events, [], NOW)


class TestAnchors:
    def test_select_top_m_positive_only(self):
        weights = {"a": 2.0, "b": 0.5, "c": -1.0, "d": 1.0}
        anchors = select_positive_anchors(weights, top_m=2)
        assert [a.track_id for a in anchors] == ["a", "d"]

    def test_merge_collapses_near_duplicates(self):
        """Three likes off one album → one anchor with summed weight."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_close = np.array([0.99, 0.14, 0.0], dtype=np.float32)   # cos ≈ 0.99
        v_far = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        anchors = [Anchor("a", 2.0), Anchor("b", 1.5), Anchor("c", 1.0)]
        vectors = {"a": v, "b": v_close, "c": v_far}

        merged = merge_anchors(anchors, vectors)

        by_id = {a.track_id: a for a in merged}
        assert set(by_id) == {"a", "c"}
        assert by_id["a"].weight == pytest.approx(3.5)   # absorbed b
        assert by_id["a"].vector is not None

    def test_merge_keeps_distinct_anchors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)   # cos = 0 < threshold
        merged = merge_anchors([Anchor("a", 1.0), Anchor("b", 1.0)], {"a": v1, "b": v2})
        assert len(merged) == 2

    def test_anchor_without_vector_dropped(self):
        merged = merge_anchors([Anchor("a", 1.0), Anchor("ghost", 5.0)],
                               {"a": np.array([1.0, 0.0], dtype=np.float32)})
        assert [a.track_id for a in merged] == ["a"]

    def test_just_below_threshold_not_merged(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        ang = np.arccos(ANCHOR_MERGE_THRESHOLD - 0.01)
        v2 = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        merged = merge_anchors([Anchor("a", 1.0), Anchor("b", 1.0)], {"a": v1, "b": v2})
        assert len(merged) == 2


class TestSessionBlend:
    @pytest.mark.parametrize("n,expected", [(0, 0.0), (5, 0.5), (10, 1.0), (25, 1.0)])
    def test_w_s_saturates_at_10(self, n, expected):
        assert session_blend_weight(n) == pytest.approx(expected)

    def test_count_ignores_zero_weight_events(self):
        """Neutral-zone listens carry no information — not signals."""
        events = [_ev("a"), _ev("b", played=60.0)]  # full + neutral
        assert count_session_signals(events, []) == 1

    def test_count_includes_reactions(self):
        assert count_session_signals([_ev("a")], [_like("x")]) == 2

    def test_union_boosts_session_anchors(self):
        long_w = {"a": 1.0, "b": 0.4}
        sess_w = {"b": 0.4}
        out = union_anchor_weights(long_w, sess_w, w_s=1.0)
        assert out["a"] == 1.0
        assert out["b"] == pytest.approx(0.4 * (1 + SESSION_ANCHOR_BOOST))  # ×3 at w_s=1

    def test_union_takes_max_not_sum(self):
        """Session events already sit inside long-term history — no double count."""
        out = union_anchor_weights({"a": 5.0}, {"a": 0.4}, w_s=1.0)
        assert out["a"] == 5.0


class TestAxisPreferences:
    def test_weighted_mean_of_positive_tracks(self):
        weights = {"a": 1.0, "b": 3.0, "neg": -5.0}
        z = {
            "a": {ax: 1.0 for ax in AXIS_NAMES},
            "b": {ax: -1.0 for ax in AXIS_NAMES},
            "neg": {ax: 10.0 for ax in AXIS_NAMES},   # must be ignored
        }
        p, conf = axis_preferences(weights, z, AXIS_NAMES)
        assert p["energy"] == pytest.approx((1.0 - 3.0) / 4.0)
        assert 0.0 < conf <= 1.0

    def test_no_positive_signals_returns_none(self):
        p, conf = axis_preferences({"a": -1.0}, {}, AXIS_NAMES)
        assert p is None and conf == 0.0

    def test_tracks_without_axes_skipped(self):
        p, conf = axis_preferences({"a": 1.0}, {}, AXIS_NAMES)
        assert p is None and conf == 0.0

    def test_blend_full_session_dominance(self):
        p_long = {ax: 0.0 for ax in AXIS_NAMES}
        p_sess = {ax: 2.0 for ax in AXIS_NAMES}
        out = blend_axis_preferences(p_long, p_sess, w_s=1.0, axis_names=AXIS_NAMES)
        assert out["energy"] == 2.0

    def test_blend_half(self):
        p_long = {ax: 0.0 for ax in AXIS_NAMES}
        p_sess = {ax: 2.0 for ax in AXIS_NAMES}
        out = blend_axis_preferences(p_long, p_sess, w_s=0.5, axis_names=AXIS_NAMES)
        assert out["energy"] == 1.0

    def test_blend_one_sided(self):
        p_long = {ax: 1.5 for ax in AXIS_NAMES}
        assert blend_axis_preferences(p_long, None, 0.9, AXIS_NAMES)["energy"] == 1.5
        assert blend_axis_preferences(None, p_long, 0.1, AXIS_NAMES)["energy"] == 1.5
        assert blend_axis_preferences(None, None, 0.5, AXIS_NAMES) is None
