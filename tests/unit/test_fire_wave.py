"""Unit tests for the Огонёк/Вода/Волна signal layer (app.services.stream_service).

Covers the pure functions added by the 2026-06-29 fire-wave redesign:
  - fire decay curves (time × track-count)
  - ephemeral fire/water anchor aggregation
  - low-learning-rate island feed (fires + ≥85% completions only)
  - computed «favorites» ranking (replaces heart-likes for the slider pool)
  - session explore warmup ramp
  - water penalty term in score_candidates
"""
from datetime import datetime, timedelta

import pytest

from app.services import stream_service as ss
from app.services.stream_service import (
    H_REACTION_DAYS,
    COMPLETION_DEPOSIT,
    EXPLORE_SHARE,
    FIRE_BASE,
    FIRE_COUNT_FULL,
    FIRE_COUNT_ZERO,
    FIRE_ISLAND_DEPOSIT,
    FIRE_TIME_MAX_H,
    H_IMPLICIT_DAYS,
    H_ISLAND_DAYS,
    WARMUP_EXPLORE_SHARE,
    WARMUP_SIGNALS,
    WATER_BASE,
    FireSignal,
    PlaybackSignal,
    aggregate_taste_anchors,
    completion_counts,
    decayed_fire_counts,
    explore_share_for_warmth,
    favorite_weights,
    fire_count_factor,
    fire_time_factor,
    island_taste_weights,
    latest_per_track,
    reaction_contribution,
    score_candidates,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 6, 29, 12, 0, 0)


def _fire(track, kind="fire", hours_ago=0.0):
    return FireSignal(track_id=track, kind=kind,
                      created_at=NOW - timedelta(hours=hours_ago))


def _play(track, played=190.0, dur=200.0, days_ago=0.0, session="s1"):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=NOW - timedelta(days=days_ago), session_id=session,
    )


def _cand(tid, score=0.0):
    return ss.StreamCandidate(track_id=tid, payload={"artist": "x", "title": tid},
                              pool="anchor", score=score)


# ── Fire decay curves ────────────────────────────────────────────────────────

class TestFireTimeFactor:
    def test_full_at_zero(self):
        assert fire_time_factor(0.0) == pytest.approx(1.0)

    def test_half_at_midpoint(self):
        assert fire_time_factor(FIRE_TIME_MAX_H / 2) == pytest.approx(0.5)

    def test_zero_at_max(self):
        assert fire_time_factor(FIRE_TIME_MAX_H) == pytest.approx(0.0)

    def test_zero_beyond_max(self):
        assert fire_time_factor(FIRE_TIME_MAX_H + 1.0) == 0.0

    def test_monotonic_decreasing(self):
        assert fire_time_factor(1.0) > fire_time_factor(3.0) > 0.0


class TestFireCountFactor:
    def test_full_until_threshold(self):
        assert fire_count_factor(0) == 1.0
        assert fire_count_factor(FIRE_COUNT_FULL) == pytest.approx(1.0)

    def test_half_at_midpoint(self):
        midpoint = (FIRE_COUNT_FULL + FIRE_COUNT_ZERO) // 2  # 40
        assert fire_count_factor(midpoint) == pytest.approx(0.5)

    def test_zero_at_and_beyond_end(self):
        assert fire_count_factor(FIRE_COUNT_ZERO) == 0.0
        assert fire_count_factor(FIRE_COUNT_ZERO + 10) == 0.0


# ── Ephemeral fire/water anchors (dual decay applied) ───────────────────────

class TestAggregateTasteAnchors:
    def test_fresh_fire_full_weight(self):
        anchors = aggregate_taste_anchors(
            [_fire("a")], kind="fire", session_play_times=[], now=NOW,
        )
        assert anchors["a"] == pytest.approx(FIRE_BASE)

    def test_time_decay_applied(self):
        anchors = aggregate_taste_anchors(
            [_fire("a", hours_ago=FIRE_TIME_MAX_H / 2)], kind="fire",
            session_play_times=[], now=NOW,
        )
        assert anchors["a"] == pytest.approx(FIRE_BASE * 0.5)

    def test_count_decay_from_session_plays(self):
        # 40 tracks played after the fire → count_factor 0.5
        fire_at = NOW - timedelta(hours=1)
        plays = [fire_at + timedelta(minutes=i + 1) for i in range(40)]
        anchors = aggregate_taste_anchors(
            [FireSignal("a", "fire", fire_at)], kind="fire",
            session_play_times=plays, now=NOW,
        )
        assert anchors["a"] == pytest.approx(FIRE_BASE * fire_time_factor(1.0) * 0.5)

    def test_expired_fire_dropped(self):
        anchors = aggregate_taste_anchors(
            [_fire("a", hours_ago=FIRE_TIME_MAX_H + 1)], kind="fire",
            session_play_times=[], now=NOW,
        )
        assert "a" not in anchors

    def test_multiple_fires_same_track_sum(self):
        anchors = aggregate_taste_anchors(
            [_fire("a"), _fire("a")], kind="fire", session_play_times=[], now=NOW,
        )
        assert anchors["a"] == pytest.approx(2 * FIRE_BASE)

    def test_kind_filter_isolates_water(self):
        signals = [_fire("a", kind="fire"), _fire("b", kind="water")]
        fires = aggregate_taste_anchors(signals, kind="fire",
                                        session_play_times=[], now=NOW)
        waters = aggregate_taste_anchors(signals, kind="water", base=WATER_BASE,
                                         session_play_times=[], now=NOW)
        assert set(fires) == {"a"} and set(waters) == {"b"}


# ── Island feed: low learning rate (fires + ≥85% completions only) ──────────

class TestDecayedFireCounts:
    def test_counts_only_fires_with_decay(self):
        signals = [_fire("a"), _fire("a", hours_ago=24 * H_ISLAND_DAYS),
                   _fire("b", kind="water")]
        out = decayed_fire_counts(signals, NOW, half_life=H_ISLAND_DAYS)
        assert "b" not in out
        assert out["a"] == pytest.approx(1.0 + 1.0 / 2.718281828, rel=1e-6)


class TestCompletionCounts:
    def test_only_85_percent_and_up(self):
        signals = [
            _play("full", played=200.0, dur=200.0),   # 100% → counts
            _play("edge", played=170.0, dur=200.0),    # 85% → counts
            _play("most", played=150.0, dur=200.0),    # 75% → excluded
            _play("skip", played=10.0, dur=200.0),     # skip → excluded
        ]
        out = completion_counts(signals, NOW, half_life=H_ISLAND_DAYS)
        assert set(out) == {"full", "edge"}

    def test_missing_duration_excluded(self):
        out = completion_counts([_play("x", played=500.0, dur=None)], NOW,
                                half_life=H_ISLAND_DAYS)
        assert out == {}


class TestIslandTasteWeights:
    def test_fire_dominates_completion(self):
        fires = [_fire("hot")]
        plays = [_play("warm", played=200.0, dur=200.0)]
        weights = island_taste_weights(fires, plays, NOW)
        assert weights["hot"] == pytest.approx(FIRE_ISLAND_DEPOSIT)
        assert weights["warm"] == pytest.approx(COMPLETION_DEPOSIT)
        assert weights["hot"] > weights["warm"]

    def test_partial_and_skip_excluded(self):
        plays = [_play("most", played=150.0, dur=200.0),
                 _play("skip", played=5.0, dur=200.0)]
        assert island_taste_weights([], plays, NOW) == {}


# ── Favorites ranking (replaces heart-likes for the slider pool) ────────────

class TestFavoriteWeights:
    def test_fire_weighted_above_listens(self):
        fav = favorite_weights({"fired": 1.0}, {"played": 1.0})
        assert fav["fired"] > fav["played"]

    def test_combines_both_signals(self):
        fav = favorite_weights({"t": 2.0}, {"t": 1.0})
        assert fav["t"] == pytest.approx(ss.FAV_FIRE_W * 2.0 + ss.FAV_LISTEN_W * 1.0)

    def test_drops_nonpositive(self):
        assert favorite_weights({}, {}) == {}


# ── Session explore warmup ramp ─────────────────────────────────────────────

class TestExploreWarmup:
    def test_cold_session_high_explore(self):
        assert explore_share_for_warmth(0) == pytest.approx(WARMUP_EXPLORE_SHARE)

    def test_warm_session_baseline(self):
        assert explore_share_for_warmth(WARMUP_SIGNALS) == pytest.approx(EXPLORE_SHARE)
        assert explore_share_for_warmth(WARMUP_SIGNALS * 3) == pytest.approx(EXPLORE_SHARE)

    def test_ramps_between(self):
        mid = explore_share_for_warmth(WARMUP_SIGNALS / 2)
        assert EXPLORE_SHARE < mid < WARMUP_EXPLORE_SHARE


# ── Water penalty in scoring ────────────────────────────────────────────────

class TestWaterPenalty:
    def test_watered_neighbor_demoted(self):
        from app.resources.clap_features import AXIS_NAMES
        hot = _cand("hot")
        cool = _cand("cool")
        score_candidates(
            [hot, cool], p_final=None, confidence=0.0, play_counts={},
            recency_hours={}, axis_stats=None, axis_names=AXIS_NAMES,
            water_proximity={"cool": 1.0},
        )
        assert hot.score > cool.score

    def test_no_water_map_is_noop(self):
        from app.resources.clap_features import AXIS_NAMES
        a = _cand("a")
        score_candidates(
            [a], p_final=None, confidence=0.0, play_counts={},
            recency_hours={}, axis_stats=None, axis_names=AXIS_NAMES,
        )
        # pure novelty term, no penalty
        assert a.score == pytest.approx(ss.SCORE_W_NOVELTY)


# ── latest-wins: one active reaction per track (cancel semantics) ────────────

class TestLatestPerTrack:
    def test_water_supersedes_older_fire(self):
        fire = FireSignal("a", "fire", NOW - timedelta(hours=2))
        water = FireSignal("a", "water", NOW - timedelta(hours=1))
        out = {s.track_id: s.kind for s in latest_per_track([fire, water])}
        assert out == {"a": "water"}

    def test_fire_supersedes_older_water(self):
        water = FireSignal("a", "water", NOW - timedelta(hours=2))
        fire = FireSignal("a", "fire", NOW - timedelta(hours=1))
        out = {s.track_id: s.kind for s in latest_per_track([water, fire])}
        assert out == {"a": "fire"}

    def test_repeat_fire_collapses_to_newest(self):
        old = FireSignal("a", "fire", NOW - timedelta(days=3))
        new = FireSignal("a", "fire", NOW - timedelta(hours=1))
        out = latest_per_track([old, new])
        assert len(out) == 1 and out[0].created_at == new.created_at

    def test_distinct_tracks_kept(self):
        out = latest_per_track([_fire("a"), _fire("b", kind="water")])
        assert {s.track_id for s in out} == {"a", "b"}


# ── Persistent «заряд»: meter + unlock boundary ─────────────────────────────

class TestReactionContribution:
    def test_full_when_fresh(self):
        assert reaction_contribution(0.0) == pytest.approx(1.0)

    def test_half_at_halflife(self):
        # exactly at the unlock boundary (age == H_REACTION_DAYS == 1 day)
        assert reaction_contribution(H_REACTION_DAYS) == pytest.approx(0.5)

    def test_quarter_at_two_halflives(self):
        assert reaction_contribution(2 * H_REACTION_DAYS) == pytest.approx(0.25)

    def test_monotonic_decreasing_and_clamped(self):
        assert reaction_contribution(0.5) > reaction_contribution(1.5) > 0.0
        assert reaction_contribution(-1.0) == 1.0
