"""Unit tests for the Stream RecSys reward model (design §3).

Covers every boundary in the event→weight table: skip thresholds (30s abs /
25% short-track), completeness zones (65% / 85%), instant-replay detection,
the idle rule (streak of 5), and time-decay.
"""
from datetime import datetime, timedelta

import pytest

from app.services.stream_service import (
    H_IMPLICIT_DAYS,
    H_LIKE_DAYS,
    IDLE_STREAK,
    PlaybackSignal,
    W_FULL,
    W_MOST,
    W_NEUTRAL,
    W_REPLAY,
    W_SKIP,
    base_weight,
    decayed,
    is_skip,
    weight_events,
)

T0 = datetime(2026, 6, 1, 12, 0, 0)


def _ev(track="t1", played=230.0, dur=240.0, interacted=None, session="s1", minute=0):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=T0 + timedelta(minutes=minute), session_id=session,
        interacted=interacted,
    )


class TestIsSkip:
    def test_under_30s_is_skip_for_normal_track(self):
        assert is_skip(29.9, 240.0) is True

    def test_exactly_30s_is_not_skip(self):
        assert is_skip(30.0, 240.0) is False

    def test_short_track_uses_25_percent_rule(self):
        # 90s track: threshold = 22.5s, NOT the absolute 30s
        assert is_skip(22.4, 90.0) is True
        assert is_skip(23.0, 90.0) is False

    def test_unknown_duration_falls_back_to_absolute(self):
        assert is_skip(29.0, None) is True
        assert is_skip(31.0, None) is False


class TestBaseWeight:
    @pytest.mark.parametrize("played,dur,expected", [
        (20.0, 240.0, W_SKIP),       # skip
        (60.0, 240.0, W_NEUTRAL),    # 25% — noisy zone
        (155.9, 240.0, W_NEUTRAL),   # just under 65%
        (156.0, 240.0, W_MOST),      # exactly 65%
        (203.9, 240.0, W_MOST),      # just under 85%
        (204.0, 240.0, W_FULL),      # exactly 85%
        (240.0, 240.0, W_FULL),      # full listen
    ])
    def test_zones(self, played, dur, expected):
        assert base_weight(played, dur) == expected

    def test_unknown_duration_over_30s_is_neutral(self):
        """Ratios are uncomputable without duration — never reward blindly."""
        assert base_weight(500.0, None) == W_NEUTRAL


class TestReplay:
    def test_replay_after_full_listen_gets_replay_weight(self):
        events = [_ev(played=230.0), _ev(played=230.0, minute=4)]
        assert weight_events(events)[1] == W_REPLAY

    def test_no_replay_when_first_listen_incomplete(self):
        """Double-click from search (first listen < 85%) is NOT a replay."""
        events = [_ev(played=10.0), _ev(played=230.0, minute=1)]
        w = weight_events(events)
        assert w[1] == W_FULL  # plain full listen, no replay boost

    def test_no_replay_across_sessions(self):
        events = [_ev(played=230.0, session="s1"), _ev(played=230.0, session="s2", minute=4)]
        assert weight_events(events)[1] == W_FULL

    def test_no_replay_for_different_track(self):
        events = [_ev(track="a", played=230.0), _ev(track="b", played=230.0, minute=4)]
        assert weight_events(events)[1] == W_FULL


class TestIdleRule:
    def test_events_after_streak_get_zero(self):
        events = [_ev(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK + 2)]
        w = weight_events(events)
        assert w[:IDLE_STREAK] == [W_FULL] * IDLE_STREAK   # streak itself keeps weights
        assert w[IDLE_STREAK] == 0.0
        assert w[IDLE_STREAK + 1] == 0.0

    def test_action_resets_streak(self):
        events = [_ev(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK + 1)]
        events.append(_ev(track="t_act", interacted=True, minute=10))    # user acted
        events.append(_ev(track="t_after", interacted=False, minute=11))  # passive again — streak restarts
        w = weight_events(events)
        assert w[IDLE_STREAK] == 0.0          # zeroed during idle
        assert w[IDLE_STREAK + 1] == W_FULL   # interacted event counts again
        assert w[IDLE_STREAK + 2] == W_FULL   # 1st passive after reset — under streak

    def test_legacy_none_counts_as_interacted(self):
        """Events predating the interacted column must keep moving the profile."""
        events = [_ev(track=f"t{i}", interacted=None, minute=i) for i in range(IDLE_STREAK + 3)]
        assert all(w == W_FULL for w in weight_events(events))

    def test_replay_in_idle_zone_is_zeroed(self):
        """Loop-one background listening must not pump the profile via replays."""
        events = [_ev(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK)]
        events.append(_ev(track="loop", interacted=False, played=230.0, minute=20))
        events.append(_ev(track="loop", interacted=False, played=230.0, minute=24))  # instant replay
        w = weight_events(events)
        assert w[-1] == 0.0


class TestDecay:
    def test_zero_age_no_decay(self):
        assert decayed(1.0, 0.0, H_LIKE_DAYS) == 1.0

    def test_decay_at_half_life_param(self):
        """w·exp(−Δ/H): at Δ=H the weight is w/e (e-folding, per design formula)."""
        assert decayed(1.0, H_IMPLICIT_DAYS, H_IMPLICIT_DAYS) == pytest.approx(1 / 2.718281828, rel=1e-6)

    def test_explicit_outlives_implicit(self):
        """Same age — a like keeps more weight than an implicit signal."""
        age = 45.0
        assert decayed(1.0, age, H_LIKE_DAYS) > decayed(1.0, age, H_IMPLICIT_DAYS)

    def test_future_timestamp_clamps(self):
        assert decayed(0.5, -3.0, H_LIKE_DAYS) == 0.5

    def test_negative_weight_decays_toward_zero(self):
        d = decayed(W_SKIP, 30.0, H_IMPLICIT_DAYS)
        assert W_SKIP < d < 0.0
