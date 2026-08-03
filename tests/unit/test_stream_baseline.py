"""Unit tests for app.services.stream.baseline (design §2.3).

The premise: a skip from someone who finishes everything carries more
information than a skip from someone who skips half the library. These cases
pin the direction of that adjustment, its hard caps, and the cold-start ramp
that stops a handful of events from whipsawing a new account.
"""
from datetime import datetime, timedelta

import pytest

from app.services.stream import baseline as bl
from app.services.stream.signals import FireSignal, PlaybackSignal

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 3, 12, 0, 0)


def _ev(track, played, dur=240.0, minute=0, influence=True):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=NOW - timedelta(minutes=minute), session_id="s",
        influence=influence,
    )


def _history(*, n, skip_share, full_share):
    """``n`` events with the requested skip/full mix (rest land in the neutral zone)."""
    n_skip = int(n * skip_share)
    n_full = int(n * full_share)
    out = []
    for i in range(n_skip):
        out.append(_ev(f"s{i}", played=5.0, minute=i))
    for i in range(n_full):
        out.append(_ev(f"f{i}", played=230.0, minute=n_skip + i))
    for i in range(n - n_skip - n_full):
        out.append(_ev(f"n{i}", played=120.0, minute=n_skip + n_full + i))
    return out


class TestColdStart:
    def test_no_history_is_neutral(self):
        b = bl.compute([], [])
        assert (b.m_skip, b.m_full, b.m_react) == (1.0, 1.0, 1.0)

    def test_below_the_floor_stays_neutral(self):
        events = _history(n=bl.BASELINE_MIN - 1, skip_share=0.9, full_share=0.0)
        b = bl.compute(events, [])
        assert (b.m_skip, b.m_full, b.m_react) == (1.0, 1.0, 1.0)

    def test_ramp_is_partial_between_floor_and_full(self):
        few = bl.compute(_history(n=100, skip_share=0.8, full_share=0.1), [])
        many = bl.compute(_history(n=400, skip_share=0.8, full_share=0.1), [])
        # same behaviour, more evidence → the adjustment is applied harder
        assert 1.0 > few.m_skip > many.m_skip

    def test_ramp_endpoints(self):
        assert bl._ramp(bl.BASELINE_MIN) == 0.0
        assert bl._ramp(bl.BASELINE_FULL) == 1.0
        assert bl._ramp(bl.BASELINE_FULL * 10) == 1.0


class TestDirection:
    def test_rare_skipper_gets_a_heavier_skip(self):
        b = bl.compute(_history(n=400, skip_share=0.05, full_share=0.8), [])
        assert b.m_skip > 1.5

    def test_chronic_skipper_gets_a_lighter_skip(self):
        b = bl.compute(_history(n=400, skip_share=0.8, full_share=0.1), [])
        assert b.m_skip < 0.7

    def test_someone_who_finishes_everything_gets_cheaper_completions(self):
        b = bl.compute(_history(n=400, skip_share=0.0, full_share=1.0), [])
        assert b.m_full < 1.0

    def test_someone_who_rarely_finishes_gets_dearer_completions(self):
        b = bl.compute(_history(n=400, skip_share=0.1, full_share=0.05), [])
        assert b.m_full > 1.0

    def test_reference_behaviour_lands_near_neutral(self):
        b = bl.compute(
            _history(n=400, skip_share=bl.P_SKIP_REF, full_share=bl.P_FULL_REF), [])
        assert b.m_skip == pytest.approx(1.0, abs=0.05)
        assert b.m_full == pytest.approx(1.0, abs=0.05)


class TestReactionRate:
    def test_rare_presser_gets_a_louder_reaction(self):
        events = _history(n=400, skip_share=0.2, full_share=0.5)
        taste = [FireSignal(f"t{i}", "fire", NOW) for i in range(2)]  # 0.5%
        assert bl.compute(events, taste).m_react > 1.5

    def test_frequent_presser_gets_a_quieter_reaction(self):
        events = _history(n=400, skip_share=0.2, full_share=0.5)
        taste = [FireSignal(f"t{i}", "fire", NOW) for i in range(200)]  # 50%
        assert bl.compute(events, taste).m_react < 0.7

    def test_no_reactions_at_all_hits_the_ceiling_not_infinity(self):
        b = bl.compute(_history(n=400, skip_share=0.2, full_share=0.5), [])
        assert b.m_react == pytest.approx(bl.ADAPT_CAP)


class TestCapsAndInputs:
    @pytest.mark.parametrize("skip_share,full_share", [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)])
    def test_every_multiplier_stays_inside_the_cap(self, skip_share, full_share):
        b = bl.compute(_history(n=400, skip_share=skip_share, full_share=full_share), [])
        for m in (b.m_skip, b.m_full, b.m_react):
            assert 1.0 / bl.ADAPT_CAP - 1e-9 <= m <= bl.ADAPT_CAP + 1e-9

    def test_events_without_duration_are_not_counted(self):
        events = [_ev(f"t{i}", played=200.0, dur=None, minute=i) for i in range(400)]
        assert bl.compute(events, []).n_events == 0

    def test_hand_queued_events_are_excluded(self):
        events = _history(n=400, skip_share=0.2, full_share=0.5)
        for e in events:
            object.__setattr__(e, "influence", False)
        assert bl.compute(events, []).n_events == 0

    def test_diagnostics_shape(self):
        d = bl.compute(_history(n=400, skip_share=0.2, full_share=0.5), []).as_diagnostics()
        assert set(d) == {"n_events", "p_skip", "p_full", "p_react",
                          "m_skip", "m_full", "m_react"}
