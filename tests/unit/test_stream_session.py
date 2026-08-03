"""Unit tests for app.services.stream.session (design §2.1, §3, §4).

Covers the four rules that make the session honest: reactions superseding the
listens they happened on, skip forgiveness per positive cluster, carryover from
where the listener stopped last time, and the long-term seed fading out.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.services.stream import session as sess
from app.services.stream.baseline import Baseline, NEUTRAL
from app.services.stream.calibration import Calibration
from app.services.stream.signals import (
    FireSignal,
    PlaybackSignal,
    W_FULL,
    W_SKIP,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 3, 12, 0, 0)

# Cosine ≈ percentile, so thresholds in these tests read as plain similarity.
CALIB = Calibration([i / 100.0 for i in range(101)], source="table")


def _ev(track, *, played=230.0, dur=240.0, minute=0, session="s1", influence=True):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=NOW - timedelta(minutes=minute), session_id=session,
        influence=influence,
    )


def _skip_ev(track, *, minute=0, session="s1"):
    return _ev(track, played=5.0, minute=minute, session=session)


def _fire(track, *, kind="fire", hours=0.0):
    return FireSignal(track_id=track, kind=kind,
                      created_at=NOW - timedelta(hours=hours))


def _vec(angle_deg):
    a = np.radians(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=np.float32)


# ── §2.1 reaction supersedes the listen ────────────────────────────────────

class TestReactionCutoff:
    def test_cutoff_is_the_newest_reaction_per_track(self):
        old = _fire("t", hours=5)
        new = _fire("t", kind="water", hours=1)
        assert sess.reaction_cutoffs([old, new])["t"] == new.created_at

    def test_listen_during_the_reaction_is_superseded(self):
        ev = _ev("t")
        assert sess.superseded(ev, {"t": ev.played_at}) is True

    def test_grace_covers_the_event_flushed_after_the_press(self):
        """The button is pressed mid-track; the event lands on the NEXT track
        change, i.e. slightly after the reaction timestamp."""
        ev = _ev("t")
        pressed = ev.played_at - timedelta(seconds=sess.REACTION_GRACE_SEC - 1)
        assert sess.superseded(ev, {"t": pressed}) is True

    def test_beyond_the_grace_the_listen_counts(self):
        ev = _ev("t")
        pressed = ev.played_at - timedelta(seconds=sess.REACTION_GRACE_SEC + 60)
        assert sess.superseded(ev, {"t": pressed}) is False

    def test_untouched_tracks_are_unaffected(self):
        assert sess.superseded(_ev("t"), {"other": NOW}) is False

    def test_nearly_finished_watered_track_contributes_nothing(self):
        """The exact complaint: water at the three-minute mark of a four-minute
        track must not credit the track for the last minute."""
        ev = _ev("t", played=230.0, dur=240.0)
        cutoffs = sess.reaction_cutoffs([_fire("t", kind="water")])
        assert sess.weight_events([ev], cutoffs=cutoffs) == [0.0]
        assert sess.aggregate_event_weights([ev], NOW, cutoffs=cutoffs) == {}

    def test_without_a_reaction_the_same_listen_is_a_full_one(self):
        assert sess.weight_events([_ev("t")]) == [W_FULL]


class TestAdaptiveWeighting:
    def _baseline(self, m_skip=1.0, m_full=1.0):
        return Baseline(n_events=400, p_skip=0.25, p_full=0.5, p_react=0.08,
                        m_skip=m_skip, m_full=m_full, m_react=1.0)

    def test_skip_scales_with_the_listener_baseline(self):
        w = sess.weight_events([_skip_ev("t")], baseline=self._baseline(m_skip=2.0))
        assert w == [pytest.approx(W_SKIP * 2.0)]

    def test_completion_scales_with_the_listener_baseline(self):
        w = sess.weight_events([_ev("t")], baseline=self._baseline(m_full=0.5))
        assert w == [pytest.approx(W_FULL * 0.5)]

    def test_neutral_baseline_reproduces_the_raw_reward(self):
        assert sess.weight_events([_skip_ev("t")], baseline=NEUTRAL) == [W_SKIP]


# ── §4.3 skip forgiveness ──────────────────────────────────────────────────

class TestSkipForgiveness:
    def _cluster(self, tid, angle=0.0, weight=1.0):
        c = sess.Cluster(kind="positive", track_id=tid, weight=weight, members=[tid])
        c.vec = _vec(angle)
        return c

    def _skip(self, tid, minute=0):
        return sess.SkipEvent(track_id=tid, weight=1.0,
                              played_at=NOW - timedelta(minutes=minute))

    def test_a_single_similar_skip_is_forgiven(self):
        pos = [self._cluster("p", angle=0.0)]
        kept, n = sess.forgive_skips([self._skip("s1")], pos,
                                     {"s1": _vec(1.0)}, CALIB)
        assert kept == [] and n == 1

    def test_the_second_similar_skip_brings_the_first_back(self):
        pos = [self._cluster("p", angle=0.0)]
        skips = [self._skip("s1", 5), self._skip("s2", 1)]
        kept, n = sess.forgive_skips(
            skips, pos, {"s1": _vec(1.0), "s2": _vec(2.0)}, CALIB)
        assert {s.track_id for s in kept} == {"s1", "s2"}
        assert n == 0

    def test_clusters_forgive_independently(self):
        """Liking both rock and rap must not spend the rap forgiveness on rock."""
        pos = [self._cluster("rock", angle=0.0), self._cluster("rap", angle=80.0)]
        skips = [self._skip("r1"), self._skip("p1")]
        kept, n = sess.forgive_skips(
            skips, pos, {"r1": _vec(1.0), "p1": _vec(81.0)}, CALIB)
        assert kept == [] and n == 2

    def test_a_dissimilar_skip_is_never_forgiven(self):
        pos = [self._cluster("p", angle=0.0)]
        kept, n = sess.forgive_skips([self._skip("far")], pos,
                                     {"far": _vec(89.0)}, CALIB)
        assert [s.track_id for s in kept] == ["far"] and n == 0

    def test_a_skip_without_a_vector_is_never_forgiven(self):
        pos = [self._cluster("p", angle=0.0)]
        kept, n = sess.forgive_skips([self._skip("novec")], pos, {}, CALIB)
        assert [s.track_id for s in kept] == ["novec"] and n == 0

    def test_no_positive_clusters_means_nothing_to_forgive_against(self):
        skips = [self._skip("s1")]
        kept, n = sess.forgive_skips(skips, [], {"s1": _vec(0.0)}, CALIB)
        assert kept == skips and n == 0


# ── §4.4 carryover, §4.5 long-term fade ────────────────────────────────────

class TestCarryover:
    def test_picks_the_session_immediately_before_this_one(self):
        signals = [
            _ev("old", minute=600, session="s0"),
            _ev("prev1", minute=120, session="s1"),
            _ev("prev2", minute=100, session="s1"),
            _ev("cur", minute=1, session="s2"),
        ]
        tail, prev_id = sess.previous_session_tail(signals, "s2", NOW)
        assert prev_id == "s1"
        assert [e.track_id for e in tail] == ["prev1", "prev2"]

    def test_a_brand_new_session_still_gets_the_last_one(self):
        """No own events yet — the very first chunk should resume where the
        listener stopped, not start cold."""
        signals = [_ev("prev", minute=60, session="s1")]
        tail, prev_id = sess.previous_session_tail(signals, "s2", NOW)
        assert prev_id == "s1" and [e.track_id for e in tail] == ["prev"]

    def test_too_old_a_session_is_not_carried(self):
        stale = _ev("prev", minute=60 * 24 * (sess.CARRYOVER_MAX_AGE_D + 1), session="s1")
        tail, _ = sess.previous_session_tail([stale], "s2", NOW)
        assert tail == []

    def test_tail_is_capped(self):
        signals = [_ev(f"p{i}", minute=100 - i, session="s1") for i in range(30)]
        tail, _ = sess.previous_session_tail(signals, "s2", NOW)
        assert len(tail) == sess.CARRYOVER_TAIL

    def test_no_previous_session(self):
        assert sess.previous_session_tail([_ev("a", session="s2")], "s2", NOW) == ([], None)

    def test_scale_starts_at_the_configured_weight_and_fades_out(self):
        assert sess.carryover_scale(0) == pytest.approx(sess.CARRYOVER_W)
        assert sess.carryover_scale(sess.CARRYOVER_FADE_SIG) == 0.0
        assert sess.carryover_scale(sess.CARRYOVER_FADE_SIG * 3) == 0.0
        mid = sess.carryover_scale(sess.CARRYOVER_FADE_SIG // 2)
        assert 0.0 < mid < sess.CARRYOVER_W


class TestLongTermFade:
    def test_cold_session_is_all_long_term(self):
        assert sess.long_term_weight(0) == pytest.approx(1.0)

    def test_it_decays_to_a_floor_not_to_zero(self):
        assert sess.long_term_weight(sess.LONG_TERM_FADE) == sess.LONG_TERM_FLOOR
        assert sess.long_term_weight(1000) == sess.LONG_TERM_FLOOR

    def test_monotone_decreasing(self):
        vals = [sess.long_term_weight(n) for n in range(0, 12)]
        assert vals == sorted(vals, reverse=True)


# ── §4.6 affinity / repulsion ──────────────────────────────────────────────

class TestAffinityRepulsion:
    def _profile(self):
        pos = sess.Cluster("positive", "p", 1.0, ["p"]); pos.vec = _vec(0.0)
        water = sess.Cluster("water", "w", 1.0, ["w"]); water.vec = _vec(90.0)
        skip = sess.Cluster("skip", "s", 1.0, ["s"]); skip.vec = _vec(180.0)
        return sess.SessionProfile(positive=[pos], negative=[water, skip])

    def test_affinity_peaks_on_the_centroid(self):
        p = self._profile()
        assert p.affinity(_vec(0.0), CALIB) > p.affinity(_vec(45.0), CALIB)

    def test_water_repels_harder_than_a_skip_at_equal_similarity(self):
        p = self._profile()
        near_water = p.repulsion(_vec(90.0), CALIB)
        near_skip = p.repulsion(_vec(180.0), CALIB)
        assert near_water > near_skip
        assert near_water / max(near_skip, 1e-9) == pytest.approx(
            sess.REPEL_K_WATER / sess.REPEL_K_SKIP, rel=0.01)

    def test_a_candidate_far_from_everything_scores_zero_on_both(self):
        p = sess.SessionProfile()
        assert p.affinity(_vec(0.0), CALIB) == 0.0
        assert p.repulsion(_vec(0.0), CALIB) == 0.0

    def test_missing_vector_is_neutral(self):
        p = self._profile()
        assert p.affinity(None, CALIB) == 0.0
        assert p.repulsion(None, CALIB) == 0.0


# ── build(): the whole profile end to end ──────────────────────────────────

class TestBuild:
    def _build(self, *, signals, session_taste=(), all_taste=(), long_weights=None,
               vectors=None):
        return sess.build(
            signals=signals, session_id="s1",
            session_taste=list(session_taste), all_taste=list(all_taste),
            long_weights=long_weights or {}, baseline=NEUTRAL,
            calibration=CALIB, fetch_vectors=lambda ids: {
                t: v for t, v in (vectors or {}).items() if t in ids},
            now=NOW,
        )

    def test_positive_listening_forms_a_pull_cluster(self):
        p = self._build(
            signals=[_ev("a", minute=3), _ev("b", minute=2)],
            vectors={"a": _vec(0.0), "b": _vec(2.0)},
        )
        assert len(p.positive) == 1
        assert set(p.positive[0].members) == {"a", "b"}
        assert p.negative == []

    def test_water_forms_a_negative_cluster_and_mutes_the_track(self):
        p = self._build(
            signals=[_ev("a", minute=3)],
            all_taste=[_fire("cool", kind="water", hours=2)],
            vectors={"a": _vec(0.0), "cool": _vec(90.0)},
        )
        assert [c.kind for c in p.negative] == ["water"]
        assert p.muted == {"cool"}

    def test_a_lone_similar_skip_leaves_no_negative_cluster(self):
        p = self._build(
            signals=[_ev("a", minute=5), _ev("b", minute=4), _skip_ev("s", minute=1)],
            vectors={"a": _vec(0.0), "b": _vec(1.0), "s": _vec(2.0)},
        )
        assert p.forgiven_skips == 1
        assert p.negative == []

    def test_two_similar_skips_do_form_one(self):
        p = self._build(
            signals=[_ev("a", minute=5), _ev("b", minute=4),
                     _skip_ev("s1", minute=2), _skip_ev("s2", minute=1)],
            vectors={"a": _vec(0.0), "b": _vec(1.0),
                     "s1": _vec(2.0), "s2": _vec(3.0)},
        )
        assert p.forgiven_skips == 0
        assert [c.kind for c in p.negative] == ["skip"]

    def test_cold_session_leans_on_the_long_term_seed(self):
        p = self._build(signals=[], long_weights={"island": 1.0},
                        vectors={"island": _vec(0.0)})
        assert p.w_long == pytest.approx(1.0)
        assert [c.track_id for c in p.positive] == ["island"]

    def test_hand_queued_plays_do_not_shape_the_session(self):
        p = self._build(
            signals=[_ev("queued", minute=1, influence=False)],
            vectors={"queued": _vec(0.0)},
        )
        assert p.positive == []
        assert p.session_played == {"queued"}   # still remembered for anti-repeat

    def test_cluster_weights_are_normalised_to_one(self):
        p = self._build(
            signals=[_ev("a", minute=9), _ev("b", minute=8), _ev("c", minute=1)],
            vectors={"a": _vec(0.0), "b": _vec(1.0), "c": _vec(90.0)},
        )
        assert max(c.weight for c in p.positive) == pytest.approx(1.0)
