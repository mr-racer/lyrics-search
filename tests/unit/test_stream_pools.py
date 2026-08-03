"""Unit tests for app.services.stream.pools (design §4.6, §6).

The load-bearing case is the slider's «РЕДКОЕ» extreme: it must be a hard
promise, not a preference. Everything heard in the last month is out, and if
that empties the pool the chunk comes back short rather than quietly serving
yesterday's tracks.
"""
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.resources.clap_features import AXIS_NAMES, axis_version
from app.services.stream import pools
from app.services.stream.calibration import Calibration
from app.services.stream.session import Cluster
from app.services.stream.signals import PlaybackSignal, StreamCandidate, is_skip

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 3, 12, 0, 0)
CALIB = Calibration([i / 100.0 for i in range(101)], source="table")


def RNG():
    return random.Random(1234)


def _axes(value):
    return {a: value for a in AXIS_NAMES}


def _stats():
    return {"version": axis_version(), "n": 200,
            "mean": {a: 0.0 for a in AXIS_NAMES},
            "std": {a: 1.0 for a in AXIS_NAMES}}


def _cand(tid, *, artist="x", pool="familiar", score=0.0):
    return StreamCandidate(track_id=tid, payload={"artist": artist, "title": tid},
                           pool=pool, score=score)


class _Hit:
    def __init__(self, tid, score, payload=None):
        self.id = tid
        self.score = score
        self.payload = payload or {"title": tid, "artist": "a", "duration": 200.0}


class _FakeSearch:
    """query_points() keyed by the first component of the query vector."""

    def __init__(self, hits_by_marker):
        self.hits_by_marker = hits_by_marker
        self.calls = 0

    def query_points(self, collection_name, query, using, limit, with_payload):
        self.calls += 1
        return SimpleNamespace(points=self.hits_by_marker.get(round(query[0], 3), []))


def _cluster(tid, marker, *, kind="positive", weight=1.0):
    c = Cluster(kind=kind, track_id=tid, weight=weight, members=[tid])
    c.vec = np.array([marker, 0.0], dtype=np.float32)
    return c


# ── freshness ──────────────────────────────────────────────────────────────

class TestIsFresh:
    def test_never_played_is_fresh(self):
        assert pools.is_fresh("t", {}) is True

    def test_played_beyond_the_window_is_fresh(self):
        assert pools.is_fresh("t", {"t": pools.FRESH_WINDOW_DAYS * 24 + 1}) is True

    def test_played_yesterday_is_not_fresh(self):
        """The whole complaint: «редкое» must not mean «heard a week ago»."""
        assert pools.is_fresh("t", {"t": 24.0}) is False
        assert pools.is_fresh("t", {"t": 24.0 * 7}) is False

    def test_exactly_at_the_boundary_counts_as_fresh(self):
        assert pools.is_fresh("t", {"t": pools.FRESH_WINDOW_DAYS * 24}) is True


class TestSkipBurst:
    def _ev(self, played, minute):
        return PlaybackSignal(track_id=f"t{minute}", played_sec=played, total_dur=240.0,
                              played_at=NOW, session_id="s")

    def test_three_of_the_last_five_trigger(self):
        events = [self._ev(230.0, i) for i in range(5)] + \
                 [self._ev(5.0, 5), self._ev(230.0, 6), self._ev(5.0, 7), self._ev(5.0, 8)]
        assert pools.skip_burst(events, is_skip) is True

    def test_old_skips_buried_by_fresh_listens_do_not(self):
        events = [self._ev(5.0, i) for i in range(3)] + \
                 [self._ev(230.0, i) for i in range(3, 8)]
        assert pools.skip_burst(events, is_skip) is False


# ── scoring ────────────────────────────────────────────────────────────────

class TestScoreCandidate:
    def test_affinity_dominates(self):
        near = pools.score_candidate(affinity=0.9, repulsion=0.0, axis_match=0.0,
                                     play_count=0, recency_h=None)
        far = pools.score_candidate(affinity=0.1, repulsion=0.0, axis_match=0.0,
                                    play_count=0, recency_h=None)
        assert near > far
        assert near == pytest.approx(pools.W_AFFINITY * 0.9 + pools.W_NOVELTY)

    def test_repulsion_subtracts(self):
        clean = pools.score_candidate(affinity=0.5, repulsion=0.0, axis_match=0.0,
                                      play_count=0, recency_h=None)
        repelled = pools.score_candidate(affinity=0.5, repulsion=1.0, axis_match=0.0,
                                         play_count=0, recency_h=None)
        assert repelled == pytest.approx(clean - pools.W_REPULSION)

    def test_recent_play_penalised(self):
        recent = pools.score_candidate(affinity=0.5, repulsion=0.0, axis_match=0.0,
                                       play_count=1, recency_h=0.5)
        stale = pools.score_candidate(affinity=0.5, repulsion=0.0, axis_match=0.0,
                                      play_count=1, recency_h=None)
        assert recent < stale

    def test_novelty_boosts_unplayed(self):
        unplayed = pools.score_candidate(affinity=0.5, repulsion=0.0, axis_match=0.0,
                                         play_count=0, recency_h=None)
        worn = pools.score_candidate(affinity=0.5, repulsion=0.0, axis_match=0.0,
                                     play_count=20, recency_h=None)
        assert unplayed > worn


class TestClusterNeighbourhood:
    def test_best_value_per_track_wins_and_records_its_cluster(self):
        clusters = [_cluster("c1", 1.0, weight=1.0), _cluster("c2", 2.0, weight=1.0)]
        fake = _FakeSearch({1.0: [_Hit("t1", 0.7), _Hit("t2", 0.9)],
                            2.0: [_Hit("t1", 0.95)]})
        vals, payloads, owner, sims = pools.cluster_neighbourhood(
            fake, "col", clusters, CALIB, limit=10)
        assert vals["t1"] == pytest.approx(0.95)
        assert owner["t1"] == "c2"
        assert owner["t2"] == "c1"
        assert sims["t1"] == pytest.approx(0.95)
        assert set(payloads) == {"t1", "t2"}

    def test_cluster_weight_scales_the_value(self):
        clusters = [_cluster("c1", 1.0, weight=0.5)]
        fake = _FakeSearch({1.0: [_Hit("t1", 0.8)]})
        vals, _, _, sims = pools.cluster_neighbourhood(fake, "col", clusters, CALIB, limit=10)
        assert vals["t1"] == pytest.approx(0.4)
        assert sims["t1"] == pytest.approx(0.8)   # the raw similarity is unscaled

    def test_repel_coefficient_applies_only_with_use_k(self):
        water = [_cluster("w", 1.0, kind="water")]
        skip = [_cluster("s", 1.0, kind="skip")]
        fake = _FakeSearch({1.0: [_Hit("t", 0.8)]})
        w_val, _, _, _ = pools.cluster_neighbourhood(fake, "col", water, CALIB,
                                                     limit=10, use_k=True)
        s_val, _, _, _ = pools.cluster_neighbourhood(fake, "col", skip, CALIB,
                                                     limit=10, use_k=True)
        assert w_val["t"] > s_val["t"]

    def test_excluded_ids_never_surface(self):
        fake = _FakeSearch({1.0: [_Hit("bad", 0.99), _Hit("ok", 0.5)]})
        vals, _, _, _ = pools.cluster_neighbourhood(
            fake, "col", [_cluster("c", 1.0)], CALIB, limit=10, excluded={"bad"})
        assert set(vals) == {"ok"}

    def test_a_search_failure_is_survivable(self):
        class _Boom:
            def query_points(self, **kw):
                raise RuntimeError("qdrant down")
        vals, _, _, _ = pools.cluster_neighbourhood(
            _Boom(), "col", [_cluster("c", 1.0)], CALIB, limit=10)
        assert vals == {}

    def test_cluster_without_a_centroid_is_skipped(self):
        c = Cluster(kind="positive", track_id="c", weight=1.0, members=["c"])
        fake = _FakeSearch({})
        pools.cluster_neighbourhood(fake, "col", [c], CALIB, limit=10)
        assert fake.calls == 0


class TestBuildCandidates:
    def test_labels_pools_and_drops_short_tracks(self):
        out = pools.build_candidates(
            affinity_by_id={"a": 0.9, "b": 0.5, "tiny": 0.9},
            repulsion_by_id={"b": 1.0},
            payload_by_id={"a": {"artist": "x", "duration": 200.0},
                           "b": {"artist": "y", "duration": 200.0},
                           "tiny": {"artist": "z", "duration": 12.0}},
            owner_by_id={"a": "c1"}, axis_match_by_id={},
            play_counts={}, recency_hours={"b": 5.0},
            pool_of=lambda t: "fresh" if t == "a" else "familiar",
        )
        by_id = {c.track_id: c for c in out}
        assert set(by_id) == {"a", "b"}          # the 12s interlude is gone
        assert by_id["a"].pool == "fresh"
        assert by_id["a"].anchor_track_id == "c1"
        assert by_id["a"].score > by_id["b"].score


# ── stratified sampling ────────────────────────────────────────────────────

class TestStratify:
    def test_round_robin_across_bins(self):
        def z(e, x):
            d = {a: 0.0 for a in AXIS_NAMES}
            d["energy"], d["experimental"] = e, x
            return d
        eligible = ([(f"low{i}", z(-2.0, 0.0)) for i in range(5)]
                    + [(f"high{i}", z(2.0, 0.0)) for i in range(5)])
        out = pools.stratify(eligible, pool_size=4, rng=RNG())
        assert sum(1 for t in out if t.startswith("low")) == 2
        assert sum(1 for t in out if t.startswith("high")) == 2

    def test_tracks_without_axes_still_sampled(self):
        assert len(pools.stratify([(f"t{i}", None) for i in range(3)], 2, RNG())) == 2

    def test_drains_when_pool_smaller_than_request(self):
        assert pools.stratify([("only", None)], 10, RNG()) == ["only"]


class TestStratifiedFresh:
    def _client(self, monkeypatch, points):
        monkeypatch.setattr(pools, "light_points", lambda c, n, **kw: points)
        return object()

    def _pts(self):
        return [(tid, {"artist": "a", "title": tid, "duration": 200.0,
                       "sonic_axes": _axes(0.0)})
                for tid in ("fresh1", "fresh2", "heard", "near_neg")]

    def test_only_fresh_unexcluded_tracks_survive(self, monkeypatch):
        client = self._client(monkeypatch, self._pts())
        out = pools.stratified_fresh(
            client, "col", excluded=set(), recency_hours={"heard": 2.0},
            repulsion_by_id={}, neg_sim_by_id={},
            axis_stats=_stats(), axis_names=AXIS_NAMES,
            p_final=None, confidence=0.0, play_counts={}, pool_size=10, rng=RNG(),
        )
        assert "heard" not in {c.track_id for c in out}
        assert {"fresh1", "fresh2"} <= {c.track_id for c in out}

    def test_picks_inside_a_negative_neighbourhood_are_dropped(self, monkeypatch):
        client = self._client(monkeypatch, self._pts())
        out = pools.stratified_fresh(
            client, "col", excluded=set(), recency_hours={},
            repulsion_by_id={}, neg_sim_by_id={"near_neg": pools.EXPLORE_NEG_PCT + 0.01},
            axis_stats=_stats(), axis_names=AXIS_NAMES,
            p_final=None, confidence=0.0, play_counts={}, pool_size=10, rng=RNG(),
        )
        assert "near_neg" not in {c.track_id for c in out}

    def test_everything_it_returns_is_labelled_fresh(self, monkeypatch):
        client = self._client(monkeypatch, self._pts())
        out = pools.stratified_fresh(
            client, "col", excluded=set(), recency_hours={},
            repulsion_by_id={}, neg_sim_by_id={},
            axis_stats=_stats(), axis_names=AXIS_NAMES,
            p_final=None, confidence=0.0, play_counts={}, pool_size=10, rng=RNG(),
        )
        assert out and all(c.pool == "fresh" for c in out)


# ── assembly + the slider ──────────────────────────────────────────────────

class TestAssembleChunk:
    def _fresh(self, n=5):
        return [_cand(f"f{i}", artist=f"fa{i}", pool="fresh", score=0.9 - i * 0.1)
                for i in range(n)]

    def _familiar(self, n=5):
        return [_cand(f"m{i}", artist=f"ma{i}", pool="familiar", score=0.8 - i * 0.1)
                for i in range(n)]

    def test_quota_is_honoured(self):
        out = pools.assemble_chunk(self._fresh(), self._familiar(), n=3, fresh_quota=1,
                                   recent_artists=[], fresh_is_hard=False)
        assert len(out) == 3
        assert sum(1 for c in out if c.pool == "fresh") == 1

    def test_slider_at_rare_serves_only_fresh(self):
        out = pools.assemble_chunk(self._fresh(), self._familiar(), n=3, fresh_quota=3,
                                   recent_artists=[], fresh_is_hard=True)
        assert [c.pool for c in out] == ["fresh"] * 3

    def test_slider_at_rare_returns_a_short_chunk_when_nothing_is_fresh(self):
        """The promise is hard: a dry fresh pool must NOT leak heard tracks."""
        out = pools.assemble_chunk([], self._familiar(), n=3, fresh_quota=3,
                                   recent_artists=[], fresh_is_hard=True)
        assert out == []

    def test_slider_at_rare_serves_what_it_has_and_stops(self):
        out = pools.assemble_chunk(self._fresh(1), self._familiar(), n=3, fresh_quota=3,
                                   recent_artists=[], fresh_is_hard=True)
        assert [c.track_id for c in out] == ["f0"]

    def test_slider_at_liked_prefers_familiar(self):
        out = pools.assemble_chunk(self._fresh(), self._familiar(), n=3, fresh_quota=0,
                                   recent_artists=[], fresh_is_hard=False)
        assert all(c.pool != "fresh" for c in out)

    def test_dry_familiar_tops_up_from_fresh(self):
        """The other direction is NOT a hard promise — undersupply is worse."""
        out = pools.assemble_chunk(self._fresh(), [], n=3, fresh_quota=0,
                                   recent_artists=[], fresh_is_hard=False)
        assert len(out) == 3

    def test_higher_score_first_within_a_pool(self):
        lo = _cand("lo", artist="a1", pool="familiar", score=0.1)
        hi = _cand("hi", artist="a2", pool="familiar", score=0.9)
        out = pools.assemble_chunk([], [lo, hi], n=2, fresh_quota=0,
                                   recent_artists=[], fresh_is_hard=False)
        assert [c.track_id for c in out] == ["hi", "lo"]

    def test_no_three_consecutive_same_artist(self):
        main = [_cand("x1", artist="X", pool="familiar", score=0.99),
                _cand("y1", artist="Y", pool="familiar", score=0.5)]
        out = pools.assemble_chunk([], main, n=2, fresh_quota=0,
                                   recent_artists=["x", "x"], fresh_is_hard=False)
        assert out[0].track_id == "y1"

    def test_artist_rule_bends_rather_than_undersupply(self):
        main = [_cand(f"x{i}", artist="X", pool="familiar", score=0.9) for i in range(3)]
        out = pools.assemble_chunk([], main, n=3, fresh_quota=0,
                                   recent_artists=["x", "x"], fresh_is_hard=False)
        assert len(out) == 3

    def test_no_more_than_two_fresh_in_a_row_while_familiar_is_available(self):
        out = pools.assemble_chunk(self._fresh(6), self._familiar(6), n=6, fresh_quota=3,
                                   recent_artists=[], fresh_is_hard=False)
        run = best = 0
        for c in out:
            run = run + 1 if c.pool == "fresh" else 0
            best = max(best, run)
        assert best <= pools.MAX_CONSECUTIVE_SAME_POOL

    def test_both_pools_dry_returns_nothing(self):
        assert pools.assemble_chunk([], [], n=3, fresh_quota=1,
                                    recent_artists=[], fresh_is_hard=False) == []

    def test_never_returns_more_than_n(self):
        out = pools.assemble_chunk(self._fresh(10), self._familiar(10), n=3,
                                   fresh_quota=2, recent_artists=[], fresh_is_hard=False)
        assert len(out) == 3

    def test_no_track_is_served_twice(self):
        out = pools.assemble_chunk(self._fresh(4), self._familiar(4), n=6,
                                   fresh_quota=3, recent_artists=[], fresh_is_hard=False)
        assert len({c.track_id for c in out}) == len(out)
