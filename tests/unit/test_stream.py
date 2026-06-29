"""Consolidated unit tests for the Stream RecSys (app.services.stream_service).

Merges the former per-surface suites into one file, one class per source:
  - TestDurationFilter        (test_stream_duration_filter.py)
  - pools/scoring classes     (test_stream_pools_scoring.py)
  - profile-surface classes   (test_stream_profile_surface.py)
  - taste-profile classes     (test_stream_profiles.py)
  - reward-model classes      (test_stream_reward_model.py)
  - TestSimilar               (test_stream_similar.py)
  - stream-token classes      (test_stream_token.py)

Two tests (TestAxisPlaylist::test_no_axis_stats_returns_empty_with_reason and
TestSimilar::test_no_axis_stats_degrades_to_pure_cosine) fail as baseline noise
on boxes that ship axis stats — that pre-existing behaviour is unchanged here.
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user, get_user_for_stream
from app.resources.clap_features import AXIS_NAMES, axis_version
from app.resources.metadata_db import MetadataDB
from app.services import stream_service as ss
from app.services.auth_service import (
    AuthService,
    InvalidTokenError,
    STREAM_TOKEN_TTL_SECONDS,
    TokenExpiredError,
)
from app.services.stream_service import (
    ANCHOR_MERGE_THRESHOLD,
    ANTIREPEAT_FLOOR_TRACKS,
    AXIS_MATCH_DIST_SCALE,
    Anchor,
    H_IMPLICIT_DAYS,
    H_LIKE_DAYS,
    IDLE_STREAK,
    LIKED_COOLDOWN_H,
    PlaybackSignal,
    ReactionSignal,
    SCORE_W_ANCHOR,
    SCORE_W_NOVELTY,
    SESSION_ANCHOR_BOOST,
    SIMILAR_W_AXES,
    SIMILAR_W_CLAP,
    StreamCandidate,
    W_FULL,
    W_MOST,
    W_NEUTRAL,
    W_REPLAY,
    W_SKIP,
    _anti_repeat_floor,
    aggregate_event_weights,
    aggregate_reaction_weights,
    assemble_chunk,
    axis_match_score,
    axis_playlist,
    axis_preferences,
    base_weight,
    blend_axis_preferences,
    combine_weights,
    count_session_signals,
    decayed,
    is_skip,
    long_term_profile,
    merge_anchors,
    negative_track_ids,
    pool_anchor_candidates,
    sample_liked_tracks,
    score_candidates,
    select_positive_anchors,
    session_blend_weight,
    similar_tracks,
    stratify_explore,
    union_anchor_weights,
    weight_events,
    z_scores_for_axes,
)

# ── shared constants ─────────────────────────────────────────────────────────

RNG = lambda: random.Random(42)  # noqa: E731

# pools/scoring STATS (no version/n keys)
STATS = {
    "mean": {a: 0.0 for a in AXIS_NAMES},
    "std": {a: 1.0 for a in AXIS_NAMES},
}

# similar STATS (carries version/n; renamed — body differs from pools/scoring STATS)
STATS_SIMILAR = {
    "version": None,  # filled by axis_stats_in_db fixture below
    "n": 200,
    "mean": {a: 0.0 for a in AXIS_NAMES},
    "std": {a: 1.0 for a in AXIS_NAMES},
}

NOW = datetime(2026, 6, 11, 12, 0, 0)            # profile-surface clock
NOW_PROFILES = datetime(2026, 6, 10, 12, 0, 0)   # taste-profile clock (renamed; differs)
T0 = datetime(2026, 6, 1, 12, 0, 0)              # reward-model clock
JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


# ── shared helpers ───────────────────────────────────────────────────────────

def _axes(value):
    return {a: value for a in AXIS_NAMES}


def _stats():
    return {
        "version": axis_version(), "n": 200,
        "mean": {a: 0.0 for a in AXIS_NAMES},
        "std": {a: 1.0 for a in AXIS_NAMES},
    }


def _cand(tid, artist="x", pool="anchor", cos=0.0, axes=None):
    payload = {"artist": artist, "title": tid}
    if axes is not None:
        payload["sonic_axes"] = axes
    return StreamCandidate(track_id=tid, payload=payload, pool=pool, max_anchor_cos=cos)


def _ev(track, played=230.0, dur=240.0, days_ago=0.0, session="s1", interacted=None):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=NOW_PROFILES - timedelta(days=days_ago), session_id=session,
        interacted=interacted,
    )


def _like(track, days_ago=0.0):
    return ReactionSignal(track_id=track, reaction="like",
                          updated_at=NOW_PROFILES - timedelta(days=days_ago))


def _dislike(track, days_ago=0.0):
    return ReactionSignal(track_id=track, reaction="dislike",
                          updated_at=NOW_PROFILES - timedelta(days=days_ago))


def _ev_reward(track="t1", played=230.0, dur=240.0, interacted=None, session="s1", minute=0):
    return PlaybackSignal(
        track_id=track, played_sec=played, total_dur=dur,
        played_at=T0 + timedelta(minutes=minute), session_id=session,
        interacted=interacted,
    )


def _seed(axes_value=1.0):
    return _PointSimilar("seed", vector=[1.0, 0.0], payload={"sonic_axes": _axes(axes_value)})


class _Hit:
    def __init__(self, tid, score, payload=None):
        self.id = tid
        self.score = score
        self.payload = payload or {"title": tid, "artist": "a"}


class _FakeQdrantSearch:
    """query_points() returns canned hits per anchor track_id (keyed by query[0]).

    Mirrors qdrant-client >= 1.10 — the legacy .search() method was REMOVED
    upstream, so fakes must not offer it (that's how the breakage slipped by)."""

    def __init__(self, hits_by_marker):
        self.hits_by_marker = hits_by_marker

    def query_points(self, collection_name, query, using, limit, with_payload):
        from types import SimpleNamespace
        marker = query[0]  # first vector component identifies the anchor
        return SimpleNamespace(points=self.hits_by_marker[marker])


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


class _PointSimilar:
    def __init__(self, tid, vector=None, payload=None, score=None):
        self.id = tid
        self.vector = {"clap": vector} if vector is not None else {}
        self.payload = payload or {}
        self.score = score


class FakeQdrantSimilar:
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


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


@pytest.fixture
def axis_stats_in_db(_isolated_db):
    MetadataDB.set_axis_norm_stats("col", {**STATS_SIMILAR, "version": axis_version()})


@pytest.fixture
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "stream_token_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


@pytest.fixture
def auth():
    return AuthService(jwt_secret=JWT_SECRET)


@pytest.fixture
def owner(reset_db, auth):
    auth.create_owner(email="o@x.y", password="ownerpass1234")
    user, _ = auth.login(email="o@x.y", password="ownerpass1234")
    return user


def _make_app(auth: AuthService):
    app = FastAPI()
    app.state.auth_service = auth

    @app.get("/audio")
    def audio(user=Depends(get_user_for_stream)):
        return {"id": user.id}

    return app


def _make_auth_app(auth: AuthService):
    from app.api.routes.auth import router as auth_router
    app = FastAPI()
    app.state.auth_service = auth
    app.include_router(auth_router, prefix="/api/v1")
    return app


# ── <60s duration filter (test_stream_duration_filter.py) ────────────────────

class TestDurationFilter:
    @pytest.mark.parametrize("dur,ok", [
        (None, True),      # unknown → keep
        (0, True),         # zero/unknown → keep
        (5, False),        # short → drop
        (59.9, False),
        (60.0, True),      # exactly 60 → keep
        (180, True),
        ("75", True),      # coerced
        ("12", False),
    ])
    def test_duration_ok(self, dur, ok):
        assert ss._duration_ok({"duration": dur}) is ok

    def test_duration_ok_missing_key(self):
        assert ss._duration_ok({}) is True
        assert ss._duration_ok(None) is True


# ── candidate pools, scoring, chunk assembly (test_stream_pools_scoring.py) ───

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

    def test_slider_zero_strict_no_liked_topup(self):
        """Slider at the «новое» extreme is a hard promise: a dry main pool
        returns a short chunk instead of leaking liked tracks beyond quota.
        In small libraries main dries constantly (liked + session-played are
        excluded), so without this the slider's zero position is a no-op."""
        liked = [_cand(f"l{i}", artist=f"a{i}", pool="liked") for i in range(5)]
        out = assemble_chunk([], liked, n=3, liked_share=0.0, recent_artists=[])
        assert out == []

    def test_slider_zero_strict_partial_main(self):
        """Same promise when main has SOME candidates: serve them, then stop."""
        main = [_cand("m0", artist="a0", cos=0.5)]
        liked = [_cand(f"l{i}", artist=f"la{i}", pool="liked") for i in range(5)]
        out = assemble_chunk(main, liked, n=3, liked_share=0.0, recent_artists=[])
        assert [c.pool for c in out] == ["anchor"]

    def test_both_dry_returns_short_chunk(self):
        out = assemble_chunk([], [], n=3, liked_share=0.5, recent_artists=[])
        assert out == []


class TestAntiRepeatFloor:
    """«Жёсткий пол» под round-replay: just-heard tracks never recur (design 2026-06-14)."""

    def test_empty_recency_is_empty(self):
        assert _anti_repeat_floor({}) == set()

    def test_last_n_tracks_floored(self):
        # 15 tracks all played long ago (outside the minutes window) → floor is the
        # ANTIREPEAT_FLOOR_TRACKS most-recent by recency, nothing more.
        recency = {f"t{i}": float(i + 1) for i in range(15)}  # 1h … 15h ago
        floor = _anti_repeat_floor(recency)
        assert len(floor) == ANTIREPEAT_FLOOR_TRACKS
        assert "t0" in floor          # 1h ago — most recent
        assert "t14" not in floor     # 15h ago — oldest, beyond last-N

    def test_within_minutes_floored_even_beyond_n(self):
        # n_tracks=0 isolates the minutes rule: a fresh play floors, an old one doesn't.
        floor = _anti_repeat_floor({"fresh": 0.1, "old": 100.0}, n_tracks=0)
        assert floor == {"fresh"}

    def test_union_of_both_rules(self):
        recency = {"a": 0.1, "b": 0.2, "c": 1.0, "d": 2.0}
        # minutes(30 → 0.5h): {a, b} ; last-1 by recency: {a} → union {a, b}
        assert _anti_repeat_floor(recency, n_tracks=1, minutes=30) == {"a", "b"}


# ── long_term_profile islands/axes + axis_playlist (test_stream_profile_surface.py) ──

@pytest.mark.usefixtures("_isolated_db")
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


@pytest.mark.usefixtures("_isolated_db")
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
        # t1+t2 merge into one island (cos≈1); t3 is a lone track and a
        # single-song «island» is dropped — only the real cluster survives.
        assert len(out["islands"]) == 1
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


@pytest.mark.usefixtures("_isolated_db")
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


# ── taste profiles: aggregation/anchors/merging/blending (test_stream_profiles.py) ──

class TestAggregation:
    def test_fresh_full_listen_weight(self):
        w = aggregate_event_weights([_ev("a")], NOW_PROFILES)
        assert w["a"] == pytest.approx(W_FULL)

    def test_old_event_decays(self):
        w = aggregate_event_weights([_ev("a", days_ago=H_IMPLICIT_DAYS)], NOW_PROFILES)
        assert w["a"] == pytest.approx(W_FULL / 2.718281828, rel=1e-6)

    def test_sessions_weighted_independently(self):
        """Replay detection must not fire across different sessions."""
        events = [_ev("a", session="s1"), _ev("a", session="s2")]
        w = aggregate_event_weights(events, NOW_PROFILES)
        assert w["a"] == pytest.approx(2 * W_FULL)  # two full listens, no replay boost

    def test_reactions_decay_with_h_like(self):
        w = aggregate_reaction_weights([_like("a", days_ago=90.0)], NOW_PROFILES)
        assert w["a"] == pytest.approx(1.0 / 2.718281828, rel=1e-6)

    def test_combine_sums_across_sources(self):
        total = combine_weights({"a": 0.4, "b": -0.6}, {"a": 1.0})
        assert total == {"a": 1.4, "b": -0.6}


class TestNegativeSet:
    def test_dislike_is_hard_negative_regardless_of_age(self):
        neg = negative_track_ids([], [_dislike("d", days_ago=500.0)], NOW_PROFILES)
        assert "d" in neg

    def test_two_fresh_skips_cross_threshold(self):
        events = [_ev("s", played=5.0, days_ago=1.0), _ev("s", played=5.0, days_ago=0.5)]
        assert "s" in negative_track_ids(events, [], NOW_PROFILES)

    def test_one_skip_is_not_enough(self):
        assert negative_track_ids([_ev("s", played=5.0)], [], NOW_PROFILES) == set()

    def test_two_ancient_skips_decay_out(self):
        events = [_ev("s", played=5.0, days_ago=60.0), _ev("s", played=5.0, days_ago=50.0)]
        assert "s" not in negative_track_ids(events, [], NOW_PROFILES)


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
    @pytest.mark.parametrize("n,expected", [(0, 0.0), (3, 0.3), (5, 0.5), (10, 0.5), (25, 0.5)])
    def test_w_s_caps_at_half(self, n, expected):
        """Session influence ramps 0.1/signal but never exceeds parity (0.5) —
        long-term taste always keeps an equal vote (anti-trap guarantee)."""
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


# ── reward model (test_stream_reward_model.py) ───────────────────────────────

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
        events = [_ev_reward(played=230.0), _ev_reward(played=230.0, minute=4)]
        assert weight_events(events)[1] == W_REPLAY

    def test_no_replay_when_first_listen_incomplete(self):
        """Double-click from search (first listen < 85%) is NOT a replay."""
        events = [_ev_reward(played=10.0), _ev_reward(played=230.0, minute=1)]
        w = weight_events(events)
        assert w[1] == W_FULL  # plain full listen, no replay boost

    def test_no_replay_across_sessions(self):
        events = [_ev_reward(played=230.0, session="s1"), _ev_reward(played=230.0, session="s2", minute=4)]
        assert weight_events(events)[1] == W_FULL

    def test_no_replay_for_different_track(self):
        events = [_ev_reward(track="a", played=230.0), _ev_reward(track="b", played=230.0, minute=4)]
        assert weight_events(events)[1] == W_FULL


class TestIdleRule:
    def test_events_after_streak_get_zero(self):
        events = [_ev_reward(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK + 2)]
        w = weight_events(events)
        assert w[:IDLE_STREAK] == [W_FULL] * IDLE_STREAK   # streak itself keeps weights
        assert w[IDLE_STREAK] == 0.0
        assert w[IDLE_STREAK + 1] == 0.0

    def test_action_resets_streak(self):
        events = [_ev_reward(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK + 1)]
        events.append(_ev_reward(track="t_act", interacted=True, minute=10))    # user acted
        events.append(_ev_reward(track="t_after", interacted=False, minute=11))  # passive again — streak restarts
        w = weight_events(events)
        assert w[IDLE_STREAK] == 0.0          # zeroed during idle
        assert w[IDLE_STREAK + 1] == W_FULL   # interacted event counts again
        assert w[IDLE_STREAK + 2] == W_FULL   # 1st passive after reset — under streak

    def test_legacy_none_counts_as_interacted(self):
        """Events predating the interacted column must keep moving the profile."""
        events = [_ev_reward(track=f"t{i}", interacted=None, minute=i) for i in range(IDLE_STREAK + 3)]
        assert all(w == W_FULL for w in weight_events(events))

    def test_replay_in_idle_zone_is_zeroed(self):
        """Loop-one background listening must not pump the profile via replays."""
        events = [_ev_reward(track=f"t{i}", interacted=False, minute=i) for i in range(IDLE_STREAK)]
        events.append(_ev_reward(track="loop", interacted=False, played=230.0, minute=20))
        events.append(_ev_reward(track="loop", interacted=False, played=230.0, minute=24))  # instant replay
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


# ── similar_tracks(): CLAP neighbors re-ranked by axes (test_stream_similar.py) ──

@pytest.mark.usefixtures("_isolated_db")
class TestSimilar:
    def test_axis_closeness_rescores_equal_cosines(self, axis_stats_in_db):
        """Same CLAP cosine — the candidate whose axes match the seed wins."""
        hits = [
            _PointSimilar("far_axes", score=0.9, payload={"sonic_axes": _axes(-2.0)}),
            _PointSimilar("near_axes", score=0.9, payload={"sonic_axes": _axes(1.0)}),
        ]
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(_seed(axes_value=1.0), hits),
            collection_name="col", seed_track_id="seed",
        )
        ids = [c.track_id for c in out["tracks"]]
        assert ids == ["near_axes", "far_axes"]
        near = out["tracks"][0]
        assert near.axis_match == pytest.approx(1.0)
        assert near.score == pytest.approx(SIMILAR_W_CLAP * 0.9 + SIMILAR_W_AXES * 1.0)

    def test_no_axis_stats_degrades_to_pure_cosine(self):
        """Without usable stats the axis term is 0 — order = CLAP order."""
        hits = [
            _PointSimilar("second", score=0.8, payload={"sonic_axes": _axes(1.0)}),
            _PointSimilar("first", score=0.95, payload={"sonic_axes": _axes(-3.0)}),
        ]
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(_seed(), hits),
            collection_name="col", seed_track_id="seed",
        )
        assert [c.track_id for c in out["tracks"]] == ["first", "second"]
        assert all(c.axis_match == 0.0 for c in out["tracks"])

    def test_seed_itself_and_dislikes_filtered(self, axis_stats_in_db):
        MetadataDB.set_reaction("hated", "col", "dislike")
        hits = [
            _PointSimilar("seed", score=1.0, payload={"sonic_axes": _axes(1.0)}),
            _PointSimilar("hated", score=0.99, payload={"sonic_axes": _axes(1.0)}),
            _PointSimilar("ok", score=0.5, payload={"sonic_axes": _axes(1.0)}),
        ]
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(_seed(), hits),
            collection_name="col", seed_track_id="seed",
        )
        assert [c.track_id for c in out["tracks"]] == ["ok"]

    def test_exclude_ids_respected(self, axis_stats_in_db):
        hits = [_PointSimilar("a", score=0.9, payload={}), _PointSimilar("b", score=0.8, payload={})]
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(_seed(), hits),
            collection_name="col", seed_track_id="seed", exclude_ids=["a"],
        )
        assert [c.track_id for c in out["tracks"]] == ["b"]

    def test_seed_without_clap_vector_returns_empty(self):
        seed = _PointSimilar("seed", vector=None, payload={})
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(seed, []),
            collection_name="col", seed_track_id="seed",
        )
        assert out["tracks"] == []

    def test_limit_applied_after_rerank(self, axis_stats_in_db):
        hits = [_PointSimilar(f"t{i}", score=0.9 - i * 0.01, payload={"sonic_axes": _axes(1.0)})
                for i in range(20)]
        out = similar_tracks(
            qdrant_client=FakeQdrantSimilar(_seed(), hits),
            collection_name="col", seed_track_id="seed", limit=5,
        )
        assert len(out["tracks"]) == 5


# ── stream-scoped tokens (test_stream_token.py) ──────────────────────────────

@pytest.mark.usefixtures("reset_db")
class TestStreamTokenService:
    def test_roundtrip_returns_same_user(self, auth, owner):
        st = auth.issue_stream_token(owner)
        verified = auth.verify_stream_token(st)
        assert verified.id == owner.id
        assert verified.role == owner.role

    def test_rejects_regular_login_token(self, auth, owner):
        login_token = auth.issue_token(owner)
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token(login_token)

    def test_main_verify_rejects_stream_token(self, auth, owner):
        st = auth.issue_stream_token(owner)
        with pytest.raises(InvalidTokenError):
            auth.verify_token(st)

    def test_expired_stream_token_raises(self, auth, owner, monkeypatch):
        st = auth.issue_stream_token(owner)
        import app.services.auth_service as mod
        real_now = mod._now()
        monkeypatch.setattr(
            mod, "_now", lambda: real_now + STREAM_TOKEN_TTL_SECONDS + 60,
        )
        with pytest.raises(TokenExpiredError):
            auth.verify_stream_token(st)

    def test_unknown_user_raises(self, auth, owner):
        st = auth.issue_stream_token(owner)
        MetadataDB.delete_user(owner.id)
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token(st)

    def test_garbage_raises(self, auth):
        with pytest.raises(InvalidTokenError):
            auth.verify_stream_token("garbage.token.here")


@pytest.mark.usefixtures("reset_db")
class TestStreamDependency:
    def test_401_without_header_or_st(self, auth):
        r = TestClient(_make_app(auth)).get("/audio")
        assert r.status_code == 401

    def test_200_with_valid_st_query(self, auth, owner):
        st = auth.issue_stream_token(owner)
        r = TestClient(_make_app(auth)).get(f"/audio?st={st}")
        assert r.status_code == 200
        assert r.json()["id"] == owner.id

    def test_200_with_bearer_header_still_works(self, auth, owner):
        token = auth.issue_token(owner)
        r = TestClient(_make_app(auth)).get(
            "/audio", headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["id"] == owner.id

    def test_401_with_login_token_in_st_query(self, auth, owner):
        login_token = auth.issue_token(owner)
        r = TestClient(_make_app(auth)).get(f"/audio?st={login_token}")
        assert r.status_code == 401

    def test_401_with_garbage_st(self, auth):
        r = TestClient(_make_app(auth)).get("/audio?st=garbage.token.here")
        assert r.status_code == 401


@pytest.mark.usefixtures("reset_db")
class TestStreamTokenRoute:
    def test_requires_auth(self, auth):
        r = TestClient(_make_auth_app(auth)).post("/api/v1/auth/stream-token")
        assert r.status_code == 401

    def test_returns_verifiable_stream_token(self, auth, owner):
        token = auth.issue_token(owner)
        r = TestClient(_make_auth_app(auth)).post(
            "/api/v1/auth/stream-token",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["expires_in"] == STREAM_TOKEN_TTL_SECONDS
        verified = auth.verify_stream_token(body["token"])
        assert verified.id == owner.id
