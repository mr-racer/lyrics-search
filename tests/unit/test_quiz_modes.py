"""The mode registry and M1 "What's playing".

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §4, §7.
"""
import random

import pytest

from app.services.quiz.context import RoundContext
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.modes import MODES, get_mode
from app.services.quiz.modes import track_snippet

pytestmark = pytest.mark.unit

AXES = ("energy", "vocal_lead", "spacious", "experimental",
        "brightness", "acousticness")
NOW = 1_000_000.0


def track(i, *, artist=None, album=None, year=2010, genre="rock",
          duration=200.0, plays=5):
    return {
        "track_id": f"t{i}",
        "title": f"Song {i}",
        "artist": artist or f"Artist {i}",
        "primary_artist_slug": (artist or f"Artist {i}").lower(),
        "artists": [artist or f"Artist {i}"],
        "album": album or f"Album {i}",
        "year": year,
        "genre": genre,
        "duration_sec": duration,
        "cover_art_path": f"/covers/t{i}.jpg",
        "sonic_axes": dict(zip(AXES, (0.05 * i, 0.2, 0.3, 0.4, 0.5, 0.6))),
    }


def ctx(n=12, *, plays=None, exclude=None, seed=0):
    tracks = [track(i) for i in range(1, n + 1)]
    play_map = plays if plays is not None else {t["track_id"]: 5 for t in tracks}
    return RoundContext(
        collection_name="acct_1",
        tracks=tracks,
        plays=play_map,
        last_played={t["track_id"]: NOW - 86400.0 for t in tracks},
        percentiles={t["track_id"]: (i * 100.0 / (n - 1))
                     for i, t in enumerate(tracks)},
        skill={"skill": 0.5, "n_answered": 0, "band_lo": 0.0,
               "band_hi": 100.0, "out_of_band": 0},
        exclude=exclude or set(),
        axis_stats=None,
        rng=random.Random(seed),
        now=NOW,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_lists_the_main_mode():
    assert track_snippet.KEY in MODES


def test_registry_entries_expose_the_mode_interface():
    for key, mode in MODES.items():
        assert mode.KEY == key
        assert callable(mode.pool_size)
        assert callable(mode.build)
        assert callable(mode.score)


def test_get_mode_returns_none_for_an_unknown_key():
    assert get_mode("no_such_mode") is None


def test_get_mode_returns_the_module():
    assert get_mode(track_snippet.KEY) is track_snippet


# ── Pool ─────────────────────────────────────────────────────────────────────

def test_pool_size_counts_only_played_tracks():
    c = ctx(10, plays={f"t{i}": (3 if i <= 4 else 0) for i in range(1, 11)})
    assert track_snippet.pool_size(c) == 4


def test_pool_size_is_zero_on_an_unplayed_library():
    c = ctx(10, plays={f"t{i}": 0 for i in range(1, 11)})
    assert track_snippet.pool_size(c) == 0


# ── Building a round ─────────────────────────────────────────────────────────

def test_build_returns_four_options():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    assert len(spec.options) == 4


def test_option_ids_are_unique():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    ids = [o["option_id"] for o in spec.options]
    assert len(set(ids)) == 4


def test_exactly_one_option_is_the_answer():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    matching = [o for o in spec.options if o["option_id"] == spec.correct_option_id]
    assert len(matching) == 1


def test_the_correct_option_describes_the_answer_track():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    answer = next(t for t in ctx().tracks if t["track_id"] == spec.track_id)
    correct = next(o for o in spec.options
                   if o["option_id"] == spec.correct_option_id)
    assert correct["title"] == answer["title"]
    assert correct["artist"] == answer["artist"]


def test_options_do_not_leak_track_ids():
    """A track_id in the options would let a client look the answer up."""
    spec = track_snippet.build(ctx(), snippet_sec=3)
    for option in spec.options:
        assert "track_id" not in option


def test_the_answer_is_not_always_in_the_same_position():
    positions = set()
    for seed in range(30):
        spec = track_snippet.build(ctx(seed=seed), snippet_sec=3)
        ids = [o["option_id"] for o in spec.options]
        positions.add(ids.index(spec.correct_option_id))
    assert len(positions) > 1


def test_no_option_shares_an_artist_or_album_with_the_answer():
    for seed in range(20):
        c = ctx(seed=seed)
        spec = track_snippet.build(c, snippet_sec=3)
        answer = next(t for t in c.tracks if t["track_id"] == spec.track_id)
        wrong = [o for o in spec.options
                 if o["option_id"] != spec.correct_option_id]
        for option in wrong:
            assert option["artist"] != answer["artist"]


# ── The snippet window ───────────────────────────────────────────────────────

def test_start_point_sits_between_15_and_70_percent():
    for seed in range(30):
        spec = track_snippet.build(ctx(seed=seed), snippet_sec=3)
        assert 0.15 * 200.0 <= spec.start_sec <= 0.70 * 200.0


def test_snippet_length_is_what_was_asked_for():
    spec = track_snippet.build(ctx(), snippet_sec=5)
    assert spec.length_sec == 5


def test_snippet_never_runs_past_the_end_of_the_track():
    c = ctx()
    for t in c.tracks:
        t["duration_sec"] = 12.0
    for seed in range(20):
        c2 = ctx(seed=seed)
        for t in c2.tracks:
            t["duration_sec"] = 12.0
        spec = track_snippet.build(c2, snippet_sec=5)
        assert spec.start_sec + spec.length_sec <= 12.0 + 1e-9


# ── Refusing to build ────────────────────────────────────────────────────────

def test_build_refuses_when_no_track_has_been_played():
    c = ctx(10, plays={f"t{i}": 0 for i in range(1, 11)})
    with pytest.raises(NoRoundAvailable):
        track_snippet.build(c, snippet_sec=3)


def test_build_refuses_when_there_are_too_few_distractors():
    """Every candidate shares the answer's artist, so no honest slate exists."""
    c = ctx(4)
    for t in c.tracks:
        t["artist"] = "The Same Band"
        t["primary_artist_slug"] = "the same band"
        t["artists"] = ["The Same Band"]
    with pytest.raises(NoRoundAvailable):
        track_snippet.build(c, snippet_sec=3)


def test_build_honours_the_anti_repeat_exclusion():
    excluded = {f"t{i}" for i in range(1, 12)}   # everything but t12
    c = ctx(12, exclude=excluded)
    spec = track_snippet.build(c, snippet_sec=3)
    assert spec.track_id == "t12"


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_score_accepts_the_correct_option():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    correct, points = track_snippet.score(
        {"correct_option_id": spec.correct_option_id},
        {"option_id": spec.correct_option_id},
    )
    assert correct is True
    assert points == 100.0


def test_score_rejects_a_wrong_option():
    spec = track_snippet.build(ctx(), snippet_sec=3)
    wrong = next(o["option_id"] for o in spec.options
                 if o["option_id"] != spec.correct_option_id)
    correct, points = track_snippet.score(
        {"correct_option_id": spec.correct_option_id},
        {"option_id": wrong},
    )
    assert correct is False
    assert points == 0.0


def test_score_rejects_a_missing_option():
    correct, points = track_snippet.score(
        {"correct_option_id": "abc"}, {},
    )
    assert correct is False
    assert points == 0.0
