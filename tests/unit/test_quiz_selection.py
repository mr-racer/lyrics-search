"""Answer-track selection: familiarity, skill and the adaptive band.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §8.
"""
import random

import pytest

from app.services.quiz.selection import (
    STARTER_BAND,
    band_for_skill,
    familiarity_percentiles,
    next_band,
    pick_answer_track,
    update_skill,
)

pytestmark = pytest.mark.unit

DAY = 86400.0
NOW = 1_000_000.0


# ── Skill ────────────────────────────────────────────────────────────────────

def test_correct_answer_raises_skill():
    assert update_skill(0.5, True) == pytest.approx(0.625)


def test_wrong_answer_lowers_skill():
    assert update_skill(0.5, False) == pytest.approx(0.375)


def test_skill_stays_inside_zero_one():
    s = 0.5
    for _ in range(50):
        s = update_skill(s, True)
    assert 0.0 <= s <= 1.0


# ── Bands ────────────────────────────────────────────────────────────────────

def test_high_skill_selects_rarely_played_band():
    assert band_for_skill(0.9, n_answered=20, library_size=1000) == (10.0, 45.0)


def test_middling_skill_selects_mid_tail_band():
    assert band_for_skill(0.7, n_answered=20, library_size=1000) == (35.0, 70.0)


def test_low_skill_selects_hits_band():
    assert band_for_skill(0.4, n_answered=20, library_size=1000) == (60.0, 100.0)


def test_band_boundaries_are_inclusive_at_the_bottom():
    assert band_for_skill(0.85, n_answered=20, library_size=1000) == (10.0, 45.0)
    assert band_for_skill(0.60, n_answered=20, library_size=1000) == (35.0, 70.0)


def test_first_five_answers_always_use_the_starter_band():
    assert band_for_skill(0.99, n_answered=4, library_size=1000) == STARTER_BAND


def test_small_library_disables_adaptivity():
    """Below 200 tracks percentiles stop meaning anything (spec §16 R-2)."""
    assert band_for_skill(0.95, n_answered=50, library_size=150) == (0.0, 100.0)


# ── Hysteresis ───────────────────────────────────────────────────────────────

def test_band_holds_until_three_consecutive_out_of_band_answers():
    current = STARTER_BAND
    out = 0
    for expected_out in (1, 2):
        current, out = next_band(current, skill=0.9, n_answered=20,
                                 out_of_band=out, library_size=1000)
        assert current == STARTER_BAND
        assert out == expected_out
    current, out = next_band(current, skill=0.9, n_answered=20,
                             out_of_band=out, library_size=1000)
    assert current == (10.0, 45.0)
    assert out == 0


def test_returning_to_the_current_band_resets_the_counter():
    """A skill back inside the current band clears a partial streak."""
    current, out = next_band(STARTER_BAND, skill=0.4, n_answered=20,
                             out_of_band=2, library_size=1000)
    assert current == STARTER_BAND
    assert out == 0


# ── Familiarity ──────────────────────────────────────────────────────────────

def test_played_track_ranks_above_never_played():
    pct = familiarity_percentiles(
        plays={"a": 10, "b": 0},
        last_played={"a": NOW - DAY, "b": None},
        now=NOW,
    )
    assert pct["a"] > pct["b"]


def test_more_plays_ranks_higher():
    pct = familiarity_percentiles(
        plays={"a": 50, "b": 2},
        last_played={"a": NOW - DAY, "b": NOW - DAY},
        now=NOW,
    )
    assert pct["a"] > pct["b"]


def test_stale_track_ranks_below_a_fresh_one_with_equal_plays():
    pct = familiarity_percentiles(
        plays={"fresh": 10, "stale": 10},
        last_played={"fresh": NOW - DAY, "stale": NOW - 400 * DAY},
        now=NOW,
    )
    assert pct["fresh"] > pct["stale"]


def test_percentiles_span_zero_to_hundred():
    pct = familiarity_percentiles(
        plays={f"t{i}": i for i in range(10)},
        last_played={f"t{i}": NOW - DAY for i in range(10)},
        now=NOW,
    )
    assert min(pct.values()) == pytest.approx(0.0)
    assert max(pct.values()) == pytest.approx(100.0)
    assert all(0.0 <= v <= 100.0 for v in pct.values())


def test_single_track_library_is_top_percentile():
    pct = familiarity_percentiles(
        plays={"only": 3}, last_played={"only": NOW}, now=NOW,
    )
    assert pct["only"] == pytest.approx(100.0)


def test_recency_weight_never_drops_below_half():
    """A very old track keeps half its weight, so plays still dominate."""
    pct = familiarity_percentiles(
        plays={"old_heavy": 1000, "new_light": 1},
        last_played={"old_heavy": NOW - 5000 * DAY, "new_light": NOW},
        now=NOW,
    )
    assert pct["old_heavy"] > pct["new_light"]


# ── Picking ──────────────────────────────────────────────────────────────────

def _lib(n=10):
    plays = {f"t{i}": i + 1 for i in range(n)}
    pct = {f"t{i}": i * 100.0 / (n - 1) for i in range(n)}
    return plays, pct


def test_pick_returns_a_track_inside_the_band():
    plays, pct = _lib()
    got = pick_answer_track(percentiles=pct, band=(60.0, 100.0), plays=plays,
                            exclude=set(), rng=random.Random(0))
    assert got is not None
    assert 60.0 <= pct[got] <= 100.0


def test_pick_never_returns_a_never_played_track():
    plays, pct = _lib()
    plays["t9"] = 0                      # top percentile but never played
    for seed in range(20):
        got = pick_answer_track(percentiles=pct, band=(90.0, 100.0),
                                plays=plays, exclude=set(),
                                rng=random.Random(seed))
        assert got != "t9"


def test_pick_respects_the_exclude_set():
    plays, pct = _lib()
    excluded = {f"t{i}" for i in range(6, 10)}
    for seed in range(20):
        got = pick_answer_track(percentiles=pct, band=(60.0, 100.0),
                                plays=plays, exclude=excluded,
                                rng=random.Random(seed))
        assert got not in excluded


def test_pick_widens_the_band_when_it_is_empty():
    plays, pct = _lib()
    # Percentiles land on 0, 11.1, 22.2, ... 100 — nothing sits in 46..50,
    # so the band has to widen before it can find anyone.
    assert not [t for t, p in pct.items() if 46.0 <= p <= 50.0]
    got = pick_answer_track(percentiles=pct, band=(46.0, 50.0), plays=plays,
                            exclude=set(), rng=random.Random(0))
    assert got is not None


def test_pick_returns_none_when_everything_is_excluded():
    plays, pct = _lib()
    got = pick_answer_track(percentiles=pct, band=(0.0, 100.0), plays=plays,
                            exclude=set(pct), rng=random.Random(0))
    assert got is None


def test_pick_returns_none_when_nothing_was_ever_played():
    plays, pct = _lib()
    plays = {k: 0 for k in plays}
    got = pick_answer_track(percentiles=pct, band=(0.0, 100.0), plays=plays,
                            exclude=set(), rng=random.Random(0))
    assert got is None
