"""M3 «Слепой год» — name the release year from the sound.

The scoring is the design here. Binary right/wrong for a question whose answer
is a number on a line makes missing by one feel identical to missing by twenty,
which is both wrong and discouraging. Partial credit turns the mode into a
calibration reading instead of a coin toss.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M3.
"""
import random

import pytest

from app.services.quiz.context import RoundContext
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.modes import MODES, get_mode
from app.services.quiz.modes import blind_year

pytestmark = pytest.mark.unit

NOW = 1_000_000.0


def track(i, *, year=2010, duration=200.0):
    return {
        "track_id": f"t{i}",
        "title": f"Song {i}",
        "artist": f"Artist {i}",
        "primary_artist_slug": f"artist-{i}",
        "artists": [f"Artist {i}"],
        "album": f"Album {i}",
        "year": year,
        "genre": "rock",
        "duration": duration,
        "cover_art_path": f"/covers/t{i}.jpg",
    }


def ctx(tracks, *, plays=None, exclude=None, seed=0):
    play_map = plays if plays is not None else {t["track_id"]: 4 for t in tracks}
    return RoundContext(
        collection_name="acct_1",
        tracks=tracks,
        plays=play_map,
        last_played={t["track_id"]: NOW for t in tracks},
        percentiles={t["track_id"]: 50.0 for t in tracks},
        skill={"skill": 0.5, "n_answered": 0, "band_lo": 0.0,
               "band_hi": 100.0, "out_of_band": 0},
        exclude=exclude or set(),
        axis_stats=None,
        rng=random.Random(seed),
        now=NOW,
    )


def _lib(n=25):
    return [track(i, year=1990 + i) for i in range(1, n + 1)]


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_lists_the_blind_year_mode():
    assert blind_year.KEY in MODES
    assert get_mode(blind_year.KEY) is blind_year


def test_blind_year_is_a_listening_round():
    assert blind_year.HAS_AUDIO is True


def test_blind_year_asks_for_a_year_not_an_option():
    assert blind_year.INPUT_KIND == "year"


# ── Pool ─────────────────────────────────────────────────────────────────────

def test_pool_counts_played_tracks_that_have_a_year():
    tracks = _lib(10)
    tracks[0]["year"] = None
    tracks[1]["year"] = None
    assert blind_year.pool_size(ctx(tracks)) == 8


def test_never_played_tracks_do_not_count():
    tracks = _lib(10)
    plays = {t["track_id"]: 0 for t in tracks}
    plays["t1"] = 3
    assert blind_year.pool_size(ctx(tracks, plays=plays)) == 1


# ── Building ─────────────────────────────────────────────────────────────────

def test_round_offers_no_options():
    """The answer is typed on a scale, not chosen from a list."""
    spec = blind_year.build(ctx(_lib()), snippet_sec=3)
    assert spec.options == []


def test_round_hides_the_year_until_it_is_answered():
    spec = blind_year.build(ctx(_lib()), snippet_sec=3)
    stored = spec.to_stored()
    assert stored["reveal"]["year"] == \
        next(t["year"] for t in _lib() if t["track_id"] == spec.track_id)
    # Nothing in the question-side payload may carry it.
    assert "year" not in stored["meta"] or isinstance(stored["meta"].get("year"), type(None))


def test_round_publishes_the_librarys_own_year_range():
    """The scale needs bounds; the library's own span is the honest one and is
    identical for every round, so it gives nothing away."""
    spec = blind_year.build(ctx(_lib(25)), snippet_sec=3)
    meta = spec.to_stored()["meta"]
    assert meta["year_min"] == 1991
    assert meta["year_max"] == 2015


def test_snippet_starts_inside_the_middle_of_the_track():
    for seed in range(20):
        spec = blind_year.build(ctx(_lib(), seed=seed), snippet_sec=3)
        assert 0.15 * 200.0 <= spec.start_sec <= 0.70 * 200.0
        assert spec.length_sec == 3.0


def test_a_track_without_a_year_is_never_asked_about():
    tracks = _lib(25)
    for t in tracks[1:]:
        t["year"] = None
    for seed in range(15):
        spec = blind_year.build(ctx(tracks, seed=seed), snippet_sec=3)
        assert spec.track_id == "t1"


def test_anti_repeat_exclusion_is_honoured():
    tracks = _lib(25)
    excluded = {f"t{i}" for i in range(1, 25)}
    spec = blind_year.build(ctx(tracks, exclude=excluded), snippet_sec=3)
    assert spec.track_id == "t25"


def test_no_dated_played_track_means_no_round():
    tracks = _lib(10)
    for t in tracks:
        t["year"] = None
    with pytest.raises(NoRoundAvailable):
        blind_year.build(ctx(tracks), snippet_sec=3)


# ── Scoring ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("off,points", [
    (0, 100.0), (1, 80.0), (2, 60.0), (3, 30.0), (4, 30.0),
    (5, 0.0), (12, 0.0),
])
def test_points_fall_off_with_distance(off, points):
    spec = {"reveal": {"year": 2000}}
    _, got = blind_year.score(spec, {"year": 2000 + off})
    assert got == points


def test_missing_low_is_scored_like_missing_high():
    spec = {"reveal": {"year": 2000}}
    assert blind_year.score(spec, {"year": 1998})[1] == \
           blind_year.score(spec, {"year": 2002})[1]


def test_within_two_years_counts_as_correct():
    spec = {"reveal": {"year": 2000}}
    assert blind_year.score(spec, {"year": 2002})[0] is True


def test_three_years_off_does_not_count_as_correct():
    spec = {"reveal": {"year": 2000}}
    assert blind_year.score(spec, {"year": 2003})[0] is False


def test_no_answer_scores_zero():
    spec = {"reveal": {"year": 2000}}
    assert blind_year.score(spec, {}) == (False, 0.0)


def test_a_nonsense_answer_scores_zero_instead_of_raising():
    spec = {"reveal": {"year": 2000}}
    assert blind_year.score(spec, {"year": "не знаю"}) == (False, 0.0)


def test_a_round_with_no_stored_year_scores_zero():
    assert blind_year.score({"reveal": {}}, {"year": 2000}) == (False, 0.0)
