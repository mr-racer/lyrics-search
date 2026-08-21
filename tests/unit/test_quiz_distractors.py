"""Distractor selection and the I-3 exclusion.

CLAP over-weights vocals, so a track's nearest sonic neighbours are almost
always the same performer or the same record. Every test here exists because
the naive "take the three closest" slate would be unplayable.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §9.
"""
import random

import pytest

from app.services.quiz.distractors import (
    axis_distance,
    pick_distractors,
    shares_artist_or_album,
)

pytestmark = pytest.mark.unit

AXES = ("energy", "vocal_lead", "spacious", "experimental",
        "brightness", "acousticness")


def track(track_id, *, artist="Someone", slug=None, album="Some Album",
          year=2010, genre="rock", artists=None, axes=None, title=None):
    return {
        "track_id": track_id,
        "title": title or f"Song {track_id}",
        "artist": artist,
        "primary_artist_slug": slug if slug is not None else artist.lower(),
        "artists": artists if artists is not None else [artist],
        "album": album,
        "year": year,
        "genre": genre,
        "sonic_axes": dict(zip(AXES, axes)) if axes else None,
    }


# ── I-3: the exclusion rule ──────────────────────────────────────────────────

def test_same_primary_artist_is_shared():
    a = track("1", artist="Kanye West")
    b = track("2", artist="Kanye West", album="Other")
    assert shares_artist_or_album(a, b) is True


def test_feature_overlap_is_shared():
    a = track("1", artist="Jay-Z", artists=["Jay-Z", "Rihanna"], album="A")
    b = track("2", artist="Rihanna", artists=["Rihanna"], album="B")
    assert shares_artist_or_album(a, b) is True


def test_same_album_is_shared():
    a = track("1", artist="A", album="Graduation")
    b = track("2", artist="B", album="Graduation")
    assert shares_artist_or_album(a, b) is True


def test_album_match_ignores_case_and_padding():
    a = track("1", artist="A", album="Graduation")
    b = track("2", artist="B", album="  graduation ")
    assert shares_artist_or_album(a, b) is True


def test_missing_album_on_both_sides_is_not_a_match():
    a = track("1", artist="A", album=None)
    b = track("2", artist="B", album="")
    assert shares_artist_or_album(a, b) is False


def test_unrelated_tracks_are_not_shared():
    a = track("1", artist="A", album="X")
    b = track("2", artist="B", album="Y")
    assert shares_artist_or_album(a, b) is False


# ── The watchdog ─────────────────────────────────────────────────────────────

def test_nearest_axis_neighbour_is_dropped_when_it_is_the_same_artist():
    """The exact failure mode I-3 exists for.

    The sonically closest track is by the same artist — which is what CLAP
    keeps handing back. It must never reach the slate, no matter how close.
    """
    answer = track("ans", artist="Burial", album="Untrue",
                   axes=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    twin = track("twin", artist="Burial", album="Kindred",
                 axes=(0.5, 0.5, 0.5, 0.5, 0.5, 0.51))   # nearest by far
    others = [
        track(f"o{i}", artist=f"Artist{i}", album=f"Album{i}", year=2000 + i,
              genre="electronic", axes=(0.1 * i, 0.2, 0.3, 0.4, 0.5, 0.6))
        for i in range(1, 8)
    ]
    for seed in range(25):
        slate = pick_distractors(
            answer=answer, candidates=[twin] + others, n=3,
            axis_stats=None, rng=random.Random(seed),
        )
        assert all(t["track_id"] != "twin" for t in slate)


# ── Slate composition ────────────────────────────────────────────────────────

def _pool(n=12):
    return [
        track(f"o{i}", artist=f"Artist{i}", album=f"Album{i}",
              year=2000 + i, genre="rock" if i % 2 else "pop",
              axes=(0.05 * i, 0.2, 0.3, 0.4, 0.5, 0.6))
        for i in range(1, n + 1)
    ]


def test_slate_has_three_distinct_tracks():
    answer = track("ans", artist="Answer", album="AnsAlbum",
                   axes=(0.5, 0.2, 0.3, 0.4, 0.5, 0.6))
    slate = pick_distractors(answer=answer, candidates=_pool(), n=3,
                             axis_stats=None, rng=random.Random(1))
    ids = [t["track_id"] for t in slate]
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_slate_never_contains_the_answer():
    answer = track("ans", artist="Answer", album="AnsAlbum",
                   axes=(0.5, 0.2, 0.3, 0.4, 0.5, 0.6))
    candidates = _pool() + [answer]
    for seed in range(15):
        slate = pick_distractors(answer=answer, candidates=candidates, n=3,
                                 axis_stats=None, rng=random.Random(seed))
        assert all(t["track_id"] != "ans" for t in slate)


def test_slate_works_without_any_sonic_axes():
    """Libraries indexed without CLAP must still produce rounds (spec §9)."""
    answer = track("ans", artist="Answer", album="AnsAlbum", year=2005,
                   genre="rock", axes=None)
    candidates = [
        track(f"o{i}", artist=f"Artist{i}", album=f"Album{i}",
              year=2003 + i, genre="rock", axes=None)
        for i in range(1, 8)
    ]
    slate = pick_distractors(answer=answer, candidates=candidates, n=3,
                             axis_stats=None, rng=random.Random(3))
    assert len(slate) == 3


def test_returns_fewer_when_the_pool_is_too_small():
    answer = track("ans", artist="Answer", album="AnsAlbum")
    candidates = [track("o1", artist="A1", album="Al1")]
    slate = pick_distractors(answer=answer, candidates=candidates, n=3,
                             axis_stats=None, rng=random.Random(0))
    assert len(slate) == 1


def test_returns_empty_when_every_candidate_is_excluded():
    answer = track("ans", artist="Solo", album="OnlyAlbum")
    candidates = [track("o1", artist="Solo", album="Another"),
                  track("o2", artist="Other", album="OnlyAlbum")]
    slate = pick_distractors(answer=answer, candidates=candidates, n=3,
                             axis_stats=None, rng=random.Random(0))
    assert slate == []


# ── Axis distance ────────────────────────────────────────────────────────────

def test_axis_distance_is_none_when_either_side_lacks_axes():
    a = track("1", axes=(0.1,) * 6)
    b = track("2", axes=None)
    assert axis_distance(a, b, None) is None
    assert axis_distance(b, a, None) is None


def test_axis_distance_is_zero_for_identical_axes():
    a = track("1", axes=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    b = track("2", axes=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    assert axis_distance(a, b, None) == pytest.approx(0.0)


def test_axis_distance_grows_with_separation():
    a = track("1", axes=(0.0,) * 6)
    near = track("2", axes=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
    far = track("3", axes=(0.9, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert axis_distance(a, near, None) < axis_distance(a, far, None)


def test_zscore_stats_rescale_the_distance():
    """An axis with a tiny spread should count for more once normalised."""
    a = track("1", axes=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b = track("2", axes=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
    stats = {
        "mean": {name: 0.0 for name in AXES},
        "std": {name: (0.01 if name == "energy" else 1.0) for name in AXES},
    }
    raw = axis_distance(a, b, None)
    scaled = axis_distance(a, b, stats)
    assert scaled > raw
