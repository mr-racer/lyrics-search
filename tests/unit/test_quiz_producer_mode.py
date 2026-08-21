"""M2 «Почерк продюсера» — three tracks share a producer, one does not.

The constraints here are the mode: without them it degenerates. Three tracks by
the same producer from the same album is not a discovery, it is the definition
of an album, and three tracks by one artist is a question about the artist, not
about the production.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M2.
"""
import random

import pytest

from app.services.quiz.context import RoundContext
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.modes import MODES, get_mode
from app.services.quiz.modes import producer as producer_mode

pytestmark = pytest.mark.unit

NOW = 1_000_000.0


def track(i, *, artist=None, album=None, year=2010, genre="rap"):
    name = artist or f"Artist {i}"
    return {
        "track_id": f"t{i}",
        "title": f"Song {i}",
        "artist": name,
        "primary_artist_slug": name.lower().replace(" ", "-"),
        "artists": [name],
        "album": album or f"Album {i}",
        "year": year,
        "genre": genre,
        "duration": 200.0,
        "cover_art_path": f"/covers/t{i}.jpg",
    }


def ctx(tracks, producers, *, seed=0, exclude=None):
    return RoundContext(
        collection_name="acct_1",
        tracks=tracks,
        plays={t["track_id"]: 3 for t in tracks},
        last_played={t["track_id"]: NOW for t in tracks},
        percentiles={t["track_id"]: 50.0 for t in tracks},
        skill={"skill": 0.5, "n_answered": 0, "band_lo": 0.0,
               "band_hi": 100.0, "out_of_band": 0},
        exclude=exclude or set(),
        axis_stats=None,
        producers=producers,
        rng=random.Random(seed),
        now=NOW,
    )


def _healthy():
    """One producer with three tracks, three artists, three albums, plus others."""
    tracks = [track(i, artist=f"Artist {i}", album=f"Album {i}") for i in range(1, 9)]
    producers = {"kanye-west": {"name": "Kanye West", "tracks": ["t1", "t2", "t3"]}}
    return tracks, producers


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_lists_the_producer_mode():
    assert producer_mode.KEY in MODES
    assert get_mode(producer_mode.KEY) is producer_mode


def test_producer_mode_declares_no_audio():
    """There is no snippet here — the client must not draw a play key."""
    assert producer_mode.HAS_AUDIO is False


# ── Pool ─────────────────────────────────────────────────────────────────────

def test_pool_counts_producers_that_can_host_a_round():
    tracks, producers = _healthy()
    assert producer_mode.pool_size(ctx(tracks, producers)) == 1


def test_producer_with_two_tracks_does_not_count():
    tracks, _ = _healthy()
    producers = {"someone": {"name": "Someone", "tracks": ["t1", "t2"]}}
    assert producer_mode.pool_size(ctx(tracks, producers)) == 0


def test_producer_whose_tracks_are_one_album_does_not_count():
    tracks = [track(i, artist=f"Artist {i}", album="One Record") for i in range(1, 9)]
    producers = {"p": {"name": "P", "tracks": ["t1", "t2", "t3"]}}
    assert producer_mode.pool_size(ctx(tracks, producers)) == 0


def test_producer_whose_tracks_are_one_artist_does_not_count():
    tracks = [track(i, artist="Same Artist", album=f"Album {i}") for i in range(1, 9)]
    producers = {"p": {"name": "P", "tracks": ["t1", "t2", "t3"]}}
    assert producer_mode.pool_size(ctx(tracks, producers)) == 0


def test_tracks_missing_from_the_library_are_ignored():
    tracks, _ = _healthy()
    producers = {"p": {"name": "P", "tracks": ["t1", "t2", "gone", "also-gone"]}}
    assert producer_mode.pool_size(ctx(tracks, producers)) == 0


# ── Building ─────────────────────────────────────────────────────────────────

def test_round_offers_four_tracks():
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    assert len(spec.options) == 4


def test_exactly_one_option_is_the_odd_one_out():
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    matching = [o for o in spec.options if o["option_id"] == spec.correct_option_id]
    assert len(matching) == 1


def test_the_answer_is_the_track_the_producer_did_not_make():
    for seed in range(15):
        tracks, producers = _healthy()
        spec = producer_mode.build(ctx(tracks, producers, seed=seed), snippet_sec=3)
        assert spec.track_id not in ("t1", "t2", "t3")


def test_the_three_matching_tracks_come_from_three_artists():
    for seed in range(15):
        tracks, producers = _healthy()
        spec = producer_mode.build(ctx(tracks, producers, seed=seed), snippet_sec=3)
        artists = [o["artist"] for o in spec.options]
        assert len(set(artists)) == 4


def test_options_carry_covers_unlike_the_snippet_mode():
    """M2 asks about production, not recognition — hiding art buys nothing."""
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    assert all("cover_art_path" in o for o in spec.options)


def test_options_never_leak_track_ids():
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    assert all("track_id" not in o for o in spec.options)


def test_the_producer_name_is_withheld_until_the_answer():
    """Naming the producer in the question would give the group away."""
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    stored = spec.to_stored()
    assert stored.get("reveal", {}).get("producer") == "Kanye West"
    assert all("Kanye" not in (o.get("title", "") + o.get("artist", ""))
               for o in spec.options)


# ── Refusing to build ────────────────────────────────────────────────────────

def test_no_credited_producers_means_no_round():
    tracks, _ = _healthy()
    with pytest.raises(NoRoundAvailable):
        producer_mode.build(ctx(tracks, {}), snippet_sec=3)


def test_no_outsider_track_means_no_round():
    """Every track in the library is by this producer — nothing is odd."""
    tracks = [track(i, artist=f"Artist {i}", album=f"Album {i}") for i in range(1, 4)]
    producers = {"p": {"name": "P", "tracks": ["t1", "t2", "t3"]}}
    with pytest.raises(NoRoundAvailable):
        producer_mode.build(ctx(tracks, producers), snippet_sec=3)


def test_anti_repeat_exclusion_is_honoured():
    tracks, producers = _healthy()
    # Everything but t8 is off the table as an answer.
    excluded = {f"t{i}" for i in range(1, 8)}
    spec = producer_mode.build(
        ctx(tracks, producers, exclude=excluded), snippet_sec=3)
    assert spec.track_id == "t8"


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_picking_the_odd_one_out_scores():
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    correct, points = producer_mode.score(
        spec.to_stored(), {"option_id": spec.correct_option_id})
    assert correct is True
    assert points == 100.0


def test_picking_a_matching_track_does_not_score():
    tracks, producers = _healthy()
    spec = producer_mode.build(ctx(tracks, producers), snippet_sec=3)
    wrong = next(o["option_id"] for o in spec.options
                 if o["option_id"] != spec.correct_option_id)
    correct, points = producer_mode.score(spec.to_stored(), {"option_id": wrong})
    assert correct is False
    assert points == 0.0
