"""M4 «Родословная» — which record was this track built on?

The mode teaches rather than tests: most people cannot answer it cold, and the
reveal is the point. Its constraint is that a distractor must not itself be
connected to the track by any link — otherwise there are two true answers and
the round is broken rather than hard.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M4.
"""
import random

import pytest

from app.services.quiz.context import RoundContext
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.modes import MODES, get_mode
from app.services.quiz.modes import lineage

pytestmark = pytest.mark.unit

NOW = 1_000_000.0


def track(i, *, year=2005):
    return {
        "track_id": f"t{i}",
        "title": f"Song {i}",
        "artist": f"Artist {i}",
        "primary_artist_slug": f"artist-{i}",
        "artists": [f"Artist {i}"],
        "album": f"Album {i}",
        "year": year,
        "genre": "rap",
        "duration": 200.0,
        "cover_art_path": f"/covers/t{i}.jpg",
    }


def link(src, title, artist, *, relation="sample"):
    return {"src_track_id": src, "dst_title": title, "dst_artist": artist,
            "dst_slug": None, "relation": relation}


def ctx(tracks, links, *, seed=0, exclude=None):
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
        sample_links=links,
        rng=random.Random(seed),
        now=NOW,
    )


def _healthy(n=8):
    tracks = [track(i) for i in range(1, n + 1)]
    links = [link(f"t{i}", f"Source {i}", f"Source Artist {i}")
             for i in range(1, n + 1)]
    return tracks, links


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_lists_the_lineage_mode():
    assert lineage.KEY in MODES
    assert get_mode(lineage.KEY) is lineage


def test_lineage_is_a_listening_round_answered_with_options():
    assert lineage.HAS_AUDIO is True
    assert lineage.INPUT_KIND == "options"


# ── Pool ─────────────────────────────────────────────────────────────────────

def test_pool_counts_links_whose_track_is_in_the_library():
    tracks, links = _healthy(5)
    assert lineage.pool_size(ctx(tracks, links)) == 5


def test_a_link_pointing_at_a_missing_track_does_not_count():
    tracks, _ = _healthy(3)
    links = [link("gone", "Source", "Somebody")]
    assert lineage.pool_size(ctx(tracks, links)) == 0


def test_a_link_without_a_named_source_does_not_count():
    tracks, _ = _healthy(3)
    links = [link("t1", "", "Somebody"), link("t2", "Source", "")]
    assert lineage.pool_size(ctx(tracks, links)) == 0


# ── Building ─────────────────────────────────────────────────────────────────

def test_round_offers_four_sources():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    assert len(spec.options) == 4


def test_options_are_sources_not_library_tracks():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    assert all(o["artist"].startswith("Source Artist") for o in spec.options)


def test_the_question_names_the_track_being_asked_about():
    """You cannot answer 'what did this sample' without knowing what 'this' is."""
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    meta = spec.to_stored()["meta"]
    answer_track = next(t for t in tracks if t["track_id"] == spec.track_id)
    assert meta["prompt"]["title"] == answer_track["title"]
    assert meta["prompt"]["artist"] == answer_track["artist"]
    assert meta["prompt"]["cover_art_path"] == answer_track["cover_art_path"]


def test_the_correct_option_is_the_real_source():
    for seed in range(15):
        tracks, links = _healthy()
        c = ctx(tracks, links, seed=seed)
        spec = lineage.build(c, snippet_sec=3)
        truth = next(l for l in links if l["src_track_id"] == spec.track_id)
        correct = next(o for o in spec.options
                       if o["option_id"] == spec.correct_option_id)
        assert correct["title"] == truth["dst_title"]
        assert correct["artist"] == truth["dst_artist"]


def test_no_distractor_is_also_a_source_of_the_same_track():
    """Two true answers is a broken round, not a hard one."""
    tracks = [track(i) for i in range(1, 9)]
    links = [link("t1", "Source A", "Artist A"), link("t1", "Source B", "Artist B")]
    links += [link(f"t{i}", f"Source {i}", f"Artist {i}") for i in range(2, 9)]
    for seed in range(20):
        c = ctx(tracks, links, seed=seed)
        spec = lineage.build(c, snippet_sec=3)
        if spec.track_id != "t1":
            continue
        titles = {o["title"] for o in spec.options}
        assert not {"Source A", "Source B"} <= titles


def test_options_never_leak_track_ids():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    assert all("track_id" not in o for o in spec.options)


def test_the_relation_is_revealed_only_after_the_answer():
    tracks = [track(1)] + [track(i) for i in range(2, 9)]
    links = [link("t1", "Source 1", "Artist 1", relation="interpolation")]
    links += [link(f"t{i}", f"Source {i}", f"Artist {i}") for i in range(2, 9)]
    for seed in range(20):
        spec = lineage.build(ctx(tracks, links, seed=seed), snippet_sec=3)
        if spec.track_id == "t1":
            assert spec.to_stored()["reveal"]["relation"] == "interpolation"
            return
    pytest.fail("t1 was never chosen across 20 seeds")


def test_snippet_window_matches_the_other_listening_modes():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    assert 0.15 * 200.0 <= spec.start_sec <= 0.70 * 200.0
    assert spec.length_sec == 3.0


# ── Refusing to build ────────────────────────────────────────────────────────

def test_no_links_means_no_round():
    tracks, _ = _healthy()
    with pytest.raises(NoRoundAvailable):
        lineage.build(ctx(tracks, []), snippet_sec=3)


def test_too_few_distinct_sources_means_no_round():
    """Four options need four different sources to choose between."""
    tracks, _ = _healthy()
    links = [link("t1", "Only Source", "Only Artist")]
    with pytest.raises(NoRoundAvailable):
        lineage.build(ctx(tracks, links), snippet_sec=3)


def test_anti_repeat_exclusion_is_honoured():
    tracks, links = _healthy()
    excluded = {f"t{i}" for i in range(1, 8)}
    spec = lineage.build(ctx(tracks, links, exclude=excluded), snippet_sec=3)
    assert spec.track_id == "t8"


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_picking_the_real_source_scores():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    assert lineage.score(spec.to_stored(),
                         {"option_id": spec.correct_option_id}) == (True, 100.0)


def test_picking_another_source_does_not_score():
    tracks, links = _healthy()
    spec = lineage.build(ctx(tracks, links), snippet_sec=3)
    wrong = next(o["option_id"] for o in spec.options
                 if o["option_id"] != spec.correct_option_id)
    assert lineage.score(spec.to_stored(), {"option_id": wrong}) == (False, 0.0)
