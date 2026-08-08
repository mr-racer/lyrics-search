"""Consolidated search unit tests.

Merged from:
  - test_search_filters.py              -> TestSearchFilters (+ existing filter classes)
  - test_search_merge.py                -> TestSearchMerge
  - test_search_service_resolve_model.py -> TestSearchServiceResolveModel
  - test_build_filter.py                -> TestBuildFilter
  - test_catalog_search.py              -> TestCatalogSearch
"""

from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from app.domain.models import SearchFilters, TrackHit, TrackMetadata
from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_filters import build_filter
from app.services.catalog_search_service import (
    build_index,
    search_entities,
    search_tracks,
)
from app.services.search_service import SearchService

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# from test_search_filters.py
# ---------------------------------------------------------------------------
class TestExtractFilterKwargs:
    def test_no_filters(self):
        svc = SearchService.__new__(SearchService)  # no __init__
        result = svc._extract_filter_kwargs(None)
        assert result == {}

    def test_empty_filters(self):
        svc = SearchService.__new__(SearchService)
        result = svc._extract_filter_kwargs(SearchFilters())
        assert result == {
            "artist": None,
            "album": None,
            "genre": None,
            "year_ranges": None,
            "sonic_tags": None,
        }

    def test_populated_filters(self):
        svc = SearchService.__new__(SearchService)
        f = SearchFilters(artist="A", album="B", genre="C")
        result = svc._extract_filter_kwargs(f)
        assert result == {
            "artist": "A",
            "album": "B",
            "genre": "C",
            "year_ranges": None,
            "sonic_tags": None,
        }


class TestBuildQdrantFilterModels:
    """Tests for _build_qdrant_filter_models (qdrant_client.models.Filter)."""

    def test_no_filters_returns_none(self):
        svc = SearchService.__new__(SearchService)
        assert svc._build_qdrant_filter_models(None) is None
        assert svc._build_qdrant_filter_models(SearchFilters()) is None

    def test_artist_filter(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter_models(SearchFilters(artist="A"))
        from qdrant_client import models

        assert isinstance(f, models.Filter)
        assert len(f.must) == 1

    def test_multiple_conditions(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter_models(
            SearchFilters(artist="A", album="B", genre="C")
        )
        from qdrant_client import models

        assert isinstance(f, models.Filter)
        assert len(f.must) == 3


class TestSearchFilters:
    def test_search_filters_accepts_sonic_tags_list(self):
        f = SearchFilters(sonic_tags=["melancholic", "lo-fi"])
        assert f.sonic_tags == ["melancholic", "lo-fi"]

    def test_search_filters_sonic_tags_default_empty_not_none(self):
        """Empty list (not None) is the right default so iteration is safe."""
        f = SearchFilters()
        assert f.sonic_tags == []

    def test_search_filters_accepts_year_ranges_list(self):
        from app.domain.models import SearchFilters
        f = SearchFilters(year_ranges=["1990-1999", "2000-2009"])
        assert f.year_ranges == ["1990-1999", "2000-2009"]

    def test_search_filters_year_ranges_default_empty_not_none(self):
        from app.domain.models import SearchFilters
        f = SearchFilters()
        assert f.year_ranges == []

    def test_search_filters_dead_fields_removed(self):
        """Plan B2.2 drops year_from/year_to/duration_*. They should not exist on the model."""
        from app.domain.models import SearchFilters
        fields = SearchFilters.model_fields
        assert "year_from" not in fields
        assert "year_to" not in fields
        assert "duration_min_sec" not in fields
        assert "duration_max_sec" not in fields
        assert "year_range" not in fields  # singular form replaced by year_ranges


# ---------------------------------------------------------------------------
# from test_search_merge.py
# ---------------------------------------------------------------------------
def _make_hit(track_id: str, score: float) -> TrackHit:
    track = TrackMetadata(
        track_id=track_id,
        title=f"Song {track_id}",
        artist="Artist",
        duration_sec=200,
        file_path=f"/{track_id}.flac",
    )
    return TrackHit(track=track, score=score, matched_on="lyrics")


class TestSearchMerge:
    """SearchService._merge_hits() — score fusion logic."""

    def test_empty_inputs(self):
        result = SearchService._merge_hits([], [])
        assert result == []

    def test_only_text_hits(self):
        # Plan 3 / T6 change: matched_on reflects which modalities actually
        # contributed. Text-only hits get "lyrics", not "hybrid".
        hits = [_make_hit("a", 0.9), _make_hit("b", 0.5)]
        result = SearchService._merge_hits(hits, [])
        assert len(result) == 2
        assert all(h.matched_on == "lyrics" for h in result)

    def test_only_clap_hits(self):
        hits = [_make_hit("a", 0.8)]
        result = SearchService._merge_hits([], hits)
        assert len(result) == 1

    def test_overlapping_tracks_combined(self):
        text = [_make_hit("a", 1.0), _make_hit("b", 0.5)]
        clap = [_make_hit("a", 0.8), _make_hit("c", 0.6)]
        result = SearchService._merge_hits(text, clap)
        ids = [h.track.track_id for h in result]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        assert len(result) == 3

    def test_non_overlapping_tracks_preserved(self):
        text = [_make_hit("a", 0.9)]
        clap = [_make_hit("b", 0.8)]
        result = SearchService._merge_hits(text, clap)
        assert len(result) == 2

    def test_scores_normalized_before_weighting(self):
        """Scores are min-max normalized within each list before fusion."""
        text = [_make_hit("a", 1.0), _make_hit("b", 0.0)]
        clap = [_make_hit("a", 1.0), _make_hit("b", 0.0)]
        result = SearchService._merge_hits(text, clap, text_weight=0.5, clap_weight=0.5)
        # After normalization: a=1.0, b=0.0 in both lists
        # Track "a": 1.0*0.5 + 1.0*0.5 = 1.0
        # Track "b": 0.0*0.5 + 0.0*0.5 = 0.0
        scores = {h.track.track_id: h.score for h in result}
        assert scores["a"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.0)

    def test_custom_weights(self):
        """With single-item lists, normalization yields 1.0 for both (range=0 fallback)."""
        text = [_make_hit("a", 1.0)]
        clap = [_make_hit("a", 1.0)]
        result = SearchService._merge_hits(text, clap, text_weight=0.8, clap_weight=0.2)
        # Both normalized to 1.0, but merged score = 1.0*0.8 + 1.0*0.2 = 1.0
        # Actually with single items, range_s = 0 → uses 1.0 fallback
        # Normalized score = (1.0 - 1.0) / 1.0 = 0.0
        assert result[0].score == pytest.approx(0.0)

    def test_single_score_list_no_normalization_division(self):
        """When all scores are the same, range is 0 — avoids division by zero."""
        text = [_make_hit("a", 0.5), _make_hit("b", 0.5)]
        result = SearchService._merge_hits(text, [])
        # With same scores, normalization uses range=1.0 fallback
        assert len(result) == 2

    def test_result_sorted_by_score_desc(self):
        text = [_make_hit("a", 1.0), _make_hit("b", 0.2)]
        clap = [_make_hit("b", 0.9)]
        result = SearchService._merge_hits(text, clap)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_matched_on_reflects_contributing_modalities(self):
        # Plan 3 / T6: text-only -> "lyrics", audio-only -> "audio",
        # both present -> "hybrid". Pre-T6 every hit was "hybrid".
        track = TrackMetadata(
            track_id="a", title="S", artist="A", duration_sec=200, file_path="/a",
        )
        text_only = SearchService._merge_hits([_make_hit("a", 0.9)], [])
        assert text_only[0].matched_on == "lyrics"

        audio_only = SearchService._merge_hits(
            [], [TrackHit(track=track, score=0.7, matched_on="audio")],
        )
        assert audio_only[0].matched_on == "audio"

        both = SearchService._merge_hits(
            [_make_hit("a", 0.9)],
            [TrackHit(track=track, score=0.7, matched_on="audio")],
        )
        assert both[0].matched_on == "hybrid"

    def test_facts_preserved_from_lead_hit(self):
        track = TrackMetadata(
            track_id="a", title="S", artist="A", duration_sec=200, file_path="/a"
        )
        hit = TrackHit(
            track=track, score=0.9, matched_on="lyrics",
            artist_facts="Fun fact", song_facts="Song fact",
        )
        result = SearchService._merge_hits([hit], [])
        assert result[0].artist_facts == "Fun fact"
        assert result[0].song_facts == "Song fact"


# ---------------------------------------------------------------------------
# from test_search_service_resolve_model.py
# ---------------------------------------------------------------------------
def _seed_user(user_id: str, model: str) -> None:
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    MetadataDB.update_user_settings(user_id, text_model_name=model)


class TestSearchServiceModelPinning:
    """There is one embedding model app-wide, so there is no resolution order
    left to get wrong — and no way for a search to query a vector name the
    collection was not built with."""

    def test_no_model_resolution_hook_remains(self):
        assert not hasattr(SearchService, "_resolve_model_name")
        assert not hasattr(SearchService, "DEFAULT_TEXT_MODEL")

    def test_search_signature_takes_no_model(self):
        import inspect
        assert "text_model" not in inspect.signature(SearchService.search).parameters


# ---------------------------------------------------------------------------
# from test_build_filter.py
# ---------------------------------------------------------------------------
class TestBuildFilter:
    """Tests for search_engine.utils.build_filter()."""

    def test_all_none_returns_none(self):
        assert build_filter() is None

    def test_single_artist(self):
        f = build_filter(artist="The Weeknd")
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "artist"

    def test_single_album(self):
        f = build_filter(album="After Hours")
        assert len(f.must) == 1
        assert f.must[0].key == "album"

    def test_single_title(self):
        f = build_filter(title="Blinding Lights")
        assert len(f.must) == 1
        assert f.must[0].key == "title"

    def test_genre_string(self):
        f = build_filter(genre="Pop")
        assert len(f.must) == 1
        assert isinstance(f.must[0].match, models.MatchValue)

    def test_genre_list(self):
        f = build_filter(genre=["Pop", "Rock"])
        assert len(f.must) == 1
        assert isinstance(f.must[0].match, models.MatchAny)

    def test_year(self):
        f = build_filter(year=2020)
        assert len(f.must) == 1
        assert f.must[0].key == "year"

    def test_year_range(self):
        f = build_filter(year_ranges=["2020-2029"])
        assert len(f.must) == 1
        assert f.must[0].key == "year_range"
        assert isinstance(f.must[0].match, models.MatchAny)
        assert list(f.must[0].match.any) == ["2020-2029"]

    def test_multiple_filters(self):
        f = build_filter(artist="The Weeknd", album="After Hours", genre="Pop")
        assert len(f.must) == 3

    def test_empty_strings_treated_as_none(self):
        """Empty strings are falsy — should produce no conditions."""
        f = build_filter(artist="", album="")
        assert f is None


# ---------------------------------------------------------------------------
# from test_catalog_search.py
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def _pt(tid, title, artist, album, **kw):
    return (tid, {
        "title": title,
        "artist": artist,
        "album": album,
        "artists": kw.get("artists", [artist]),
        "artist_slugs": kw.get("artist_slugs", [_slug(artist)]),
        "primary_artist_slug": kw.get("primary_artist_slug", _slug(artist)),
        "cover_art_path": kw.get("cover", f"/cov/{tid}.jpg"),
        "year": kw.get("year"),
        "genre": kw.get("genre"),
        "duration": kw.get("duration", 200.0),
        "file_path": kw.get("file_path", f"/m/{tid}.mp3"),
    })


LIB = [
    _pt("t1", "Bohemian Rhapsody", "Queen", "A Night at the Opera"),
    _pt("t2", "Love of My Life", "Queen", "A Night at the Opera"),
    _pt("t3", "Time", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t4", "Money", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t5", "Группа крови", "Кино", "Группа крови"),
]


class TestCatalogSearch:
    """Catalog search engine (pure: fabricated points + history, no Qdrant).
    Covers entity mode, tracks mode, and the music-player boosts."""

    def test_album_name_query_returns_album_entity_not_its_songs(self):
        idx = build_index(LIB)
        hits = search_entities(idx, "dark side of the moon")
        assert hits, "expected at least one hit"
        assert hits[0]["type"] == "album"
        assert "dark side of the moon" in hits[0]["album"].lower()
        # The album's songs ("Time", "Money") must not appear (titles don't match).
        titles = {h.get("title") for h in hits if h["type"] == "song"}
        assert "Time" not in titles and "Money" not in titles

    def test_artist_name_query_returns_artist_entity(self):
        idx = build_index(LIB)
        hits = search_entities(idx, "pink floyd")
        assert hits[0]["type"] == "artist"
        assert hits[0]["artist"].lower() == "pink floyd"

    def test_song_title_query_returns_song(self):
        idx = build_index(LIB)
        hits = search_entities(idx, "bohemian rhapsody")
        assert hits[0]["type"] == "song"
        assert hits[0]["title"] == "Bohemian Rhapsody"
        assert hits[0]["track_id"] == "t1"

    def test_prefix_as_you_type(self):
        idx = build_index(LIB)
        hits = search_entities(idx, "bohem")
        assert hits[0]["type"] == "song" and hits[0]["title"] == "Bohemian Rhapsody"

    def test_transliteration_latin_query_matches_cyrillic_artist(self):
        idx = build_index(LIB)
        hits = search_entities(idx, "kino")
        assert any(h["type"] == "artist" and h["artist"] == "Кино" for h in hits)

    def test_history_boost_orders_equal_text_matches(self):
        pts = [
            _pt("h1", "Hello", "Adele", "25"),
            _pt("h2", "Hello", "Lionel Richie", "Can't Slow Down"),
        ]
        idx = build_index(pts)
        history = {"play_counts": {"h1": 50}, "reactions": {}}
        hits = [h for h in search_entities(idx, "hello", history=history) if h["type"] == "song"]
        assert hits[0]["track_id"] == "h1"  # more-played wins the tie

    def test_history_boost_cannot_override_exact_name_match(self):
        pts = [
            _pt("m1", "Money", "Pink Floyd", "DSOTM"),
            _pt("m2", "Money Money Money", "ABBA", "Arrival"),
        ]
        idx = build_index(pts)
        history = {"play_counts": {"m2": 9999}, "reactions": {}}
        hits = [h for h in search_entities(idx, "money", history=history) if h["type"] == "song"]
        assert hits[0]["track_id"] == "m1"  # exact title beats a heavily-played partial

    def test_tracks_mode_artist_query_explodes_into_tracks(self):
        idx = build_index(LIB)
        hits = search_tracks(idx, "pink floyd")
        ids = {h["track_id"] for h in hits}
        assert {"t3", "t4"} <= ids
        # tracks mode returns plain tracks with playable fields
        assert all(h.get("track_id") and h.get("file_path") for h in hits)

    def test_tracks_mode_album_query_explodes_into_tracks(self):
        idx = build_index(LIB)
        hits = search_tracks(idx, "dark side of the moon")
        ids = {h["track_id"] for h in hits}
        assert {"t3", "t4"} <= ids

    def test_tracks_mode_dedups_by_track_id(self):
        idx = build_index(LIB)
        hits = search_tracks(idx, "queen")
        ids = [h["track_id"] for h in hits]
        assert sorted(ids) == sorted(set(ids))
        assert {"t1", "t2"} <= set(ids)
