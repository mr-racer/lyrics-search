"""Tests for search_engine.utils.build_filter()."""

from qdrant_client import models

from app.resources.qdrant_filters import build_filter


class TestBuildFilter:
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
