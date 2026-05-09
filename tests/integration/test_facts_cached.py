"""Integration tests for facts services with real SQLite backend."""

from app.services.artist_facts_service import (
    get_cached_facts,
    load_all_facts_for_collection,
)
from app.services.song_facts_service import (
    get_cached_song_facts,
    get_song_facts_key,
    load_all_song_facts_for_collection,
)
from app.resources.metadata_db import MetadataDB


class TestArtistFactsCached:
    def test_get_cached_facts_returns_from_db(self, initialized_db):
        result = get_cached_facts("test_collection", "The Weeknd")
        assert result is not None
        assert "Fact 1 about The Weeknd" in result

    def test_get_cached_facts_returns_none_when_missing(self):
        result = get_cached_facts("test_collection", "Unknown Artist")
        assert result is None

    def test_load_all_facts_for_collection(self, initialized_db):
        result = load_all_facts_for_collection("test_collection")
        assert "the-weeknd" in result
        assert "Fact 1 about The Weeknd" in result["the-weeknd"]

    def test_load_all_facts_empty_collection(self):
        result = load_all_facts_for_collection("empty_col")
        assert result == {}


class TestSongFactsCached:
    def test_get_song_facts_key(self, initialized_db):
        key = get_song_facts_key("The Weeknd", "Blinding Lights")
        assert key == "the-weeknd-blinding-lights"

    def test_get_cached_song_facts_returns_from_db(self, initialized_db):
        result = get_cached_song_facts("test_collection", "The Weeknd", "Blinding Lights")
        assert result is not None
        assert "Song fact 1" in result

    def test_get_cached_song_facts_returns_none_when_missing(self):
        result = get_cached_song_facts("col", "A", "Unknown Song")
        assert result is None

    def test_load_all_song_facts_for_collection(self, initialized_db):
        result = load_all_song_facts_for_collection("test_collection")
        assert "the-weeknd-blinding-lights" in result

    def test_load_all_song_facts_empty_collection(self):
        result = load_all_song_facts_for_collection("empty_col")
        assert result == {}


class TestSongFactsKeyConsistency:
    def test_key_matches_slug_format(self):
        """get_song_facts_key should produce the same format as DB slugs."""
        key = get_song_facts_key("The Weeknd", "Blinding Lights")
        assert key == "the-weeknd-blinding-lights"

    def test_key_with_special_chars(self):
        key = get_song_facts_key("AC/DC", "You Shook Me")
        assert key == "ac/dc-you-shook-me"
