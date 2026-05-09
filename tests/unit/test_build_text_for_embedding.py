"""Tests for search_engine.utils.build_text_for_embedding()."""

from search_engine.utils import build_text_for_embedding


class TestBuildTextForEmbedding:
    def test_full_track(self):
        track = {
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "album": "After Hours",
            "genre": "Pop",
            "lyrics": "These lyrics are long enough to be included in the embedding text",
        }
        result = build_text_for_embedding(track)
        assert "title: Blinding Lights" in result
        assert "artist: The Weeknd" in result
        assert "album: After Hours" in result
        assert "genre: Pop" in result
        assert "These lyrics are long enough" in result

    def test_missing_album_skipped(self):
        track = {
            "title": "Song",
            "artist": "Artist",
            "genre": "Pop",
            "lyrics": "Some lyrics text that is long enough to pass the minimum threshold check",
        }
        result = build_text_for_embedding(track)
        assert "album:" not in result

    def test_missing_genre_skipped(self):
        track = {
            "title": "Song",
            "artist": "Artist",
            "album": "Album",
            "lyrics": "Some lyrics text that is long enough to pass the minimum threshold check",
        }
        result = build_text_for_embedding(track)
        assert "genre:" not in result

    def test_short_lyrics_excluded(self):
        track = {
            "title": "Song",
            "artist": "Artist",
            "lyrics": "short",
        }
        result = build_text_for_embedding(track)
        assert "short" not in result

    def test_empty_track(self):
        result = build_text_for_embedding({})
        assert result == ""

    def test_only_title(self):
        track = {"title": "Hello"}
        result = build_text_for_embedding(track)
        assert result == "title: Hello"

    def test_pipe_separator(self):
        track = {
            "title": "A",
            "artist": "B",
            "lyrics": "This lyrics string is definitely long enough to be included in the output",
        }
        result = build_text_for_embedding(track)
        assert " | " in result

    def test_lyrics_exactly_at_threshold(self):
        """Lyrics of exactly 20 chars should be excluded (> 20 required)."""
        track = {"title": "S", "artist": "A", "lyrics": "12345678901234567890"}
        result = build_text_for_embedding(track)
        assert "12345678901234567890" not in result

    def test_lyrics_just_above_threshold(self):
        """Lyrics of 21 chars should be included."""
        track = {"title": "S", "artist": "A", "lyrics": "123456789012345678901"}
        result = build_text_for_embedding(track)
        assert "123456789012345678901" in result
