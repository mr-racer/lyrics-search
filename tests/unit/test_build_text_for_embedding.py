"""Tests for search_engine.utils.build_text_for_embedding()."""

from app.resources.qdrant_payload import build_text_for_embedding, unique_paragraphs


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

    def test_repeated_chorus_embedded_once(self):
        track = {
            "title": "S", "artist": "A",
            "lyrics": "verse one goes here\n\nchorus line repeated\n\n"
                      "verse two goes here\n\nchorus line repeated",
        }
        result = build_text_for_embedding(track)
        assert result.count("chorus line repeated") == 1
        assert result.index("verse one") < result.index("verse two")

    def test_lyrics_key_absent_does_not_raise(self):
        assert build_text_for_embedding({"title": "S"}) == "title: S"


class TestUniqueParagraphs:
    def test_source_order_is_preserved(self):
        text = "delta\n\nalpha\n\ncharlie\n\nalpha\n\nbravo"
        assert unique_paragraphs(text) == ["delta", "alpha", "charlie", "bravo"]

    def test_crlf_line_endings_still_split(self):
        """82 of 758 prod tracks arrive with CRLF; a bare split("\\n\\n") missed them."""
        text = "first para\r\n\r\nsecond para\r\n\r\nfirst para"
        assert unique_paragraphs(text) == ["first para", "second para"]

    def test_blank_line_holding_whitespace_still_splits(self):
        text = "first para\n   \nsecond para"
        assert unique_paragraphs(text) == ["first para", "second para"]

    def test_dedup_ignores_case_and_inner_whitespace(self):
        text = "Oh-oh   OH\n\noh-oh oh\n\nreal verse"
        assert unique_paragraphs(text) == ["Oh-oh   OH", "real verse"]

    def test_empty_and_none_are_empty_lists(self):
        assert unique_paragraphs("") == []
        assert unique_paragraphs(None) == []

    def test_single_paragraph_survives_untouched(self):
        assert unique_paragraphs("just one block") == ["just one block"]

    def test_deterministic_across_calls(self):
        """The set()-based predecessor reordered 711 of 758 prod tracks."""
        text = "\n\n".join(f"para {i}" for i in range(40))
        assert unique_paragraphs(text) == unique_paragraphs(text)
        assert unique_paragraphs(text)[0] == "para 0"
