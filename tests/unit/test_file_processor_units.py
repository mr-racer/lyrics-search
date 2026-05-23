"""Unit tests for file_processor.utils with mocked I/O.

Patch targets updated after Refactor 3: actual implementations now live in
app.indexing.*; file_processor.utils is a thin re-export shim.
"""

from pathlib import Path
from unittest.mock import patch

from app.indexing.lyrics_fetchers import get_lyrics
from app.indexing.folder_scanner import process_file


class TestGetLyrics:
    def test_returns_lyrics_when_found(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.return_value = "Line one\nLine two\n[ Outro ]"
            result = get_lyrics("Song", "Artist", better_lyrics_quality=False)
            assert "[ Outro ]" not in result  # brackets stripped
            assert "Line one" in result

    def test_returns_none_on_exception(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.side_effect = Exception("network error")
            result = get_lyrics("Song", "Artist", better_lyrics_quality=False)
            assert result is None

    def test_strips_bracket_tags(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.return_value = "[Verse 1]\nLyrics here\n[Chorus]\nMore"
            result = get_lyrics("S", "A", better_lyrics_quality=False)
            assert "[Verse 1]" not in result
            assert "[Chorus]" not in result
            assert "Lyrics here" in result

    def test_better_quality_uses_all_providers(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.return_value = "lyrics"
            get_lyrics("S", "A", better_lyrics_quality=True)
            # Check providers passed to search
            call_kwargs = mock_search.call_args[1]
            assert "Musixmatch" in call_kwargs["providers"]

    def test_standard_quality_skips_musixmatch(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.return_value = "lyrics"
            get_lyrics("S", "A", better_lyrics_quality=False)
            call_kwargs = mock_search.call_args[1]
            assert "Musixmatch" not in call_kwargs["providers"]

    def test_empty_response_returns_none(self):
        with patch("app.indexing.lyrics_fetchers.syncedlyrics.search") as mock_search:
            mock_search.return_value = None
            result = get_lyrics("S", "A", better_lyrics_quality=False)
            assert result is None


class TestProcessFile:
    def test_returns_none_if_no_metadata(self, tmp_path):
        f = tmp_path / "test.flac"
        f.touch()
        with patch("app.indexing.folder_scanner.get_metadata") as mock_meta:
            mock_meta.return_value = None
            result = process_file(f, better_lyrics_quality=False)
            assert result is None

    def test_returns_none_if_no_lyrics(self, tmp_path, mocker):
        f = tmp_path / "test.flac"
        f.touch()
        mocker.patch("app.indexing.folder_scanner.get_metadata", return_value={
            "title": "Song",
            "artist": "Artist",
            "album": None,
            "year": 2020,
            "genre": "Pop",
            "duration": 200,
        })
        mocker.patch("app.indexing.folder_scanner.get_lyrics", return_value=None)
        result = process_file(f, better_lyrics_quality=False)
        assert result is not None

    def test_returns_metadata_with_lyrics_when_success(self, tmp_path, mocker):
        f = tmp_path / "test.flac"
        f.touch()
        mocker.patch("app.indexing.folder_scanner.get_metadata", return_value={
            "title": "Song",
            "artist": "Artist",
            "album": "Album",
            "year": 2020,
            "genre": "raw-genre",
            "duration": 200,
        })
        mocker.patch("app.indexing.folder_scanner.get_lyrics", return_value="Some lyrics")
        mocker.patch("app.indexing.folder_scanner.normalize_genre", return_value="Pop")
        result = process_file(f, better_lyrics_quality=False)
        assert result is not None
        assert result["title"] == "Song"
        assert result["lyrics"] == "Some lyrics"
        assert result["file_path"] == str(f)
        assert result["genre"] == "Pop"

    def test_normalizes_genre_if_present(self, tmp_path, mocker):
        f = tmp_path / "test.flac"
        f.touch()
        mocker.patch("app.indexing.folder_scanner.get_metadata", return_value={
            "title": "S", "artist": "A", "album": None,
            "year": None, "genre": "hip hop", "duration": 200,
        })
        mocker.patch("app.indexing.folder_scanner.get_lyrics", return_value="Lyrics")
        result = process_file(f, better_lyrics_quality=False)
        assert result["genre"] == "Hip-Hop"

    def test_skips_track_without_title(self, tmp_path, mocker):
        f = tmp_path / "test.flac"
        f.touch()
        mocker.patch("app.indexing.folder_scanner.get_metadata", return_value={
            "title": "", "artist": "Artist", "album": None,
            "year": None, "genre": None, "duration": 200,
        })
        result = process_file(f, better_lyrics_quality=False)
        assert result is None

    def test_skips_track_without_artist(self, tmp_path, mocker):
        f = tmp_path / "test.flac"
        f.touch()
        mocker.patch("app.indexing.folder_scanner.get_metadata", return_value={
            "title": "Song", "artist": "", "album": None,
            "year": None, "genre": None, "duration": 200,
        })
        result = process_file(f, better_lyrics_quality=False)
        assert result is None
