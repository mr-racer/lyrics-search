"""Integration tests for SearchService._points_to_hits() with real ScoredPoint objects."""

from qdrant_client.models import ScoredPoint

from app.domain.models import TrackHit
from app.services.search_service import SearchService


class TestPointsToHits:
    def _make_service(self) -> SearchService:
        """Create a SearchService without init (no Qdrant needed)."""
        return SearchService.__new__(SearchService)

    def test_empty_points(self):
        svc = self._make_service()
        result = svc._points_to_hits([], matched_on="lyrics")
        assert result == []

    def test_single_point_converted(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="abc123",
            version=1,
            score=0.85,
            payload={
                "title": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "year": 2020,
                "genre": "Pop",
                "duration": 200,
                "lyrics": "These are test lyrics that are long enough to be meaningful here",
                "file_path": "/test/file.flac",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert len(result) == 1
        hit = result[0]
        assert isinstance(hit, TrackHit)
        assert hit.track.title == "Test Song"
        assert hit.track.artist == "Test Artist"
        assert hit.track.album == "Test Album"
        assert hit.track.year == 2020
        assert hit.score == 0.85
        assert hit.matched_on == "lyrics"

    def test_year_parsed_from_year_range_string(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "year_range": "2010-2019",
                "duration": 180,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.year == 2010

    def test_duration_parsed_from_numeric(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 245,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.duration_sec == 245.0

    def test_lyrics_normalized(self):
        """Lyrics newlines are replaced with spaces."""
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "Line one\nLine two\nLine three of lyrics that is long enough to check",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert "\n" not in (result[0].lyrics or "")

    def test_empty_lyrics_becomes_none(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].lyrics is None

    def test_artist_facts_attached_when_collection_set(self, initialized_db):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "The Weeknd",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits(
            [point], matched_on="lyrics", collection_name="test_collection"
        )
        assert result[0].artist_facts is not None

    def test_song_facts_attached_when_collection_set(self, initialized_db):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "Blinding Lights",
                "artist": "The Weeknd",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits(
            [point], matched_on="lyrics", collection_name="test_collection"
        )
        assert result[0].song_facts is not None

    def test_missing_optional_payload_fields(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 0,
                "lyrics": "",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.album is None
        assert result[0].track.year is None
        assert result[0].track.genre is None

    def test_matched_on_audio(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="audio")
        assert result[0].matched_on == "audio"
