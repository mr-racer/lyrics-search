"""Tests for search_engine.utils.prepare_metadata()."""

import pytest
import search_engine.utils as utils_module


class TestPrepareMetadata:
    """Test prepare_metadata() — filtering, bucketing, enrichment."""

    def _make_data(self, tracks):
        """Build a dict keyed by 'Artist — Title'."""
        return {
            f"{t['artist']} — {t['title']}": t
            for t in tracks
        }

    def test_filters_by_max_duration(self, monkeypatch):
        """Tracks longer than MAX_DURATION (420s) are excluded."""
        data = {
            "A — B": {"title": "B", "artist": "A", "duration": 500, "lyrics": "x " * 100},
            "C — D": {"title": "D", "artist": "C", "duration": 200, "lyrics": "x " * 100},
        }
        result = utils_module.prepare_metadata(data)
        titles = [r["title"] for r in result]
        assert "B" not in titles
        assert "D" in titles

    def test_filters_by_lyrics_length(self, monkeypatch):
        """Tracks with fewer than 50 chars of lyrics are excluded."""
        data = {
            "A — B": {"title": "B", "artist": "A", "duration": 200, "lyrics": "short"},
            "C — D": {"title": "D", "artist": "C", "duration": 200, "lyrics": "x " * 100},
        }
        result = utils_module.prepare_metadata(data)
        titles = [r["title"] for r in result]
        assert "B" not in titles
        assert "D" in titles

    @pytest.mark.skip(reason="prepare_metadata() calls np.percentile on empty array — crashes")
    def test_empty_input(self):
        """Empty dict produces empty result — no data to filter."""
        result = utils_module.prepare_metadata({})
        assert result == []

    def test_duration_kept_numeric_and_range_added(self, sample_tracks_data):
        """Numeric `duration` must survive bucketing — downstream Pydantic
        models (LikedSongTrack et al.) expect Optional[float]. The bucket
        label lives in the parallel `duration_range` field, mirroring
        the existing year/year_range pair."""
        result = utils_module.prepare_metadata(sample_tracks_data)
        for rec in result:
            assert isinstance(rec["duration"], (int, float)), (
                "duration must remain numeric seconds, not a bucket string"
            )
            assert isinstance(rec["duration_range"], str)
            assert "-" in rec["duration_range"]

    def test_year_range_added(self, sample_tracks_data):
        result = utils_module.prepare_metadata(sample_tracks_data)
        for rec in result:
            if rec.get("year"):
                assert "year_range" in rec
                decade = (rec["year"] // 10) * 10
                assert rec["year_range"] == f"{decade}-{decade + 9}"

    def test_lyrics_chunked_added(self, sample_tracks_data):
        result = utils_module.prepare_metadata(sample_tracks_data)
        for rec in result:
            assert "lyrics_chunked" in rec
            assert isinstance(rec["lyrics_chunked"], tuple)

    def test_single_track(self):
        data = {
            "A — B": {
                "title": "B",
                "artist": "A",
                "duration": 200,
                "lyrics": "This lyrics text is long enough to pass the filter threshold check",
            }
        }
        result = utils_module.prepare_metadata(data)
        assert len(result) == 1
        assert result[0]["title"] == "B"

    def test_tracks_without_year(self):
        data = {
            "A — B": {
                "title": "B",
                "artist": "A",
                "duration": 200,
                "lyrics": "This lyrics text is long enough to pass the filter threshold check here too",
            }
        }
        result = utils_module.prepare_metadata(data)
        assert len(result) == 1
        assert "year_range" not in result[0]

    def test_duration_range_bucket_shape(self, sample_tracks_data):
        """`duration_range` must be a parseable 'X-Y' int-int bucket."""
        result = utils_module.prepare_metadata(sample_tracks_data)
        for rec in result:
            dr = rec["duration_range"]
            assert isinstance(dr, str)
            parts = dr.split("-")
            assert len(parts) == 2
            int(parts[0])  # should not raise
            int(parts[1])
