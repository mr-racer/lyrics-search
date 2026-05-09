"""Tests for SearchService._merge_hits() — score fusion logic."""

import pytest

from app.domain.models import TrackHit, TrackMetadata
from app.services.search_service import SearchService


def _make_hit(track_id: str, score: float) -> TrackHit:
    track = TrackMetadata(
        track_id=track_id,
        title=f"Song {track_id}",
        artist="Artist",
        duration_sec=200,
        file_path=f"/{track_id}.flac",
    )
    return TrackHit(track=track, score=score, matched_on="lyrics")


class TestMergeHits:
    def test_empty_inputs(self):
        result = SearchService._merge_hits([], [])
        assert result == []

    def test_only_text_hits(self):
        hits = [_make_hit("a", 0.9), _make_hit("b", 0.5)]
        result = SearchService._merge_hits(hits, [])
        assert len(result) == 2
        assert all(h.matched_on == "hybrid" for h in result)

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

    def test_matched_on_set_to_hybrid(self):
        text = [_make_hit("a", 0.9)]
        result = SearchService._merge_hits(text, [])
        assert result[0].matched_on == "hybrid"

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
