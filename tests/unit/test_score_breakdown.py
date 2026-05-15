"""Pydantic model tests for ScoreBreakdown."""

from app.domain.models import ScoreBreakdown, TrackMetadata, TrackHit


def test_score_breakdown_minimal():
    bd = ScoreBreakdown(final_score=0.87)
    assert bd.final_score == 0.87
    assert bd.text_dense_score is None
    assert bd.audio_score is None
    assert bd.weights == {}


def test_score_breakdown_hybrid_populated():
    bd = ScoreBreakdown(
        text_dense_score=0.72,
        text_bm25_score=4.5,
        audio_score=0.81,
        final_score=0.78,
        weights={"text_dense": 0.4, "text_bm25": 0.2, "audio": 0.4},
    )
    assert bd.weights["audio"] == 0.4


def test_track_hit_with_breakdown_serializes():
    hit = TrackHit(
        track=TrackMetadata(
            track_id="t1",
            title="x",
            artist="y",
            duration_sec=180.0,
            file_path="/path/to/file.mp3",
        ),
        score=0.78,
        matched_on="hybrid",
        score_breakdown=ScoreBreakdown(
            text_dense_score=0.72,
            audio_score=0.81,
            final_score=0.78,
            weights={"text_dense": 0.5, "audio": 0.5},
        ),
    )
    dumped = hit.model_dump()
    assert dumped["score_breakdown"]["final_score"] == 0.78
    assert dumped["score_breakdown"]["weights"]["text_dense"] == 0.5


def test_track_hit_without_breakdown_serializes():
    """Backward-compat: existing hits without breakdown still valid."""
    hit = TrackHit(
        track=TrackMetadata(
            track_id="t1",
            title="x",
            artist="y",
            duration_sec=180.0,
            file_path="/path/to/file.mp3",
        ),
        score=0.78,
        matched_on="lyrics",
    )
    dumped = hit.model_dump()
    assert dumped["score_breakdown"] is None
