"""Pydantic model tests for ScoreBreakdown."""

from app.domain.models import ScoreBreakdown, TrackMetadata, TrackHit
from app.services.search_service import SearchService


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


# ── _merge_hits breakdown tests ──

def _make_hit(track_id: str, score: float, matched_on: str = "lyrics") -> TrackHit:
    """Construct a minimal TrackHit for merge testing."""
    return TrackHit(
        track=TrackMetadata(
            track_id=track_id,
            title="title",
            artist="artist",
            duration_sec=180.0,
            file_path=f"/fake/{track_id}.mp3",
        ),
        score=score,
        matched_on=matched_on,
    )


def test_merge_hits_populates_breakdown_for_single_modality_text():
    """Single text hit with no audio — breakdown reflects text_dense and None audio.

    Min-max normalization with two different scores: [0.5, 0.8] → range 0.3.
    The higher score (0.8) normalizes to 1.0.
    """
    text_hits = [_make_hit("t1", 0.8, "lyrics"), _make_hit("t2", 0.5, "lyrics")]
    merged = SearchService._merge_hits(
        text_hits,
        [],
        text_weight=1.0,
        clap_weight=0.0,
    )
    assert len(merged) == 2
    # Top hit (t1, normalized=1.0)
    top = next(h for h in merged if h.track.track_id == "t1")
    bd = top.score_breakdown
    assert bd is not None
    assert bd.text_dense_score == 1.0
    assert bd.audio_score is None
    assert abs(bd.final_score - 1.0) < 1e-9
    assert abs(bd.final_score - top.score) < 1e-9
    assert "text_dense" in bd.weights


def test_merge_hits_combines_text_and_audio():
    """Track present in both modalities — breakdown captures both normalized scores.

    Two-element lists ensure distinct normalized scores.
    text: [t1=0.8, t2=0.5] → t1 normalized 1.0, t2 normalized 0.0
    audio: [t1=0.6, t2=0.3] → t1 normalized 1.0, t2 normalized 0.0
    t1 final = 0.6*1.0 + 0.4*1.0 = 1.0
    """
    text_hits = [_make_hit("t1", 0.8, "lyrics"), _make_hit("t2", 0.5, "lyrics")]
    clap_hits = [_make_hit("t1", 0.6, "audio"), _make_hit("t2", 0.3, "audio")]
    merged = SearchService._merge_hits(
        text_hits,
        clap_hits,
        text_weight=0.6,
        clap_weight=0.4,
    )
    by_id = {h.track.track_id: h for h in merged}
    bd = by_id["t1"].score_breakdown
    assert bd is not None
    assert bd.text_dense_score is not None
    assert bd.audio_score is not None
    assert abs(bd.text_dense_score - 1.0) < 1e-9
    assert abs(bd.audio_score - 1.0) < 1e-9
    # final = 0.6*1.0 + 0.4*1.0 = 1.0
    assert abs(bd.final_score - 1.0) < 1e-9
    assert abs(bd.final_score - by_id["t1"].score) < 1e-9
    assert bd.weights == {"text_dense": 0.6, "audio": 0.4}


def test_merge_hits_track_only_in_one_modality_other_is_none():
    """t1 is text-only, t2 is audio-only — missing modality is None in breakdown."""
    text_hits = [_make_hit("t1", 0.8, "lyrics")]
    clap_hits = [_make_hit("t2", 0.6, "audio")]
    merged = SearchService._merge_hits(
        text_hits,
        clap_hits,
        text_weight=0.5,
        clap_weight=0.5,
    )
    assert len(merged) == 2
    by_id = {h.track.track_id: h for h in merged}

    bd1 = by_id["t1"].score_breakdown
    assert bd1 is not None
    assert bd1.text_dense_score is not None
    assert bd1.audio_score is None

    bd2 = by_id["t2"].score_breakdown
    assert bd2 is not None
    assert bd2.text_dense_score is None
    assert bd2.audio_score is not None
