"""Unit tests for autoplay-related accessors and service logic (Task 7 + Task 8)."""

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_get_reactions_for_tracks_returns_mapping():
    MetadataDB.set_reaction("t1", "music", "like")
    MetadataDB.set_reaction("t2", "music", "dislike")
    out = MetadataDB.get_reactions_for_tracks("music", ["t1", "t2", "t3"])
    assert out == {"t1": "like", "t2": "dislike"}


def test_get_reactions_for_tracks_empty_list():
    assert MetadataDB.get_reactions_for_tracks("music", []) == {}


def test_get_reactions_for_tracks_unknown_collection():
    MetadataDB.set_reaction("t1", "music", "like")
    assert MetadataDB.get_reactions_for_tracks("other", ["t1"]) == {}


def test_get_reactions_for_tracks_filters_out_reactionless():
    """Track ids with no reaction row are omitted from the result."""
    MetadataDB.set_reaction("t1", "music", "like")
    out = MetadataDB.get_reactions_for_tracks("music", ["t1", "t2"])
    assert "t1" in out and "t2" not in out


# ---------------------------------------------------------------------------
# Task 8: _apply_filters pipeline tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from app.services import autoplay_service


def _candidate(track_id: str, score: float, artist: str = "A"):
    """Build a Qdrant-style point with payload usable by the service."""
    pt = MagicMock()
    pt.id = track_id
    pt.score = score
    pt.payload = {
        "track_id": track_id,
        "title": f"Title-{track_id}",
        "artist": artist,
        "year": 2020,
        "duration_sec": 200.0,
        "file_path": f"/tmp/{track_id}.mp3",
    }
    return pt


def test_filter_pipeline_drops_seed():
    candidates = [_candidate("seed", 1.0), _candidate("t1", 0.9), _candidate("t2", 0.8)]
    result, diag = autoplay_service._apply_filters(
        candidates,
        seed_track_id="seed",
        exclude_ids=set(),
        dislikes=set(),
        limit=10,
    )
    ids = [t.track_id for t in result]
    assert "seed" not in ids
    assert ids == ["t1", "t2"]
    assert diag.dropped_excluded == 1  # seed counts as excluded


def test_filter_pipeline_drops_excluded():
    candidates = [_candidate("t1", 0.9), _candidate("t2", 0.8), _candidate("t3", 0.7)]
    result, diag = autoplay_service._apply_filters(
        candidates,
        seed_track_id="seed",
        exclude_ids={"t2"},
        dislikes=set(),
        limit=10,
    )
    ids = [t.track_id for t in result]
    assert ids == ["t1", "t3"]
    assert diag.dropped_excluded == 1


def test_filter_pipeline_drops_disliked():
    candidates = [_candidate("t1", 0.9), _candidate("t2", 0.8)]
    result, diag = autoplay_service._apply_filters(
        candidates,
        seed_track_id="seed",
        exclude_ids=set(),
        dislikes={"t1"},
        limit=10,
    )
    ids = [t.track_id for t in result]
    assert ids == ["t2"]
    assert diag.dropped_disliked == 1


def test_filter_pipeline_demotes_repeated_artist():
    cands = [
        _candidate("t1", 0.9, artist="A"),
        _candidate("t2", 0.85, artist="A"),
        _candidate("t3", 0.8, artist="A"),  # 3rd consecutive A — demoted
        _candidate("t4", 0.75, artist="B"),
    ]
    result, diag = autoplay_service._apply_filters(
        cands, seed_track_id="seed", exclude_ids=set(), dislikes=set(), limit=10,
    )
    ids = [t.track_id for t in result]
    # First two A's allowed; t3 (third A) demoted; B (t4) takes its slot; t3 re-inserted at tail
    assert ids[:3] == ["t1", "t2", "t4"]
    assert "t3" in ids


def test_filter_pipeline_respects_limit():
    cands = [_candidate(f"t{i}", 1 - i*0.01) for i in range(20)]
    result, _ = autoplay_service._apply_filters(
        cands, seed_track_id="seed", exclude_ids=set(), dislikes=set(), limit=5,
    )
    assert len(result) == 5


def test_next_queue_handles_seed_without_clap_vector():
    """Legacy tracks (pre-CLAP indexing) have no 'clap' named vector — the
    service must return an empty queue gracefully, not crash."""
    seed_pt = MagicMock()
    seed_pt.id = "seed"
    seed_pt.vector = {}  # no 'clap' key — simulates pre-CLAP indexed track
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [seed_pt]

    result = autoplay_service.next_queue(
        qdrant_client=qdrant,
        collection_name="music",
        seed_track_id="seed",
        exclude_ids=[],
        limit=20,
    )
    assert result.seed_track_id == "seed"
    assert result.tracks == []
    assert result.diagnostics.returned == 0
    qdrant.query_points.assert_not_called()  # crucial: we must not have tried to search


def test_next_queue_handles_seed_not_found():
    """If Qdrant returns nothing for the seed id, return an empty queue."""
    qdrant = MagicMock()
    qdrant.retrieve.return_value = []  # seed missing

    result = autoplay_service.next_queue(
        qdrant_client=qdrant,
        collection_name="music",
        seed_track_id="missing-seed",
        exclude_ids=[],
        limit=20,
    )
    assert result.tracks == []
    qdrant.query_points.assert_not_called()
