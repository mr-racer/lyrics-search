"""Regression: the AI playlist (/recommend/ai-playlist + /stream) must overlay
producer/label/samples credits onto its tracks, like every other endpoint that
returns full TrackMetadata shapes.

Before the fix, `_build_ai_playlist_response` shaped the recsys dicts straight
into AIPlaylistTracks. Those dicts only carry Qdrant-payload fields, and the
credits live in SQLite (`songs`) — so opening an AI playlist in the player
showed no producers chevron and no samples pill. The client-side rescue in
`handlePlayTrack` didn't kick in either: it only enriches shapes with a falsy
`file_path`, and these carry a real one.
"""
import pytest

from app.api.routes.recommend import _build_ai_playlist_response
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _result(**track_extra):
    track = {
        "track_id": "t1", "title": "Stronger", "artist": "Kanye West",
        "album": "Graduation", "year": 2007, "genre": "hip hop",
        "duration": 312.0, "file_path": "/music/stronger.mp3",
        "cover_art_path": None, "tool": "library_search", "reason": "banger",
    }
    track.update(track_extra)
    return {"title": "Workout", "steps": [], "tracks": [track]}


def test_ai_playlist_overlays_extraction_credits(isolated_db):
    # Extraction exists in SQLite but NOT in the recsys dict — exactly the
    # AI-playlist situation. The overlay must still populate producer/samples.
    MetadataDB.add_song_facts_batch(
        "kanye-west-stronger", "colA", ["Produced by Kanye West."],
    )
    MetadataDB.set_song_relations(
        "kanye-west-stronger", '["Kanye West"]',
        '{"samples": [{"song": "HBFS", "artist": "Daft Punk"}], "sampled_by": []}',
    )

    resp = _build_ai_playlist_response(_result())

    assert resp.tracks[0].producer == "Kanye West"
    assert resp.tracks[0].samples == ["Daft Punk — HBFS"]


def test_ai_playlist_no_credits_when_no_extraction(isolated_db):
    resp = _build_ai_playlist_response(_result())

    assert resp.tracks[0].producer is None
    assert resp.tracks[0].samples is None
    # The rest of the shaping is untouched by the overlay.
    assert resp.tracks[0].title == "Stronger"
    assert resp.tracks[0].reason == "banger"
