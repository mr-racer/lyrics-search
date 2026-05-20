"""Unit tests for LibraryService.get_listening_stats."""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.resources.metadata_db import MetadataDB
from app.services.library_service import LibraryService


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _qdrant_with(by_id):
    qdrant = MagicMock()
    qdrant.retrieve.side_effect = lambda **kw: [
        SimpleNamespace(id=tid, payload=by_id[tid]) for tid in kw["ids"] if tid in by_id
    ]
    return qdrant


def test_listening_stats_empty_when_no_events():
    qdrant = _qdrant_with({})
    res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
    assert res.total_seconds_listened == 0
    assert res.top_track is None
    assert res.top_artist is None
    assert res.peak_hour is None


def test_listening_stats_sums_played_sec():
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t1", played_sec=120, total_dur=240)
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t2", played_sec=60, total_dur=240)
    qdrant = _qdrant_with({
        "t1": {"title": "T1", "artist": "A", "duration": 240},
        "t2": {"title": "T2", "artist": "B", "duration": 240},
    })
    res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
    assert res.total_seconds_listened == 180


def test_listening_stats_top_track_excludes_skips():
    # t1 has 2 non-skip + 1 skip; t2 has 3 skips, 0 valid plays → top should be t1
    for _ in range(2):
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t1", played_sec=200, total_dur=240)
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t1", played_sec=3, total_dur=240)
    for _ in range(3):
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t2", played_sec=3, total_dur=240)
    qdrant = _qdrant_with({
        "t1": {"title": "T1", "artist": "Radiohead", "duration": 240},
        "t2": {"title": "T2", "artist": "Other", "duration": 240},
    })
    res = LibraryService.get_listening_stats(qdrant_client=qdrant, collection_name="c", lang="en")
    assert res.top_track.track_id == "t1"
    assert res.top_track.play_count == 2
    assert res.top_artist.name == "Radiohead"
