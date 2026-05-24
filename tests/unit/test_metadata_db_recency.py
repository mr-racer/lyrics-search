import pytest
from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_recency_map_keeps_latest_per_track():
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t1", played_sec=100.0, total_dur=200.0)
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t1", played_sec=120.0, total_dur=200.0)
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t2", played_sec=90.0, total_dur=200.0)
    m = MetadataDB.get_play_recency_map("c")
    assert set(m.keys()) == {"t1", "t2"}
    assert isinstance(m["t1"], str) and "T" in m["t1"]  # ISO format


def test_recency_map_scoped_by_collection():
    MetadataDB.record_playback_event(session_id="s", collection_name="c1",
                                     track_id="t1", played_sec=100.0, total_dur=200.0)
    assert MetadataDB.get_play_recency_map("c2") == {}
