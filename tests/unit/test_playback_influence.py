"""influence column migration + persistence + signal passthrough."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_influence_column_exists():
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(playback_events)")]
    assert "influence" in cols


@pytest.mark.parametrize("value,stored", [(True, 1), (False, 0)])
def test_influence_roundtrip(value, stored):
    rid = MetadataDB.record_playback_event(
        session_id="s1", collection_name="colA", track_id="t1",
        played_sec=100.0, total_dur=200.0, influence=value,
    )
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT influence FROM playback_events WHERE id = ?", (rid,)
    ).fetchone()
    assert row[0] == stored


def test_influence_defaults_true_for_legacy_callers():
    MetadataDB.record_playback_event(
        session_id="s1", collection_name="colA", track_id="t1",
        played_sec=50.0, total_dur=None,
    )
    sigs = MetadataDB.get_playback_signals("colA")
    assert sigs and sigs[-1]["influence"] is True


def test_get_playback_signals_reports_influence_false():
    MetadataDB.record_playback_event(
        session_id="s1", collection_name="colA", track_id="t9",
        played_sec=50.0, total_dur=None, influence=False,
    )
    sigs = MetadataDB.get_playback_signals("colA")
    last = [s for s in sigs if s["track_id"] == "t9"][-1]
    assert last["influence"] is False
