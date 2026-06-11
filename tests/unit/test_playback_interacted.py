"""interacted column migration + persistence through record_playback_event."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _insert(interacted):
    return MetadataDB.record_playback_event(
        session_id="s1", collection_name="colA", track_id="t1",
        played_sec=100.0, total_dur=200.0, interacted=interacted,
    )


def test_interacted_column_exists():
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(playback_events)")]
    assert "interacted" in cols


@pytest.mark.parametrize("value,stored", [(True, 1), (False, 0), (None, None)])
def test_interacted_roundtrip(value, stored):
    row_id = _insert(value)
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT interacted FROM playback_events WHERE id = ?", (row_id,)
    ).fetchone()
    assert row[0] == stored


def test_default_is_null_for_legacy_callers():
    """Callers that don't pass interacted (old paths) keep writing NULL."""
    row_id = MetadataDB.record_playback_event(
        session_id="s1", collection_name="colA", track_id="t1",
        played_sec=10.0, total_dur=None,
    )
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT interacted FROM playback_events WHERE id = ?", (row_id,)
    ).fetchone()
    assert row[0] is None
