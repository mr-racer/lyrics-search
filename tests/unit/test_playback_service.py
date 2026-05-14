"""Tests for playback_service and the underlying MetadataDB accessor."""

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_record_playback_event_writes_row():
    MetadataDB.record_playback_event(
        session_id="sess-1",
        collection_name="music",
        track_id="t1",
        played_sec=120.0,
        total_dur=240.0,
    )
    rows = list(MetadataDB._connect().execute(
        "SELECT session_id, collection_name, track_id, played_sec, total_dur, skipped_early "
        "FROM playback_events"
    ))
    assert len(rows) == 1
    assert rows[0] == ("sess-1", "music", "t1", 120.0, 240.0, 0)


def test_skipped_early_is_true_for_short_play_and_low_ratio():
    # played_sec=12 satisfies BOTH conditions: < 30s AND 5% ratio < 30%.
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=12.0, total_dur=240.0,   # 5% — below 30% threshold AND < 30s
    )
    row = MetadataDB._connect().execute("SELECT skipped_early FROM playback_events").fetchone()
    assert row[0] == 1


def test_skipped_early_is_false_for_low_ratio_but_long_play():
    # Low ratio alone (25%) is NOT enough under the AND rule; absolute play > 30s means no skip.
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=60.0, total_dur=240.0,   # 25% ratio but 60s played — not a skip
    )
    row = MetadataDB._connect().execute(
        "SELECT skipped_early FROM playback_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 0


def test_skipped_early_is_false_for_full_play():
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=240.0, total_dur=240.0,
    )
    row = MetadataDB._connect().execute("SELECT skipped_early FROM playback_events").fetchone()
    assert row[0] == 0


def test_skipped_early_false_when_total_dur_missing_but_played_long():
    """If total_dur is None, fall back to the 30-second rule only."""
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=120.0, total_dur=None,
    )
    row = MetadataDB._connect().execute("SELECT skipped_early FROM playback_events").fetchone()
    assert row[0] == 0


def test_skipped_early_true_when_total_dur_missing_and_short():
    """If total_dur is None and played_sec is short, fall back to 30-sec rule (True)."""
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=5.0, total_dur=None,
    )
    row = MetadataDB._connect().execute(
        "SELECT skipped_early FROM playback_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 1


def test_skipped_early_false_for_short_track_fully_played():
    """A 20-second jingle played to completion should NOT be marked as skipped."""
    MetadataDB.record_playback_event(
        session_id="sess-1", collection_name="music", track_id="t1",
        played_sec=20.0, total_dur=20.0,
    )
    row = MetadataDB._connect().execute(
        "SELECT skipped_early FROM playback_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 0
