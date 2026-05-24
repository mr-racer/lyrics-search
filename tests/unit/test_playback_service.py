"""Tests for playback_service and the underlying MetadataDB accessor."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.playback_service import get_recent


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


def _stub_qdrant_retrieve(by_id):
    qdrant = MagicMock()
    def retrieve(*, collection_name, ids, with_payload, with_vectors):
        return [
            SimpleNamespace(id=tid, payload=by_id[tid])
            for tid in ids if tid in by_id
        ]
    qdrant.retrieve.side_effect = retrieve
    return qdrant


def test_get_recent_dedups_by_track_id_keeping_latest():
    MetadataDB.record_playback_event(session_id="s1", collection_name="c",
                                     track_id="t1", played_sec=200, total_dur=240)
    MetadataDB.record_playback_event(session_id="s1", collection_name="c",
                                     track_id="t2", played_sec=200, total_dur=240)
    MetadataDB.record_playback_event(session_id="s1", collection_name="c",
                                     track_id="t1", played_sec=200, total_dur=240)
    qdrant = _stub_qdrant_retrieve({
        "t1": {"title": "T1", "artist": "A", "duration": 240},
        "t2": {"title": "T2", "artist": "B", "duration": 240},
    })
    res = get_recent(qdrant_client=qdrant, collection_name="c", limit=10)
    assert len(res.tracks) == 2
    # t1 most recent (had 2 plays), t2 has 1; ordering by last_played DESC means t1 first
    assert res.tracks[0].track_id == "t1"
    assert res.tracks[0].play_count == 2
    assert res.tracks[1].play_count == 1


def test_get_recent_excludes_skipped_from_play_count():
    MetadataDB.record_playback_event(session_id="s1", collection_name="c",
                                     track_id="t1", played_sec=200, total_dur=240)   # not skipped
    MetadataDB.record_playback_event(session_id="s1", collection_name="c",
                                     track_id="t1", played_sec=5, total_dur=240)     # skipped_early
    qdrant = _stub_qdrant_retrieve({"t1": {"title":"T1","artist":"A","duration":240}})
    res = get_recent(qdrant_client=qdrant, collection_name="c", limit=10)
    assert res.tracks[0].play_count == 1   # skipped excluded


def _insert_event_at(played_at_utc: str, track_id: str = "t1"):
    """Insert a non-skipped event with an explicit UTC played_at timestamp."""
    conn = MetadataDB._connect()
    conn.execute(
        "INSERT INTO playback_events "
        "(session_id, collection_name, track_id, played_at, played_sec, total_dur, skipped_early) "
        "VALUES (?, ?, ?, ?, ?, ?, 0)",
        ("s1", "c", track_id, played_at_utc, 200.0, 240.0),
    )
    conn.commit()


def test_get_peak_hour_defaults_to_utc():
    # Event at 22:30 UTC → peak hour 22 with no offset.
    _insert_event_at("2026-05-24 22:30:00")
    assert MetadataDB.get_peak_hour("c") == 22


def test_get_peak_hour_applies_positive_tz_offset():
    # 22:30 UTC + 180 min (UTC+3, Moscow) → 01:30 local → hour 1 (wraps past midnight).
    _insert_event_at("2026-05-24 22:30:00")
    assert MetadataDB.get_peak_hour("c", tz_offset_minutes=180) == 1


def test_get_peak_hour_applies_negative_tz_offset():
    # 02:30 UTC - 300 min (UTC-5) → 21:30 previous day local → hour 21 (wraps back).
    _insert_event_at("2026-05-24 02:30:00")
    assert MetadataDB.get_peak_hour("c", tz_offset_minutes=-300) == 21
