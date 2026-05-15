"""Integration test for POST /playback/events."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield TestClient(app)
    MetadataDB._reset_for_tests()


def test_post_playback_event_inserts_row(client):
    resp = client.post("/api/v1/playback/events", json={
        "session_id": "sess-abc",
        "collection_name": "music",
        "track_id": "t-1",
        "played_sec": 180.0,
        "total_dur": 240.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] >= 1

    row = MetadataDB._connect().execute(
        "SELECT session_id, track_id, played_sec, skipped_early FROM playback_events"
    ).fetchone()
    assert row == ("sess-abc", "t-1", 180.0, 0)


def test_post_playback_event_marks_skip(client):
    client.post("/api/v1/playback/events", json={
        "session_id": "sess-abc",
        "collection_name": "music",
        "track_id": "t-1",
        "played_sec": 5.0,
        "total_dur": 240.0,
    })
    row = MetadataDB._connect().execute("SELECT skipped_early FROM playback_events").fetchone()
    assert row[0] == 1


def test_post_playback_event_validation_missing_field(client):
    resp = client.post("/api/v1/playback/events", json={"session_id": "x"})
    assert resp.status_code == 422
