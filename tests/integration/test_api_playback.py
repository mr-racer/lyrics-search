"""Integration test for POST /playback/events."""

import json
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.api.routes import playback as pb_route
from app.resources.metadata_db import MetadataDB
from app.services import playback_service
from ._auth_helper import authenticate_test_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
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


def test_playback_events_ignores_supplied_collection():
    """D-soft: even when client supplies collection_name='acct_BAD', the server
    derives the real collection from the JWT user and ignores the supplied value."""
    captured = {}

    def fake_record(*, session_id, collection_name, track_id, played_sec, total_dur,
                    interacted=None, influence=True):
        captured["collection_name"] = collection_name
        return 1

    orig = playback_service.record_event
    playback_service.record_event = fake_record
    try:
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[pb_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            c.post(
                "/api/v1/playback/events",
                data=json.dumps({
                    "session_id": "S1",
                    "collection_name": "acct_BAD",
                    "track_id": "T1",
                    "played_sec": 30.0,
                }),
                headers={"Content-Type": "text/plain"},
            )
        assert captured["collection_name"] == "acct_user-A"
    finally:
        playback_service.record_event = orig
