from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _pt(tid, title="T", artist="A"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {"title": title, "artist": artist, "album": "Al",
                  "year": 1999, "duration": 200.0, "cover_art_path": f"/c/{tid}.jpg",
                  "genre": "rock"}
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([_pt("t1"), _pt("t2"), _pt("t3")], None)
    qdrant.retrieve.side_effect = lambda **kw: [_pt(kw["ids"][0])]
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def test_rediscover_prefers_never_played(client):
    MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                     track_id="t1", played_sec=150.0, total_dur=200.0)
    resp = client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["never_played"] is True
    assert body["track"]["track_id"] in {"t2", "t3"}


def test_rediscover_qdrant_down_returns_empty(client):
    app.state.db_client = None
    resp = client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
    assert resp.status_code == 200
    assert resp.json()["track"] is None


def test_rediscover_missing_collection_returns_422(client):
    resp = client.get("/api/v1/library/rediscover")
    assert resp.status_code == 422
