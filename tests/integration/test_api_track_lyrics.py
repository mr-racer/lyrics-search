"""Integration tests for GET /search/tracks/{track_id}/lyrics.

The player reads full lyrics on demand by track_id so EVERY playback source
(search, autoplay queue, recently-played, liked, stream) shows lyrics — not
just search results, which were the only payloads that ever carried the
`lyrics` field inline.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _mk_point(track_id, lyrics):
    pt = MagicMock()
    pt.id = track_id
    pt.payload = {
        "track_id": track_id,
        "title": f"Title-{track_id}",
        "artist": "A",
        "lyrics": lyrics,
        "file_path": f"/tmp/{track_id}.mp3",
    }
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_mk_point("t1", "line one\nline two")]
    db_stub = MagicMock()
    db_stub.qdrant = qdrant
    app.state.db_client = db_stub

    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()


def test_lyrics_returns_full_text(client):
    resp = client.get("/api/v1/search/tracks/t1/lyrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["track_id"] == "t1"
    assert body["lyrics"] == "line one\nline two"


def test_lyrics_null_when_track_has_none(client):
    client.app.state.db_client.qdrant.retrieve.return_value = [_mk_point("t2", "")]
    resp = client.get("/api/v1/search/tracks/t2/lyrics")
    assert resp.status_code == 200
    assert resp.json()["lyrics"] is None


def test_lyrics_404_when_track_missing(client):
    client.app.state.db_client.qdrant.retrieve.return_value = []
    resp = client.get("/api/v1/search/tracks/nope/lyrics")
    assert resp.status_code == 404


def test_lyrics_requires_auth():
    app.state.db_client = MagicMock()
    c = TestClient(app)  # no authenticate_test_client → no Bearer token
    resp = c.get("/api/v1/search/tracks/t1/lyrics")
    assert resp.status_code == 401
