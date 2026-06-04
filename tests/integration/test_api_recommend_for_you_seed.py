from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _pt(tid):
    pt = MagicMock(); pt.id = tid
    pt.payload = {"title": f"T{tid}", "artist": "A", "cover_art_path": f"/{tid}.jpg"}
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([_pt("r1"), _pt("r2")], None)
    qdrant.retrieve.side_effect = lambda **kw: [_pt(kw["ids"][0])]
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def test_for_you_seed_returns_seed(client):
    resp = client.get("/api/v1/recommend/for-you-seed", params={"collection": "c"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["seed_track_id"] in {"r1", "r2"}
    assert body["track"]["track_id"] == body["seed_track_id"]


def test_for_you_seed_db_down_returns_503(client):
    app.state.db_client = None
    resp = client.get("/api/v1/recommend/for-you-seed", params={"collection": "c"})
    assert resp.status_code == 503
