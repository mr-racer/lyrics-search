from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _pt(tid, artist, album="Al", year=2000, cover="/c.jpg"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {"title": f"T{tid}", "artist": artist, "album": album,
                  "year": year, "duration": 200.0, "cover_art_path": cover, "genre": "rock"}
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    pts = [_pt("1", "Radiohead"), _pt("2", "Radiohead"), _pt("3", "Portishead")]
    qdrant = MagicMock()
    qdrant.scroll.return_value = (pts, None)
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def test_featured_artist_deterministic_per_date(client):
    a = client.get("/api/v1/library/featured-artist",
                   params={"collection_name": "c", "date": "2026-05-24"})
    b = client.get("/api/v1/library/featured-artist",
                   params={"collection_name": "c", "date": "2026-05-24"})
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["slug"] == b.json()["slug"]
    assert a.json()["name"] in {"Radiohead", "Portishead"}


def test_featured_artist_qdrant_down_returns_404(client):
    app.state.db_client = None
    resp = client.get("/api/v1/library/featured-artist", params={"collection_name": "c"})
    assert resp.status_code in (404, 503)
