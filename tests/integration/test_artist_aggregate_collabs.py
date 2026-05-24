"""Opening an artist surfaces their collaborations; non-participants excluded."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.metadata_db import MetadataDB


def _pt(tid, artist, slugs, primary, album="Al"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {
        "title": f"T{tid}", "artist": artist, "artists": artist.split(", "),
        "artist_slugs": slugs, "primary_artist_slug": primary,
        "album": album, "year": 2008, "duration": 200.0,
        "cover_art_path": "/c.jpg", "genre": "rap",
    }
    return pt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    pts = [
        _pt("1", "Kanye West", ["kanye-west"], "kanye-west"),
        _pt("2", "Kanye West, Sia", ["kanye-west", "sia"], "kanye-west"),
        _pt("3", "Drake, Kanye West", ["drake", "kanye-west"], "drake"),
        _pt("4", "Taylor Swift", ["taylor-swift"], "taylor-swift"),  # must be excluded
    ]
    qdrant = MagicMock()
    # MagicMock ignores scroll_filter and returns all points — the client-side
    # membership guard is what filters here (and on un-backfilled real data).
    qdrant.scroll.return_value = (pts, None)
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    yield TestClient(app)
    MetadataDB._reset_for_tests()
    app.state.db_client = None


def test_collabs_included_nonparticipant_excluded(client):
    resp = client.get("/api/v1/artists/kanye-west", params={"collection": "c"})
    assert resp.status_code == 200
    data = resp.json()
    titles = {t["title"] for al in data["albums"] for t in al["tracks"]}
    assert {"T1", "T2", "T3"} <= titles
    assert "T4" not in titles
    assert data["track_count"] == 3


def test_feat_role_visible_via_primary_slug(client):
    resp = client.get("/api/v1/artists/kanye-west", params={"collection": "c"})
    tracks = {t["title"]: t for al in resp.json()["albums"] for t in al["tracks"]}
    assert tracks["T2"]["primary_artist_slug"] == "kanye-west"
    assert tracks["T3"]["primary_artist_slug"] == "drake"
