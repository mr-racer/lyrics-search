"""Integration tests for GET /artists/{slug}."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()

    # Stub db_client with a qdrant mock that returns canned points
    class FakeQdrant:
        def __init__(self, points):
            self._points = points
        def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
            if offset is None:
                return list(self._points), None
            return [], None
        def get_collections(self):
            m = MagicMock(); m.collections = [MagicMock(name="x")]
            return m

    class FakePoint:
        def __init__(self, id, payload):
            self.id = id
            self.payload = payload

    points = [
        FakePoint("t1", {"artist": "Dua Lipa", "title": "Physical", "album": "Future Nostalgia",
                          "year": 2020, "duration": 195.0, "file_path": "/a.flac",
                          "cover_art_path": "/covers/c1.jpg", "genre": "pop"}),
        FakePoint("t2", {"artist": "Dua Lipa", "title": "Levitating", "album": "Future Nostalgia",
                          "year": 2020, "duration": 203.0, "file_path": "/b.flac",
                          "cover_art_path": "/covers/c2.jpg"}),
        FakePoint("t3", {"artist": "Other", "title": "x", "album": "y",
                          "year": 2010, "duration": 120.0, "file_path": "/c.flac"}),
    ]
    db = MagicMock()
    db.qdrant = FakeQdrant(points)
    app.state.db_client = db
    yield TestClient(app)
    MetadataDB._reset_for_tests()


def test_get_artist_not_found(client):
    r = client.get("/api/v1/artists/no-such?collection=col_a")
    assert r.status_code == 404


def test_get_artist_aggregates_tracks_and_albums(client):
    # Seed artist + facts
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", "col_a"))
    conn.commit()
    MetadataDB.add_artist_facts_batch("dua-lipa", "col_a", ["fact1", "fact2"], source="test")
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "dua-lipa"
    assert body["name"] == "Dua Lipa"
    assert body["track_count"] == 2  # 'Other' artist's track excluded
    assert body["album_count"] == 1
    assert body["albums"][0]["title"] == "Future Nostalgia"
    assert len(body["albums"][0]["tracks"]) == 2
    assert body["facts"] == ["fact1", "fact2"]
    assert body["bio"] is None  # not indexed


def test_get_artist_includes_bio_when_indexed(client):
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", "col_a"))
    conn.commit()
    MetadataDB.set_artist_bio("dua-lipa", "col_a", "en", "Indie-pop, London.")
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a&lang=en")
    assert r.status_code == 200
    assert r.json()["bio"] == "Indie-pop, London."


def test_get_artist_decade_range_from_year_span(client):
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", "col_a"))
    conn.commit()
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
    body = r.json()
    # All tracks 2020 → single decade
    assert body["decade_range"] == "2020s"
