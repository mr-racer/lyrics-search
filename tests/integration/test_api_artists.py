"""Integration tests for GET /artists/{slug}."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import artists as art_route
from app.resources.metadata_db import MetadataDB

# Phase D-soft: server derives the collection from the JWT user, never from the
# client. Override get_current_user with a fixed user so the derived collection
# is deterministic ("acct_user-A") and seed all collection-scoped data under it.
_FIXED_USER = SimpleNamespace(id="user-A", email="a@x")
_DERIVED = "acct_user-A"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    app.dependency_overrides[art_route.get_current_user] = lambda: _FIXED_USER

    # Stub db_client with a qdrant mock that returns canned points
    class FakeQdrant:
        def __init__(self, points):
            self._points = points
        def scroll(self, collection_name, limit, offset, with_payload, with_vectors, scroll_filter=None):
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
    c = TestClient(app)
    yield c
    MetadataDB._reset_for_tests()


def test_get_artist_not_found(client):
    r = client.get("/api/v1/artists/no-such?collection=col_a")
    assert r.status_code == 404


def test_get_artist_aggregates_tracks_and_albums(client):
    # Seed artist + facts under the DERIVED collection (server ignores ?collection).
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", _DERIVED))
    conn.commit()
    MetadataDB.add_artist_facts_batch("dua-lipa", _DERIVED, ["fact1", "fact2"], source="test")
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
                  ("dua-lipa", "Dua Lipa", _DERIVED))
    conn.commit()
    MetadataDB.set_artist_bio("dua-lipa", _DERIVED, "en", "Indie-pop, London.")
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a&lang=en")
    assert r.status_code == 200
    assert r.json()["bio"] == "Indie-pop, London."


def test_get_artist_decade_range_from_year_span(client):
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", _DERIVED))
    conn.commit()
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
    body = r.json()
    # All tracks 2020 → single decade
    assert body["decade_range"] == "2020s"


def test_track_coercion_handles_messy_payload_values():
    """Real Qdrant payloads sometimes carry hyphen-range strings ('154-179')
    in the duration field — must not 500 the endpoint."""
    from app.api.routes.artists import _coerce_float, _coerce_year

    # duration coercion
    assert _coerce_float(195.0) == 195.0
    assert _coerce_float(195) == 195.0
    assert _coerce_float("195") == 195.0
    assert _coerce_float("195.5") == 195.5
    assert _coerce_float("154-179") == 166.5  # average of range
    assert _coerce_float(None) == 0.0
    assert _coerce_float("") == 0.0
    assert _coerce_float("garbage") == 0.0
    assert _coerce_float("154-") == 154.0  # trailing dash → use what we have

    # year coercion
    assert _coerce_year(2020) == 2020
    assert _coerce_year("2020") == 2020
    assert _coerce_year("2018-2020") == 2018  # first valid
    assert _coerce_year(None) is None
    assert _coerce_year("") is None
    assert _coerce_year(0) is None  # zero treated as missing
    assert _coerce_year("garbage") is None


def test_get_artist_album_liked_track_count(client):
    """Albums carry the number of liked tracks (drives the gold-star marker)."""
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", _DERIVED))
    conn.commit()
    # Like both album tracks under the derived collection; a like on the other
    # artist's track (t3) must not bleed into Dua Lipa's count.
    MetadataDB.set_reaction("t1", _DERIVED, "like")
    MetadataDB.set_reaction("t2", _DERIVED, "like")
    MetadataDB.set_reaction("t3", _DERIVED, "like")
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
    assert r.status_code == 200
    album = r.json()["albums"][0]
    assert album["liked_track_count"] == 2


def test_get_artist_liked_track_count_zero_without_reactions(client):
    conn = MetadataDB.get()
    conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                  ("dua-lipa", "Dua Lipa", _DERIVED))
    conn.commit()
    r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
    assert r.json()["albums"][0]["liked_track_count"] == 0


def test_get_artist_ignores_supplied_collection(client):
    """D-soft: even when the client passes ?collection=acct_BAD, the server uses
    the JWT-derived collection (acct_user-A) when building the aggregate."""
    from unittest.mock import patch

    captured: dict = {}

    def fake_build(db, collection, slug, lang):
        captured["collection"] = collection
        from app.domain.models import ArtistAggregate
        return ArtistAggregate(slug=slug, name="X", track_count=0, album_count=0)

    with patch.object(art_route, "build_artist_aggregate", side_effect=fake_build):
        r = client.get("/api/v1/artists/some-slug?collection=acct_BAD")
    assert r.status_code == 200
    assert captured["collection"] == _DERIVED
