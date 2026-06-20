"""Integration tests for GET /api/v1/search/catalog (fake Qdrant injected — the
engine is unit-tested separately; this checks route wiring + auth + response)."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.qdrant_utils import invalidate_light_cache
from app.services import catalog_search_service
from ._auth_helper import authenticate_test_client

pytestmark = pytest.mark.integration


class _P:
    def __init__(self, i, p):
        self.id, self.payload = i, p


class _Count:
    def __init__(self, n):
        self.count = n


class _FakeQdrant:
    def __init__(self, pts):
        self._pts = pts

    def scroll(self, collection_name, limit, with_payload, with_vectors, offset=None):
        if offset:
            return [], None
        return [_P(i, p) for i, p in self._pts], None

    def count(self, collection_name, exact=True):
        return _Count(len(self._pts))


class _FakeDb:
    def __init__(self, q):
        self.qdrant = q


def _pt(tid, title, artist, album):
    slug = artist.lower().replace(" ", "-")
    return (tid, {
        "title": title, "artist": artist, "album": album,
        "artists": [artist], "artist_slugs": [slug], "primary_artist_slug": slug,
        "cover_art_path": None, "file_path": f"/m/{tid}.mp3", "duration": 200.0,
    })


POINTS = [
    _pt("t1", "Bohemian Rhapsody", "Queen", "A Night at the Opera"),
    _pt("t2", "Time", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t3", "Money", "Pink Floyd", "The Dark Side of the Moon"),
]


def _client():
    app = create_app()
    c = TestClient(app)
    authenticate_test_client(c, app)
    app.state.db_client = _FakeDb(_FakeQdrant(POINTS))
    invalidate_light_cache()
    catalog_search_service.invalidate()
    return c


def test_catalog_short_query_returns_empty():
    r = _client().get("/api/v1/search/catalog?q=a")
    assert r.status_code == 200
    assert r.json() == []


def test_catalog_returns_song_hit():
    r = _client().get("/api/v1/search/catalog?q=bohem")
    assert r.status_code == 200
    data = r.json()
    assert data and data[0]["type"] == "song"
    assert data[0]["title"] == "Bohemian Rhapsody"
    assert data[0]["track_id"] == "t1"


def test_catalog_album_query_returns_album_entity():
    r = _client().get("/api/v1/search/catalog?q=dark side of the moon")
    assert r.status_code == 200
    data = r.json()
    assert data and data[0]["type"] == "album"
    assert "dark side of the moon" in data[0]["album"].lower()


def test_catalog_requires_auth():
    app = create_app()
    # Seed auth_service (+owner) via a throwaway client, then hit it WITHOUT a token.
    authenticate_test_client(TestClient(app), app)
    app.state.db_client = _FakeDb(_FakeQdrant(POINTS))
    r = TestClient(app).get("/api/v1/search/catalog?q=bohem")
    assert r.status_code == 401
