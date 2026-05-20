"""Integration test for GET /library/albums (qdrant-unavailable smoke)."""
from fastapi.testclient import TestClient
from app.api.main import create_app


def test_albums_returns_empty_when_qdrant_down():
    app = create_app()
    with TestClient(app) as c:
        resp = c.get("/api/v1/library/albums", params={"collection_name": "music_explorer"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["albums"] == []
        assert body["qdrant_available"] is False


def test_albums_accepts_sort_param():
    app = create_app()
    with TestClient(app) as c:
        for s in ("alphabetical", "year_desc", "year_asc", "track_count_desc"):
            resp = c.get("/api/v1/library/albums",
                         params={"collection_name": "x", "sort": s})
            assert resp.status_code == 200
