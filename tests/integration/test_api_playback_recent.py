"""Integration tests for GET /api/v1/playback/recent."""

from fastapi.testclient import TestClient

from app.api.main import create_app
from ._auth_helper import authenticate_test_client


def test_recent_returns_empty_when_qdrant_down():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent", params={"collection_name": "x"})
        assert resp.status_code == 200
        assert resp.json() == {"tracks": [], "collection_name": "x"}


def test_recent_validates_limit_bounds():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent", params={"collection_name": "x", "limit": 999})
        assert resp.status_code == 422


def test_recent_missing_collection_name_returns_422():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent")
        assert resp.status_code == 422
