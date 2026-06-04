"""Integration test for GET /library/liked-songs (qdrant-down path)."""
from fastapi.testclient import TestClient
from app.api.main import create_app
from ._auth_helper import authenticate_test_client


def test_liked_songs_returns_empty_when_qdrant_down():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/library/liked-songs", params={"collection_name": "x"})
        assert resp.status_code == 200
        assert resp.json() == {"tracks": [], "collection_name": "x"}
