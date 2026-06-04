"""Integration test for GET /library/listening-stats."""
from fastapi.testclient import TestClient
from app.api.main import create_app
from ._auth_helper import authenticate_test_client


def test_listening_stats_empty_response_when_qdrant_down():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/library/listening-stats",
                     params={"collection_name": "x", "lang": "en"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_seconds_listened"] == 0
        assert body["top_track"] is None
        assert body["top_artist"] is None
        assert body["peak_hour"] is None


def test_listening_stats_rejects_unknown_lang():
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/library/listening-stats",
                     params={"collection_name": "x", "lang": "fr"})
        assert resp.status_code == 422
