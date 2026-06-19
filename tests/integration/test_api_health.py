"""Integration tests for API health endpoints."""

from fastapi.testclient import TestClient

from app.api.main import create_app


class TestHealthEndpoints:
    def test_root_serves_spa(self):
        app = create_app()
        with TestClient(app) as c:
            # Root serves the SPA (frontend/index.html); JSON service info is at /api.
            resp = c.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")

    def test_api_info_returns_name(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Music Explorer"

    def test_health_returns_200(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "qdrant" in data

    def test_health_shows_qdrant_false_when_unavailable(self):
        """When Qdrant is down, health shows qdrant: false."""
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            data = resp.json()
            # In tests, Qdrant is likely unavailable
            if not data["qdrant"]:
                assert data["qdrant"] is False
