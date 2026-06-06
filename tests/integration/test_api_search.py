"""Integration tests for search API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.search_service import SearchService
from ._auth_helper import authenticate_test_client


class TestSearchAPI:
    def test_search_503_when_service_unavailable(self):
        """POST /api/v1/search/ returns 503 when the search service is down.

        Force search_service=None so this is deterministic regardless of whether
        Qdrant happens to be reachable in the dev env (the lifespan would
        otherwise wire a real SearchService when Qdrant is up)."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            app.state.search_service = None
            resp = c.post(
                "/api/v1/search/",
                json={"query": "test", "mode": "text"},
            )
            assert resp.status_code == 503

    def test_search_returns_results_when_service_available(self):
        """POST /api/v1/search/ returns hits when SearchService is mocked."""
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={
                    "query": "hello",
                    "mode": "text",
                    "limit": 5,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "hits" in data
            assert data["query"] == "hello"
            assert data["mode"] == "text"

    def test_search_with_audio_mode(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "beat", "mode": "audio"},
            )
            assert resp.status_code == 200
            assert resp.json()["mode"] == "audio"

    def test_search_with_hybrid_mode(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "vibe", "mode": "hybrid"},
            )
            assert resp.status_code == 200
            assert resp.json()["mode"] == "hybrid"

    def test_search_with_filters(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={
                    "query": "pop",
                    "mode": "text",
                    "filters": {"genre": "Pop", "year_ranges": ["2000-2009"]},
                },
            )
            assert resp.status_code == 200

    def test_search_request_validation_missing_query(self):
        """Empty query should fail validation."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/search/",
                json={"mode": "text"},
            )
            assert resp.status_code == 422

    def test_search_models_text_endpoint(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/search/models/text")
            assert resp.status_code == 200

    def test_search_models_loaded_endpoint(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/search/models/loaded")
            assert resp.status_code == 200
            data = resp.json()
            assert "text_models" in data
            assert "clap_available" in data
