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


# ── Phase D-soft tests ────────────────────────────────────────────────────────

class TestPhaseDSoft:
    """D-soft: endpoints derive collection from JWT, ignore client-supplied value."""

    def test_search_ignores_client_supplied_collection_name(self):
        """D-soft: even when client passes collection_name in body, server uses
        the JWT-derived value."""
        from unittest.mock import MagicMock
        from app.api.main import create_app
        from app.services.search_service import SearchService
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        mock_service = MagicMock(spec=SearchService)
        received = {}

        async def mock_search(**kwargs):
            received.update(kwargs)
            return []

        mock_service.search = mock_search
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        with TestClient(app) as c:
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "x", "mode": "text", "collection_name": "acct_user-B"},
            )
        assert resp.status_code == 200
        assert received["collection_name"] == "acct_user-A"

    def test_stream_uses_derived_collection(self):
        """D-soft: GET stream ignores ?collection_name and uses JWT-derived value."""
        from unittest.mock import MagicMock
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fake_db = MagicMock()
        fake_db.qdrant.retrieve.return_value = []
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        with TestClient(app) as c:
            c.app.state.db_client = fake_db
            c.get("/api/v1/search/tracks/T1/stream?collection_name=acct_user-B")
        assert fake_db.qdrant.retrieve.call_args.kwargs["collection_name"] == "acct_user-A"

    def test_reaction_set_ignores_client_supplied_collection_name(self):
        """D-soft: POST reaction ignores collection_name in body, uses JWT-derived value."""
        from unittest.mock import MagicMock, patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_set_reaction(track_id, collection_name, reaction):
            captured["collection_name"] = collection_name

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.set_reaction", side_effect=fake_set_reaction):
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/search/tracks/T1/reaction",
                    json={"collection_name": "acct_user-B", "reaction": "like"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert captured["collection_name"] == "acct_user-A"
        assert data["collection_name"] == "acct_user-A"

    def test_reaction_get_uses_derived_collection(self):
        """D-soft: GET reaction ignores ?collection_name query param, uses JWT-derived value."""
        from unittest.mock import patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_get_reaction(track_id, collection_name):
            captured["collection_name"] = collection_name
            return "like"

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.get_reaction", side_effect=fake_get_reaction):
            with TestClient(app) as c:
                resp = c.get(
                    "/api/v1/search/tracks/T1/reaction?collection_name=acct_user-B",
                )
        assert resp.status_code == 200
        data = resp.json()
        assert captured["collection_name"] == "acct_user-A"
        assert data["collection_name"] == "acct_user-A"

    def test_reaction_post_without_collection_name_is_accepted(self):
        """D-soft: a client that DROPS collection_name from the body must NOT
        get a 422 — the field is optional now (frontend stops sending it at the
        same deploy as D-soft). Server still derives + uses acct_<user.id>."""
        from unittest.mock import patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_set_reaction(track_id, collection_name, reaction):
            captured["collection_name"] = collection_name

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.set_reaction", side_effect=fake_set_reaction):
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/search/tracks/T1/reaction",
                    json={"reaction": "like"},  # NO collection_name
                )
        assert resp.status_code == 200
        assert captured["collection_name"] == "acct_user-A"
