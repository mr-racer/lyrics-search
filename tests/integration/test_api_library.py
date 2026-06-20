"""Integration tests for library API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import create_app
from ._auth_helper import authenticate_test_client


class TestLibraryCollections:
    def test_collections_returns_no_data_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/collections")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["collections"] == []


class TestLibraryStats:
    def test_stats_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["total_tracks"] == 0


class TestLibraryTopPairs:
    def test_top_pairs_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/top-pairs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["similar"] == []

    def test_top_pairs_for_track_unavailable_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/top-pairs/some-track-id")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is False
            assert data["similar"] == []
            assert data["dissimilar"] == []


class TestLibraryBrowse:
    def test_browse_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse", params={"q": "test"})
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_omits_query_returns_empty_when_qdrant_down(self):
        """When q is omitted and Qdrant is unavailable, return 200 with empty list."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_rejects_short_query(self):
        """q with less than 2 characters should be rejected (min_length=2)."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse", params={"q": "a"})
            assert resp.status_code == 422


class TestLibraryIndex:
    def test_index_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": "/music", "collection_name": "test"},
            )
            assert resp.status_code == 503


class TestLibraryStatus:
    def test_status_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/status")
            assert resp.status_code == 503


class TestLibraryDeleteCollection:
    def test_delete_collection_is_gone_410(self):
        """Phase D removed self-serve collection delete — the old path now 410s
        and points callers at the admin wipe endpoint."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.delete("/api/v1/library/collection/test")
            assert resp.status_code == 410
            assert "wipe" in resp.json()["detail"]


# ── Phase D (D-soft): collection derived from JWT, supplied param ignored ──────

def test_library_endpoints_ignore_supplied_collection():
    """Every collection-aware GET tolerates a bogus supplied collection_name:
    the server derives the collection from the JWT and never 5xxs on it."""
    from app.api.routes import library as lib_route
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        for path in [
            "/api/v1/library/browse?collection_name=acct_BAD",
            "/api/v1/library/random?collection_name=acct_BAD",
            "/api/v1/library/albums?collection_name=acct_BAD",
            "/api/v1/library/liked-songs?collection_name=acct_BAD",
            "/api/v1/library/rediscover?collection_name=acct_BAD",
            "/api/v1/library/listening-stats?collection_name=acct_BAD",
            "/api/v1/library/stats?collection_name=acct_BAD",
            "/api/v1/library/top-pairs?collection_name=acct_BAD",
        ]:
            resp = c.get(path)
            assert resp.status_code < 500, path
    app.dependency_overrides.clear()


def test_library_echoes_derived_collection_not_supplied():
    """The response echo reflects the JWT-derived collection, not the param."""
    from app.api.routes import library as lib_route
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.state.db_client = None  # force the graceful-empty echo branch
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        resp = c.get("/api/v1/library/liked-songs?collection_name=acct_BAD")
        assert resp.status_code == 200
        assert resp.json()["collection_name"] == "acct_user-A"
    app.dependency_overrides.clear()


class TestLibrarySettingsRename:
    def test_settings_uses_derived_collection(self):
        from app.api.routes import library as lib_route
        from types import SimpleNamespace
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["collection_name"] == "acct_user-A"
            assert body["ai_enabled"] is True  # no row → AI-on default
        app.dependency_overrides.clear()

    def test_old_settings_path_is_410(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/collections/whatever/settings")
            assert resp.status_code == 410
            assert "settings" in resp.json()["detail"].lower()

    def test_ai_enabled_uses_derived_collection(self):
        from app.api.routes import library as lib_route
        from types import SimpleNamespace
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            resp = c.patch("/api/v1/library/ai-enabled", json={"enabled": False})
            assert resp.status_code == 200
            assert resp.json()["collection_name"] == "acct_user-A"
            assert resp.json()["ai_enabled"] is False
        app.dependency_overrides.clear()

    def test_old_ai_enabled_path_is_410(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.patch(
                "/api/v1/library/collections/whatever/ai-enabled",
                json={"enabled": False},
            )
            assert resp.status_code == 410
