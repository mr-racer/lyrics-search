"""Integration tests for library API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import create_app


class TestLibraryCollections:
    def test_collections_returns_no_data_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/collections")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["collections"] == []


class TestLibraryStats:
    def test_stats_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["total_tracks"] == 0


class TestLibraryTopPairs:
    def test_top_pairs_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/top-pairs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["similar"] == []


class TestLibraryBrowse:
    def test_browse_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/browse", params={"q": "test"})
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_omits_query_returns_empty_when_qdrant_down(self):
        """When q is omitted and Qdrant is unavailable, return 200 with empty list."""
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/browse")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_rejects_short_query(self):
        """q with less than 2 characters should be rejected (min_length=2)."""
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/browse", params={"q": "a"})
            assert resp.status_code == 422


class TestLibraryIndex:
    def test_index_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": "/music", "collection_name": "test"},
            )
            assert resp.status_code == 503


class TestLibraryStatus:
    def test_status_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/status")
            assert resp.status_code == 503


class TestLibraryDeleteCollection:
    def test_delete_503_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            resp = c.delete("/api/v1/library/collection/test")
            assert resp.status_code == 503
