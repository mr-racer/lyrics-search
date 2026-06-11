"""Integration test for GET /library/liked-songs (qdrant-down path).

Phase D (D-soft): the collection is derived from the JWT (``acct_<user.id>``),
not from the supplied ``collection_name`` param. We pin a fixed user via
dependency_overrides so the echoed collection is deterministic, and assert that
the supplied bogus param is ignored.
"""
from types import SimpleNamespace

from fastapi.testclient import TestClient
from app.api.main import create_app
from app.api.routes import library as lib_route


def test_liked_songs_returns_empty_when_qdrant_down():
    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.state.db_client = None  # force qdrant-down branch
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        # Supplied collection_name is intentionally bogus — it must be ignored.
        resp = c.get("/api/v1/library/liked-songs", params={"collection_name": "x"})
        assert resp.status_code == 200
        assert resp.json() == {"tracks": [], "collection_name": "acct_user-A"}
    app.dependency_overrides.clear()
