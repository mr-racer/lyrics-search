"""Integration tests for GET /api/v1/playback/recent."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import playback as pb_route
from ._auth_helper import authenticate_test_client


def _app_with_fixed_user(user_id: str = "user-T"):
    """Return a fresh app with get_current_user overridden to a fixed user."""
    app = create_app()
    fixed = SimpleNamespace(id=user_id, email="t@x")
    app.dependency_overrides[pb_route.get_current_user] = lambda: fixed
    return app


def test_recent_returns_empty_when_qdrant_down():
    """D-soft: collection_name param is accepted but ignored; server derives
    from JWT. Response echoes the derived collection, not the supplied one."""
    app = _app_with_fixed_user("user-T")
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent", params={"collection_name": "x"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["tracks"] == []
        # Derived collection, NOT the supplied "x"
        assert body["collection_name"] == "acct_user-T"


def test_recent_validates_limit_bounds():
    app = _app_with_fixed_user()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent", params={"collection_name": "x", "limit": 999})
        assert resp.status_code == 422


def test_recent_no_collection_name_uses_derived():
    """D-soft: collection_name is now optional; omitting it still works and
    returns the derived collection from JWT."""
    app = _app_with_fixed_user("user-T")
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        resp = c.get("/api/v1/playback/recent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_name"] == "acct_user-T"
