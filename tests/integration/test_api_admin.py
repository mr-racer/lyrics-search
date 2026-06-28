"""Integration tests for the owner-only /api/v1/admin routes.

Consolidated from:
  - test_api_admin_members.py  -> class TestAdminMembers
  - test_api_admin_wipe.py     -> class TestAdminWipe
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import admin as admin_route
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


# --- helpers (members) ------------------------------------------------------

def _reset_server_mode():
    conn = MetadataDB.get()
    for t in ("users", "invites", "instance_config", "instance_settings"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    MetadataDB.set_instance_config(mode="server", created_at=1.0)
    AuthService(jwt_secret=JWT_SECRET).create_owner(
        email="owner@example.com", password="ownerpass12345",
    )


def _bearer(t):
    return {"Authorization": f"Bearer {t}"}


def _login(c, email, password):
    return c.post("/api/v1/auth/login",
                  json={"email": email, "password": password}).json()["token"]


# --- helpers / fixtures (wipe) ----------------------------------------------

_OWNER = SimpleNamespace(id="owner-1", email="owner@x", role="owner")
_MEMBER = SimpleNamespace(id="member-1", email="m@x", role="member")


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point MetadataDB at a throwaway SQLite file for this test."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "admin.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _app_as(user):
    """Build an app whose get_current_user resolves to `user` (so get_owner sees
    its role). No lifespan needed — the override bypasses the JWT/auth_service."""
    app = create_app()
    app.dependency_overrides[admin_route.get_current_user] = lambda: user
    return app


def _seed_user(user_id: str):
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x", password_hash="h",
        role="member", created_at=0.0,
    )


class TestAdminMembers:
    """GET /api/v1/admin/members (owner-only, server mode)."""

    @pytest.fixture(autouse=True)
    def env(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        yield

    def test_members_lists_owner_and_member_with_invite_code(self):
        app = create_app()
        with TestClient(app) as c:
            _reset_server_mode()
            owner_token = _login(c, "owner@example.com", "ownerpass12345")
            code = c.post("/api/v1/auth/invites", headers=_bearer(owner_token)).json()["code"]
            c.post("/api/v1/auth/register", json={
                "email": "member@x.y", "password": "memberpass1", "invite_code": code,
            })
            r = c.get("/api/v1/admin/members", headers=_bearer(owner_token))
            assert r.status_code == 200, r.text
            members = {m["email"]: m for m in r.json()}
            assert members["owner@example.com"]["role"] == "owner"
            assert members["owner@example.com"]["invite_code"] is None
            assert members["member@x.y"]["role"] == "member"
            assert members["member@x.y"]["invite_code"] == code

    def test_members_403_for_member(self):
        app = create_app()
        with TestClient(app) as c:
            _reset_server_mode()
            owner_token = _login(c, "owner@example.com", "ownerpass12345")
            code = c.post("/api/v1/auth/invites", headers=_bearer(owner_token)).json()["code"]
            c.post("/api/v1/auth/register", json={
                "email": "mem@x.y", "password": "memberpass1", "invite_code": code,
            })
            mem_token = _login(c, "mem@x.y", "memberpass1")
            assert c.get("/api/v1/admin/members", headers=_bearer(mem_token)).status_code == 403

    def test_members_401_without_token(self):
        app = create_app()
        with TestClient(app) as c:
            _reset_server_mode()
            assert c.get("/api/v1/admin/members").status_code == 401

    def test_members_404_in_sharing_mode(self):
        app = create_app()
        with TestClient(app) as c:
            conn = MetadataDB.get()
            for t in ("users", "invites", "instance_config", "instance_settings"):
                conn.execute(f"DELETE FROM {t}")
            conn.commit()
            MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
            AuthService(jwt_secret=JWT_SECRET).create_owner(
                email="owner@example.com", password="ownerpass12345",
            )
            token = _login(c, "owner@example.com", "ownerpass12345")
            assert c.get("/api/v1/admin/members", headers=_bearer(token)).status_code == 404


class TestAdminWipe:
    """Owner-only POST /admin/accounts/{user_id}/wipe.

    Replaces the self-serve DELETE /library/collection/{name} removed in Phase D.
    Covers: role gate (403 for members), happy path (owner wipes any account →
    Qdrant delete_collection called with acct_<user_id>), unknown-user 404 guard,
    and 503 when Qdrant is unavailable.
    """

    def test_member_forbidden(self, db):
        """A logged-in member hits the role gate → 403 (not 401: they ARE logged in)."""
        app = _app_as(_MEMBER)
        app.state.db_client = MagicMock()
        c = TestClient(app)
        r = c.post("/api/v1/admin/accounts/member-1/wipe")
        assert r.status_code == 403

    def test_owner_can_wipe_any_account(self, db):
        """Owner wipes a real member account → 200 + delete_collection(acct_<id>)."""
        _seed_user("some-user")
        app = _app_as(_OWNER)
        fake_qdrant = MagicMock()
        fake_qdrant.scroll.return_value = ([], None)
        app.state.db_client = MagicMock(qdrant=fake_qdrant)
        c = TestClient(app)
        r = c.post("/api/v1/admin/accounts/some-user/wipe")
        assert r.status_code == 200
        fake_qdrant.delete_collection.assert_called_once_with("acct_some-user")
        body = r.json()
        assert body["deleted"] is True
        assert body["user_id"] == "some-user"
        assert body["collection_name"] == "acct_some-user"

    def test_owner_wipe_unknown_user_returns_404(self, db):
        """Owner targets a non-existent user → 404 before any Qdrant call."""
        app = _app_as(_OWNER)
        fake_qdrant = MagicMock()
        app.state.db_client = MagicMock(qdrant=fake_qdrant)
        c = TestClient(app)
        r = c.post("/api/v1/admin/accounts/ghost/wipe")
        assert r.status_code == 404
        fake_qdrant.delete_collection.assert_not_called()

    def test_owner_wipe_503_when_qdrant_unavailable(self, db):
        """Owner wipe with Qdrant down → 503 (db_client is None)."""
        _seed_user("some-user")
        app = _app_as(_OWNER)
        app.state.db_client = None
        c = TestClient(app)
        r = c.post("/api/v1/admin/accounts/some-user/wipe")
        assert r.status_code == 503
