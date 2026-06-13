"""Integration tests: GET /api/v1/admin/members (owner-only, server mode)."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    yield


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


def test_members_lists_owner_and_member_with_invite_code():
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


def test_members_403_for_member():
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


def test_members_401_without_token():
    app = create_app()
    with TestClient(app) as c:
        _reset_server_mode()
        assert c.get("/api/v1/admin/members").status_code == 401


def test_members_404_in_sharing_mode():
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
