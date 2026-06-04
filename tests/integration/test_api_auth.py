"""Integration tests for /api/v1/auth/* — full HTTP round-trips."""
import os
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    yield


@pytest.fixture
def server_mode_app():
    """App with instance_config = server + an owner pre-seeded."""
    from app.services.auth_service import AuthService
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM invites")
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="server", created_at=1.0)
        auth = AuthService(jwt_secret=JWT_SECRET)
        auth.create_owner(email="owner@example.com", password="ownerpass12345")
        yield c


@pytest.fixture
def sharing_mode_app():
    from app.services.auth_service import AuthService
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM invites")
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        auth = AuthService(jwt_secret=JWT_SECRET)
        auth.create_owner(email="owner@example.com", password="ownerpass12345")
        yield c


def _login(c, email, password):
    r = c.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ── /auth/login ──

def test_login_success_returns_token_and_user(server_mode_app):
    r = server_mode_app.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "ownerpass12345"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token"].count(".") == 2
    assert body["user"]["email"] == "owner@example.com"
    assert body["user"]["role"] == "owner"
    assert "password_hash" not in body["user"]


def test_login_wrong_password_returns_401(server_mode_app):
    r = server_mode_app.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "WRONG"},
    )
    assert r.status_code == 401


def test_login_unknown_email_returns_401(server_mode_app):
    r = server_mode_app.post(
        "/api/v1/auth/login",
        json={"email": "ghost@example.com", "password": "anything12345"},
    )
    assert r.status_code == 401


# ── /auth/register ──

def test_register_with_invite_success(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    iv = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()
    r = server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "new@x.y", "password": "newuserpass1", "invite_code": iv["code"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["user"]["role"] == "member"


def test_register_404_in_sharing_mode(sharing_mode_app):
    r = sharing_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "x@y.z", "password": "abcdefgh1234", "invite_code": "any"},
    )
    assert r.status_code == 404


def test_register_400_on_invalid_invite(server_mode_app):
    r = server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "x@y.z", "password": "abcdefgh1234", "invite_code": "nope"},
    )
    assert r.status_code == 400
    assert "invite" in r.json()["detail"].lower()


def test_register_400_on_weak_password(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    code = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()["code"]
    r = server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "weak@x.y", "password": "short", "invite_code": code},
    )
    assert r.status_code == 400


def test_register_409_on_duplicate_email(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    c1 = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()["code"]
    c2 = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()["code"]
    server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "dup@x.y", "password": "duppassword1", "invite_code": c1},
    )
    r = server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "dup@x.y", "password": "duppassword1", "invite_code": c2},
    )
    assert r.status_code == 409


# ── /auth/me ──

def test_me_returns_user(server_mode_app):
    token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    r = server_mode_app.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == "owner@example.com"


def test_me_401_without_token(server_mode_app):
    r = server_mode_app.get("/api/v1/auth/me")
    assert r.status_code == 401


# ── /auth/logout ──

def test_logout_returns_ok(server_mode_app):
    token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    r = server_mode_app.post(
        "/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── /auth/invites ──

def test_create_invite_owner_only(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    r = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["code"]) == 12
    assert body["consumed"] is False


def test_create_invite_403_for_member(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    code = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()["code"]
    server_mode_app.post(
        "/api/v1/auth/register",
        json={"email": "mem@x.y", "password": "memberpass1", "invite_code": code},
    )
    mem_token = _login(server_mode_app, "mem@x.y", "memberpass1")
    r = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {mem_token}"},
    )
    assert r.status_code == 403


def test_list_invites_returns_array(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    r = server_mode_app.get(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_delete_invite_removes_row(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    code = server_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()["code"]
    r = server_mode_app.delete(
        f"/api/v1/auth/invites/{code}",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert r.status_code == 200
    assert MetadataDB.get_invite(code) is None


def test_revoke_unknown_invite_returns_404(server_mode_app):
    owner_token = _login(server_mode_app, "owner@example.com", "ownerpass12345")
    r = server_mode_app.delete(
        "/api/v1/auth/invites/does-not-exist",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert r.status_code == 404


def test_invite_routes_404_in_sharing_mode(sharing_mode_app):
    token = _login(sharing_mode_app, "owner@example.com", "ownerpass12345")
    assert sharing_mode_app.post(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {token}"},
    ).status_code == 404
    assert sharing_mode_app.get(
        "/api/v1/auth/invites",
        headers={"Authorization": f"Bearer {token}"},
    ).status_code == 404
    assert sharing_mode_app.delete(
        "/api/v1/auth/invites/anycode",
        headers={"Authorization": f"Bearer {token}"},
    ).status_code == 404
