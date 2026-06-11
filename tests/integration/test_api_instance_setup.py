"""Integration tests: POST /instance/setup — first-run bootstrap over HTTP.

Uses the clean_metadata_db fixture (temp SQLite per test) — these tests must
NEVER touch the developer's real cache/metadata.db because they create users.
"""
from fastapi.testclient import TestClient

from app.api.main import create_app


BODY = {"email": "owner@example.com", "password": "abc123", "mode": "sharing"}


def test_setup_creates_owner_and_returns_working_token(clean_metadata_db):
    app = create_app()
    with TestClient(app) as c:
        r = c.post("/api/v1/instance/setup", json=BODY)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["user"]["role"] == "owner"
        assert data["user"]["email"] == "owner@example.com"
        # Returned JWT must authenticate protected routes immediately.
        me = c.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {data['token']}"},
        )
        assert me.status_code == 200, me.text
        assert me.json()["id"] == data["user"]["id"]
        # Instance is now initialized with the chosen mode.
        cfg = c.get("/api/v1/instance/config")
        assert cfg.status_code == 200
        assert cfg.json() == {"mode": "sharing"}


def test_setup_409_when_already_initialized(clean_metadata_db):
    app = create_app()
    with TestClient(app) as c:
        assert c.post("/api/v1/instance/setup", json=BODY).status_code == 200
        r = c.post(
            "/api/v1/instance/setup",
            json={**BODY, "email": "second@example.com"},
        )
        assert r.status_code == 409
        assert r.json()["detail"] == "instance already initialized"


def test_setup_400_weak_password(clean_metadata_db):
    app = create_app()
    with TestClient(app) as c:
        r = c.post("/api/v1/instance/setup", json={**BODY, "password": "abc12"})
        assert r.status_code == 400
        assert "6" in r.json()["detail"]


def test_setup_422_bad_mode(clean_metadata_db):
    app = create_app()
    with TestClient(app) as c:
        r = c.post("/api/v1/instance/setup", json={**BODY, "mode": "cloud"})
        assert r.status_code == 422


def test_login_works_after_setup(clean_metadata_db):
    app = create_app()
    with TestClient(app) as c:
        c.post("/api/v1/instance/setup", json=BODY)
        r = c.post(
            "/api/v1/auth/login",
            json={"email": BODY["email"], "password": BODY["password"]},
        )
        assert r.status_code == 200, r.text
        assert r.json()["user"]["role"] == "owner"
