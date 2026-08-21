"""Integration tests for the instance API.

Consolidates three suites against the instance endpoints:
- GET /instance/config (no auth) — module-level tests below.
- POST /instance/setup — first-run bootstrap over HTTP (TestInstanceSetup).
- GET/PATCH /api/v1/instance/settings (owner-only) + the ai_available flag on
  /instance/config (TestInstanceSettings).

Real HTTP + real sqlite round-trip. The autouse clean_metadata_db fixture
(conftest) gives each test a temp SQLite DB — these tests create users and must
NEVER touch the developer's real cache/metadata.db.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"

BODY = {"email": "owner@example.com", "password": "abc123", "mode": "sharing"}


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _make_member_token(c, owner_token):
    code = c.post("/api/v1/auth/invites", headers=_bearer(owner_token)).json()["code"]
    c.post("/api/v1/auth/register", json={
        "email": "member@x.y", "password": "memberpass1", "invite_code": code,
    })
    return c.post("/api/v1/auth/login", json={
        "email": "member@x.y", "password": "memberpass1",
    }).json()["token"]


# ── GET /instance/config (no auth) ───────────────────────────────────────────

def test_instance_config_404_when_uninitialized():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 404
        assert r.json()["detail"] == "instance not initialized"


def test_instance_config_returns_sharing_mode():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 200
        assert r.json() == {"mode": "sharing", "ai_available": False}


def test_instance_config_returns_server_mode():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="server", created_at=1.0)
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 200
        assert r.json() == {"mode": "server", "ai_available": False}


# ── POST /instance/setup — first-run bootstrap ───────────────────────────────

class TestInstanceSetup:
    """POST /instance/setup — first-run bootstrap over HTTP.

    Uses the clean_metadata_db fixture (temp SQLite per test) — these tests must
    NEVER touch the developer's real cache/metadata.db because they create users.
    """

    def test_setup_creates_owner_and_returns_working_token(self, clean_metadata_db):
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
            # Instance is now initialized with the chosen mode. ai_available is
            # False until the owner configures + enables AI (instance settings).
            cfg = c.get("/api/v1/instance/config")
            assert cfg.status_code == 200
            assert cfg.json() == {"mode": "sharing", "ai_available": False}

    def test_setup_409_when_already_initialized(self, clean_metadata_db):
        app = create_app()
        with TestClient(app) as c:
            assert c.post("/api/v1/instance/setup", json=BODY).status_code == 200
            r = c.post(
                "/api/v1/instance/setup",
                json={**BODY, "email": "second@example.com"},
            )
            assert r.status_code == 409
            assert r.json()["detail"] == "instance already initialized"

    def test_setup_400_weak_password(self, clean_metadata_db):
        app = create_app()
        with TestClient(app) as c:
            r = c.post("/api/v1/instance/setup", json={**BODY, "password": "abc12"})
            assert r.status_code == 400
            assert "6" in r.json()["detail"]

    def test_setup_422_bad_mode(self, clean_metadata_db):
        app = create_app()
        with TestClient(app) as c:
            r = c.post("/api/v1/instance/setup", json={**BODY, "mode": "cloud"})
            assert r.status_code == 422

    def test_login_works_after_setup(self, clean_metadata_db):
        app = create_app()
        with TestClient(app) as c:
            c.post("/api/v1/instance/setup", json=BODY)
            r = c.post(
                "/api/v1/auth/login",
                json={"email": BODY["email"], "password": BODY["password"]},
            )
            assert r.status_code == 200, r.text
            assert r.json()["user"]["role"] == "owner"


# ── GET/PATCH /api/v1/instance/settings (owner-only) ─────────────────────────

class TestInstanceSettings:
    """GET/PATCH /api/v1/instance/settings (owner-only) and the ai_available
    flag on /instance/config. Real HTTP + real sqlite round-trip."""

    @pytest.fixture(autouse=True)
    def env(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        # Strip mirrored env vars so 'default' is the baseline (the dev shell may
        # have LLM_BASE_URL etc. set — that would make source assertions flaky).
        for name in ("LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY",
                     "EMBED_MODEL", "CLAP_ENABLED", "AI_ENABLED"):
            monkeypatch.delenv(name, raising=False)
        yield

    @pytest.fixture
    def server_app(self):
        """server mode + a pre-seeded owner. Yields (client, owner_token)."""
        app = create_app()
        with TestClient(app) as c:
            conn = MetadataDB.get()
            conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM invites")
            conn.execute("DELETE FROM instance_config")
            conn.execute("DELETE FROM instance_settings")
            conn.commit()
            MetadataDB.set_instance_config(mode="server", created_at=1.0)
            AuthService(jwt_secret=JWT_SECRET).create_owner(
                email="owner@example.com", password="ownerpass12345",
            )
            token = c.post("/api/v1/auth/login", json={
                "email": "owner@example.com", "password": "ownerpass12345",
            }).json()["token"]
            yield c, token

    # ── GET /instance/settings ───────────────────────────────────────────────

    def test_get_settings_owner_returns_resolved_defaults(self, server_app):
        c, token = server_app
        r = c.get("/api/v1/instance/settings", headers=_bearer(token))
        assert r.status_code == 200, r.text
        s = r.json()["settings"]
        # Secret is masked: value null, has_value true (built-in 'lm-studio' default).
        assert s["LLM_API_KEY"]["value"] is None
        assert s["LLM_API_KEY"]["has_value"] is True
        # Unset endpoint resolves to default.
        assert s["LLM_BASE_URL"]["value"] is None
        assert s["LLM_BASE_URL"]["source"] == "default"
        assert s["CLAP_ENABLED"]["value"] == "1"
        assert s["AI_ENABLED"]["value"] == "0"

    def test_get_settings_member_403(self, server_app):
        c, owner_token = server_app
        mem_token = _make_member_token(c, owner_token)
        r = c.get("/api/v1/instance/settings", headers=_bearer(mem_token))
        assert r.status_code == 403

    def test_get_settings_no_auth_401(self, server_app):
        c, _ = server_app
        assert c.get("/api/v1/instance/settings").status_code == 401

    # ── PATCH /instance/settings ─────────────────────────────────────────────

    def test_patch_sets_endpoint_and_enables_ai(self, server_app):
        c, token = server_app
        r = c.patch("/api/v1/instance/settings", headers=_bearer(token), json={
            "llm_base_url": "http://lan:8000/v1", "ai_enabled": True,
        })
        assert r.status_code == 200, r.text
        s = r.json()["settings"]
        assert s["LLM_BASE_URL"]["value"] == "http://lan:8000/v1"
        assert s["LLM_BASE_URL"]["source"] == "db"
        assert s["AI_ENABLED"]["value"] == "1"
        # /config now advertises AI to members.
        cfg = c.get("/api/v1/instance/config").json()
        assert cfg["ai_available"] is True

    def test_patch_api_key_is_stored_but_never_returned(self, server_app):
        c, token = server_app
        r = c.patch("/api/v1/instance/settings", headers=_bearer(token),
                    json={"llm_api_key": "sk-super-secret"})
        assert r.status_code == 200
        body = r.text
        assert "sk-super-secret" not in body   # never leaked in the response
        s = r.json()["settings"]
        assert s["LLM_API_KEY"]["value"] is None
        assert s["LLM_API_KEY"]["has_value"] is True
        assert s["LLM_API_KEY"]["source"] == "db"

    def test_patch_api_key_sentinel_leaves_secret_unchanged(self, server_app):
        c, token = server_app
        c.patch("/api/v1/instance/settings", headers=_bearer(token),
                json={"llm_api_key": "sk-keepme"})
        # A later PATCH that doesn't touch the key sends the sentinel.
        from app.services.settings_service import UNCHANGED_SENTINEL
        c.patch("/api/v1/instance/settings", headers=_bearer(token),
                json={"llm_api_key": UNCHANGED_SENTINEL, "llm_model": "qwen"})
        assert MetadataDB.get_instance_setting("LLM_API_KEY") == "sk-keepme"
        assert MetadataDB.get_instance_setting("LLM_MODEL") == "qwen"

    def test_patch_null_clears_override(self, server_app):
        c, token = server_app
        c.patch("/api/v1/instance/settings", headers=_bearer(token),
                json={"llm_model": "qwen"})
        assert MetadataDB.get_instance_setting("LLM_MODEL") == "qwen"
        c.patch("/api/v1/instance/settings", headers=_bearer(token),
                json={"llm_model": None})
        assert MetadataDB.get_instance_setting("LLM_MODEL") is None

    def test_patch_member_403(self, server_app):
        c, owner_token = server_app
        mem_token = _make_member_token(c, owner_token)
        r = c.patch("/api/v1/instance/settings", headers=_bearer(mem_token),
                    json={"llm_model": "x"})
        assert r.status_code == 403
