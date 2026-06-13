"""Integration tests: GET/PATCH /api/v1/instance/settings (owner-only) and the
ai_available flag on /instance/config. Real HTTP + real sqlite round-trip."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    # Strip mirrored env vars so 'default' is the baseline (the dev shell may
    # have LLM_BASE_URL etc. set — that would make source assertions flaky).
    for name in ("LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY",
                 "EMBED_MODEL", "CLAP_ENABLED", "AI_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def server_app():
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


# ── GET /instance/settings ───────────────────────────────────────────────────

def test_get_settings_owner_returns_resolved_defaults(server_app):
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


def test_get_settings_member_403(server_app):
    c, owner_token = server_app
    mem_token = _make_member_token(c, owner_token)
    r = c.get("/api/v1/instance/settings", headers=_bearer(mem_token))
    assert r.status_code == 403


def test_get_settings_no_auth_401(server_app):
    c, _ = server_app
    assert c.get("/api/v1/instance/settings").status_code == 401


# ── PATCH /instance/settings ─────────────────────────────────────────────────

def test_patch_sets_endpoint_and_enables_ai(server_app):
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


def test_patch_api_key_is_stored_but_never_returned(server_app):
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


def test_patch_api_key_sentinel_leaves_secret_unchanged(server_app):
    c, token = server_app
    c.patch("/api/v1/instance/settings", headers=_bearer(token),
            json={"llm_api_key": "sk-keepme"})
    # A later PATCH that doesn't touch the key sends the sentinel.
    from app.services.settings_service import UNCHANGED_SENTINEL
    c.patch("/api/v1/instance/settings", headers=_bearer(token),
            json={"llm_api_key": UNCHANGED_SENTINEL, "llm_model": "qwen"})
    assert MetadataDB.get_instance_setting("LLM_API_KEY") == "sk-keepme"
    assert MetadataDB.get_instance_setting("LLM_MODEL") == "qwen"


def test_patch_null_clears_override(server_app):
    c, token = server_app
    c.patch("/api/v1/instance/settings", headers=_bearer(token),
            json={"llm_model": "qwen"})
    assert MetadataDB.get_instance_setting("LLM_MODEL") == "qwen"
    c.patch("/api/v1/instance/settings", headers=_bearer(token),
            json={"llm_model": None})
    assert MetadataDB.get_instance_setting("LLM_MODEL") is None


def test_patch_member_403(server_app):
    c, owner_token = server_app
    mem_token = _make_member_token(c, owner_token)
    r = c.patch("/api/v1/instance/settings", headers=_bearer(mem_token),
                json={"llm_model": "x"})
    assert r.status_code == 403
