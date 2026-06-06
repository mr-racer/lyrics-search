"""Integration tests for the ai_enabled endpoint + settings shape.

Phase D (D-soft): the collection name is no longer part of the path — it is
derived from the JWT as ``acct_<user.id>``. These tests pin a fixed user via
``dependency_overrides`` so the derived collection is deterministic
(``acct_user-A``) and exercise the renamed routes:
  GET   /library/settings
  PATCH /library/ai-enabled
The old per-collection paths now return 410.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import library as lib_route
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client

_DERIVED = "acct_user-A"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    _app = create_app()
    fixed = SimpleNamespace(id="user-A", email="a@x")
    _app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    c = TestClient(_app)
    yield c
    _app.dependency_overrides.clear()
    MetadataDB._reset_for_tests()


def test_get_settings_includes_ai_enabled_field_when_no_row(client):
    """Pre-Plan-6 collections (no row) default to ai_enabled=True per
    MetadataDB.get_collection_ai_enabled semantics."""
    r = client.get("/api/v1/library/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["collection_name"] == _DERIVED
    assert "ai_enabled" in body
    assert body["ai_enabled"] is True


def test_get_settings_reflects_persisted_ai_enabled(client):
    MetadataDB.set_collection_ai_enabled(_DERIVED, False)
    r = client.get("/api/v1/library/settings")
    assert r.status_code == 200
    assert r.json()["ai_enabled"] is False


def test_patch_ai_enabled_roundtrip(client):
    r = client.patch("/api/v1/library/ai-enabled", json={"enabled": False})
    assert r.status_code == 200
    body = r.json()
    assert body["ai_enabled"] is False
    assert body["collection_name"] == _DERIVED
    # GET reflects
    r2 = client.get("/api/v1/library/settings")
    assert r2.json()["ai_enabled"] is False


def test_patch_creates_row_when_missing(client):
    r = client.patch("/api/v1/library/ai-enabled", json={"enabled": False})
    assert r.status_code == 200
    assert MetadataDB.get_collection_ai_enabled(_DERIVED) is False


def test_patch_re_enables_after_disable(client):
    """ai_enabled defaults to True, so a broken write for enabled=True
    would be masked by the default. Exercise the full False -> True
    round-trip explicitly."""
    client.patch("/api/v1/library/ai-enabled", json={"enabled": False})
    r = client.patch("/api/v1/library/ai-enabled", json={"enabled": True})
    assert r.status_code == 200
    assert r.json()["ai_enabled"] is True
    assert client.get("/api/v1/library/settings").json()["ai_enabled"] is True


def test_old_settings_path_is_410():
    """The per-collection settings path is deprecated — it 410s and points at
    GET /library/settings."""
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        r = c.get("/api/v1/library/collections/colA/settings")
        assert r.status_code == 410


def test_old_ai_enabled_path_is_410():
    """The per-collection ai-enabled path is deprecated — it 410s."""
    app = create_app()
    with TestClient(app) as c:
        authenticate_test_client(c, app)
        r = c.patch(
            "/api/v1/library/collections/colA/ai-enabled",
            json={"enabled": False},
        )
        assert r.status_code == 410
