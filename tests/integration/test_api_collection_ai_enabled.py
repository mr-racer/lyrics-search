"""Integration tests for collection ai_enabled endpoint + settings shape."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    yield TestClient(app)
    MetadataDB._reset_for_tests()


def test_get_settings_includes_ai_enabled_field_when_no_row(client):
    """Pre-Plan-6 collections (no row) default to ai_enabled=True per
    MetadataDB.get_collection_ai_enabled semantics."""
    r = client.get("/api/v1/library/collections/new_collection/settings")
    assert r.status_code == 200
    body = r.json()
    assert "ai_enabled" in body
    assert body["ai_enabled"] is True


def test_get_settings_reflects_persisted_ai_enabled(client):
    MetadataDB.set_collection_ai_enabled("colA", False)
    r = client.get("/api/v1/library/collections/colA/settings")
    assert r.status_code == 200
    assert r.json()["ai_enabled"] is False


def test_patch_ai_enabled_roundtrip(client):
    r = client.patch(
        "/api/v1/library/collections/colA/ai-enabled",
        json={"enabled": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ai_enabled"] is False
    assert body["collection_name"] == "colA"
    # GET reflects
    r2 = client.get("/api/v1/library/collections/colA/settings")
    assert r2.json()["ai_enabled"] is False


def test_patch_creates_row_when_missing(client):
    r = client.patch(
        "/api/v1/library/collections/brand_new/ai-enabled",
        json={"enabled": False},
    )
    assert r.status_code == 200
    assert MetadataDB.get_collection_ai_enabled("brand_new") is False
