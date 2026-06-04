"""Integration test for GET /library/sonic-facets endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()


def test_sonic_facets_endpoint_returns_shape_for_empty_db(client):
    r = client.get("/api/v1/library/sonic-facets")
    assert r.status_code == 200
    body = r.json()
    assert body == {"tags": []}


def test_sonic_facets_endpoint_reflects_db_state(client):
    MetadataDB.upsert_artist("a", "A", "test_col")
    MetadataDB.upsert_song("a-1", "T1", "a", "test_col")
    MetadataDB.upsert_sonic_descriptor(
        song_slug="a-1",
        tags=[{"tag": "melancholic", "score": 0.8}],
    )
    r = client.get("/api/v1/library/sonic-facets")
    assert r.status_code == 200
    body = r.json()
    assert body["tags"] == [{"value": "melancholic", "count": 1}]


def test_sonic_facets_endpoint_respects_top_k_param(client):
    MetadataDB.upsert_artist("a", "A", "test_col")
    for i in range(20):
        slug = f"a-{i}"
        MetadataDB.upsert_song(slug, f"T{i}", "a", "test_col")
        MetadataDB.upsert_sonic_descriptor(
            song_slug=slug,
            tags=[{"tag": f"tag-{i}", "score": 0.5}],
        )
    r = client.get("/api/v1/library/sonic-facets?top_k=5")
    assert r.status_code == 200
    assert len(r.json()["tags"]) == 5
