"""Integration test for GET /library/year-facets endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    yield TestClient(app)
    MetadataDB._reset_for_tests()


def test_year_facets_endpoint_returns_shape_for_empty_db(client):
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json() == {"year_ranges": []}


def test_year_facets_endpoint_reflects_db_state(client):
    MetadataDB.upsert_artist("a", "A", "test_col")
    MetadataDB.upsert_song("a-1", "T1", "a", "test_col")
    conn = MetadataDB._connect()
    conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", ("1990-1994", "a-1"))
    conn.commit()
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json()["year_ranges"] == [{"value": "1990-1994", "count": 1}]


def test_year_facets_endpoint_respects_top_k_param(client):
    MetadataDB.upsert_artist("a", "A", "test_col")
    conn = MetadataDB._connect()
    for i in range(20):
        slug = f"a-{i}"
        MetadataDB.upsert_song(slug, f"T{i}", "a", "test_col")
        conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", (f"{1990+i}-{1994+i}", slug))
    conn.commit()
    r = client.get("/api/v1/library/year-facets?top_k=5")
    assert r.status_code == 200
    assert len(r.json()["year_ranges"]) == 5
