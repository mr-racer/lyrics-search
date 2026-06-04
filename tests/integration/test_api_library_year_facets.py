"""Integration test for GET /library/year-facets endpoint.

The endpoint now reads year_range from Qdrant payload (not SQLite), so
tests mock db_client.qdrant following the pattern in
tests/integration/test_backfill_sonic_payload.py.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


def _col(name: str) -> MagicMock:
    obj = MagicMock()
    obj.name = name
    return obj


def _point(yr: str | None) -> MagicMock:
    p = MagicMock()
    p.payload = {"year_range": yr} if yr else {}
    return p


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    tc = TestClient(app)
    authenticate_test_client(tc, app)
    yield app, tc
    MetadataDB._reset_for_tests()


def _attach_qdrant(app, collections: list[str], pages: dict[str, list]):
    qdrant = MagicMock()
    qdrant.get_collections.return_value.collections = [_col(n) for n in collections]

    def scroll_side(collection_name, limit, with_payload, with_vectors, offset=None):
        return (pages.get(collection_name, []), None)

    qdrant.scroll.side_effect = scroll_side
    db_client = MagicMock()
    db_client.qdrant = qdrant
    app.state.db_client = db_client


def test_year_facets_endpoint_returns_shape_for_empty_db(client):
    app, tc = client
    _attach_qdrant(app, ["test_col"], {"test_col": []})
    r = tc.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json() == {"year_ranges": []}


def test_year_facets_endpoint_reflects_qdrant_state(client):
    app, tc = client
    _attach_qdrant(
        app,
        ["test_col"],
        {"test_col": [_point("1990-1994"), _point("1990-1994"), _point("2000-2004")]},
    )
    r = tc.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    data = r.json()
    assert data["year_ranges"][0] == {"value": "1990-1994", "count": 2}
    assert data["year_ranges"][1] == {"value": "2000-2004", "count": 1}


def test_year_facets_endpoint_respects_top_k_param(client):
    app, tc = client
    pts = [_point(f"{1960 + i}-{1964 + i}") for i in range(20)]
    _attach_qdrant(app, ["test_col"], {"test_col": pts})
    r = tc.get("/api/v1/library/year-facets?top_k=5")
    assert r.status_code == 200
    assert len(r.json()["year_ranges"]) == 5


def test_year_facets_endpoint_filters_by_collection_name(client):
    app, tc = client
    _attach_qdrant(
        app,
        ["col_a", "col_b"],
        {
            "col_a": [_point("1970-1974")],
            "col_b": [_point("2010-2014")],
        },
    )
    r = tc.get("/api/v1/library/year-facets?collection_name=col_b")
    assert r.status_code == 200
    yr = {item["value"]: item["count"] for item in r.json()["year_ranges"]}
    assert "1970-1974" not in yr
    assert yr["2010-2014"] == 1


def test_year_facets_endpoint_returns_empty_without_db_client(client):
    app, tc = client
    app.state.db_client = None
    r = tc.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json() == {"year_ranges": []}
