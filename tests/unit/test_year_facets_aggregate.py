"""Unit tests for GET /library/year-facets — aggregation from Qdrant payload.

Since the implementation moved out of MetadataDB into the route handler,
these tests mock the Qdrant client and exercise the scroll-and-count logic
via the FastAPI TestClient, matching the pattern in
tests/integration/test_backfill_sonic_payload.py.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api.main import create_app


def _make_point(yr: str | None) -> MagicMock:
    p = MagicMock()
    p.payload = {"year_range": yr} if yr else {}
    return p


def _make_qdrant(collections: list[str], pages: dict[str, list]) -> MagicMock:
    """Build a minimal Qdrant mock.

    ``pages`` maps collection_name -> flat list of points (single page, no pagination).
    """
    qdrant = MagicMock()

    col_objs = []
    for name in collections:
        obj = MagicMock()
        obj.name = name
        col_objs.append(obj)
    qdrant.get_collections.return_value.collections = col_objs

    def scroll_side_effect(collection_name, limit, with_payload, with_vectors, offset=None):
        pts = pages.get(collection_name, [])
        return (pts, None)  # single page, no next offset

    qdrant.scroll.side_effect = scroll_side_effect
    return qdrant


@pytest.fixture
def client_with_qdrant(tmp_path, monkeypatch):
    """Return a factory that creates a TestClient whose db_client.qdrant is mocked."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)

    from app.resources.metadata_db import MetadataDB
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    def _factory(qdrant_mock):
        app = create_app()
        db_client_mock = MagicMock()
        db_client_mock.qdrant = qdrant_mock
        app.state.db_client = db_client_mock
        return TestClient(app)

    yield _factory
    MetadataDB._reset_for_tests()


def test_year_facets_returns_empty_when_no_db_client():
    """Endpoint returns empty list gracefully when db_client is None."""
    app = create_app()
    app.state.db_client = None
    client = TestClient(app)
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json() == {"year_ranges": []}


def test_year_facets_returns_empty_when_qdrant_has_no_points(client_with_qdrant):
    qdrant = _make_qdrant(["col1"], {"col1": []})
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json() == {"year_ranges": []}


def test_year_facets_counts_and_sorts_descending(client_with_qdrant):
    qdrant = _make_qdrant(
        ["col1"],
        {
            "col1": [
                _make_point("1990-1994"),
                _make_point("1990-1994"),
                _make_point("2000-2004"),
            ]
        },
    )
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    data = r.json()
    assert data["year_ranges"][0] == {"value": "1990-1994", "count": 2}
    assert data["year_ranges"][1] == {"value": "2000-2004", "count": 1}


def test_year_facets_skips_null_year_range(client_with_qdrant):
    qdrant = _make_qdrant(
        ["col1"],
        {
            "col1": [
                _make_point("1990-1994"),
                _make_point(None),  # should be skipped
            ]
        },
    )
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    assert r.json()["year_ranges"] == [{"value": "1990-1994", "count": 1}]


def test_year_facets_aggregates_across_all_collections(client_with_qdrant):
    qdrant = _make_qdrant(
        ["col1", "col2"],
        {
            "col1": [_make_point("1980-1984")],
            "col2": [_make_point("1980-1984"), _make_point("2000-2004")],
        },
    )
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets")
    assert r.status_code == 200
    yr = {item["value"]: item["count"] for item in r.json()["year_ranges"]}
    assert yr["1980-1984"] == 2
    assert yr["2000-2004"] == 1


def test_year_facets_filters_to_collection_when_param_given(client_with_qdrant):
    qdrant = _make_qdrant(
        ["col1", "col2"],
        {
            "col1": [_make_point("1980-1984")],
            "col2": [_make_point("2000-2004")],
        },
    )
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets?collection_name=col2")
    assert r.status_code == 200
    yr = {item["value"]: item["count"] for item in r.json()["year_ranges"]}
    assert "1980-1984" not in yr
    assert yr["2000-2004"] == 1


def test_year_facets_caps_to_top_k(client_with_qdrant):
    points = [_make_point(f"{1960 + i}-{1964 + i}") for i in range(20)]
    qdrant = _make_qdrant(["col1"], {"col1": points})
    client = client_with_qdrant(qdrant)
    r = client.get("/api/v1/library/year-facets?top_k=5")
    assert r.status_code == 200
    assert len(r.json()["year_ranges"]) == 5
