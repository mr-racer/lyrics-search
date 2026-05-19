"""Unit test for MetadataDB.get_year_facets — aggregate counts of year_range
values present in the songs table."""
from __future__ import annotations

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


def test_get_year_facets_returns_empty_for_empty_db(db):
    out = MetadataDB.get_year_facets()
    assert out == {"year_ranges": []}


def test_get_year_facets_counts_year_range_descending(db):
    db.upsert_artist("a", "A", "test_col")
    db.upsert_song("a-1", "T1", "a", "test_col")
    db.upsert_song("a-2", "T2", "a", "test_col")
    db.upsert_song("a-3", "T3", "a", "test_col")
    conn = db._connect()
    conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", ("1990-1994", "a-1"))
    conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", ("1990-1994", "a-2"))
    conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", ("2000-2004", "a-3"))
    conn.commit()
    out = MetadataDB.get_year_facets()
    assert out["year_ranges"] == [
        {"value": "1990-1994", "count": 2},
        {"value": "2000-2004", "count": 1},
    ]


def test_get_year_facets_skips_null_year_range(db):
    db.upsert_artist("a", "A", "test_col")
    db.upsert_song("a-1", "T1", "a", "test_col")
    db.upsert_song("a-2", "T2", "a", "test_col")
    conn = db._connect()
    conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", ("1990-1994", "a-1"))
    # a-2 stays with year_range = NULL
    conn.commit()
    out = MetadataDB.get_year_facets()
    assert out["year_ranges"] == [{"value": "1990-1994", "count": 1}]


def test_get_year_facets_caps_to_top_k(db):
    db.upsert_artist("a", "A", "test_col")
    conn = db._connect()
    for i in range(20):
        slug = f"a-{i}"
        db.upsert_song(slug, f"T{i}", "a", "test_col")
        conn.execute("UPDATE songs SET year_range = ? WHERE slug = ?", (f"{1990+i}-{1994+i}", slug))
    conn.commit()
    out = MetadataDB.get_year_facets(top_k=5)
    assert len(out["year_ranges"]) == 5
