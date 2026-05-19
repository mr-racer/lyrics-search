"""Unit test for MetadataDB.get_sonic_facets — aggregate counts of top-K
sonic_tags from the songs table."""
from __future__ import annotations

import json
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


def test_get_sonic_facets_returns_empty_for_empty_db(db):
    out = MetadataDB.get_sonic_facets()
    assert out == {"tags": []}


def test_get_sonic_facets_aggregates_tags_from_json(db):
    # song row must exist before upsert_sonic_descriptor (UPDATE WHERE slug=?).
    db.upsert_artist("a", "A", "test_col")
    db.upsert_song("a-1", "T1", "a", "test_col")
    db.upsert_song("a-2", "T2", "a", "test_col")
    db.upsert_sonic_descriptor(
        song_slug="a-1",
        tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
    )
    db.upsert_sonic_descriptor(
        song_slug="a-2",
        tags=[{"tag": "melancholic", "score": 0.9}],
    )
    out = MetadataDB.get_sonic_facets()
    tag_counts = {t["value"]: t["count"] for t in out["tags"]}
    assert tag_counts == {"melancholic": 2, "lo-fi": 1}


def test_get_sonic_facets_sorts_tags_descending_by_count(db):
    db.upsert_artist("a", "A", "test_col")
    db.upsert_song("a-1", "T1", "a", "test_col")
    db.upsert_song("b-1", "T2", "a", "test_col")
    db.upsert_song("b-2", "T3", "a", "test_col")
    db.upsert_song("b-3", "T4", "a", "test_col")
    db.upsert_sonic_descriptor(
        song_slug="a-1",
        tags=[{"tag": "rare", "score": 0.5}],
    )
    for slug in ["b-1", "b-2", "b-3"]:
        db.upsert_sonic_descriptor(
            song_slug=slug,
            tags=[{"tag": "common", "score": 0.5}],
        )
    out = MetadataDB.get_sonic_facets()
    assert out["tags"][0]["value"] == "common"
    assert out["tags"][0]["count"] == 3
    assert out["tags"][1]["value"] == "rare"


def test_get_sonic_facets_tolerates_malformed_tags_json(db):
    """A bad sonic_tags_json row should not crash the aggregate."""
    db.upsert_artist("a", "A", "test_col")
    db.upsert_song("a-1", "T1", "a", "test_col")
    # Directly write a malformed value via raw SQL since upsert_sonic_descriptor json.dumps's it.
    conn = db._connect()
    conn.execute("UPDATE songs SET sonic_tags_json = ? WHERE slug = ?", ("not-json", "a-1"))
    conn.commit()
    out = MetadataDB.get_sonic_facets()
    assert out["tags"] == []


def test_get_sonic_facets_caps_to_top_k(db):
    db.upsert_artist("a", "A", "test_col")
    for i in range(50):
        slug = f"a-{i}"
        db.upsert_song(slug, f"T{i}", "a", "test_col")
        db.upsert_sonic_descriptor(
            song_slug=slug,
            tags=[{"tag": f"tag-{i}", "score": 0.5}],
        )
    out = MetadataDB.get_sonic_facets(top_k=10)
    assert len(out["tags"]) == 10
