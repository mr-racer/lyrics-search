"""Smoke test for scripts/backfill_sonic_payload.py — exercises the core
collection_iter → set_payload logic against a mocked Qdrant client."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    from app.resources.metadata_db import MetadataDB
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


def test_backfill_writes_sonic_tags_to_qdrant_payload(db):
    from app.resources.metadata_db import MetadataDB
    # slug = _slugify("A") + "-" + _slugify("X") = "a-x"
    song_slug = "a-x"
    MetadataDB.upsert_artist("a", "A", "test_col")
    MetadataDB.upsert_song(song_slug, "X", "a", "test_col")
    MetadataDB.upsert_sonic_descriptor(
        song_slug=song_slug,
        tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
    )

    qdrant = MagicMock()
    qdrant.scroll.return_value = (
        [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
        None,  # next page offset
    )

    from scripts.backfill_sonic_payload import backfill_collection
    n = backfill_collection(qdrant, collection="test_col", dry_run=False)

    assert n == 1
    qdrant.set_payload.assert_called_once()
    _, kwargs = qdrant.set_payload.call_args
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["points"] == ["point-1"]
    assert sorted(kwargs["payload"]["sonic_tags"]) == ["lo-fi", "melancholic"]


def test_backfill_skips_when_no_sqlite_row(db):
    qdrant = MagicMock()
    qdrant.scroll.return_value = (
        [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
        None,
    )

    from scripts.backfill_sonic_payload import backfill_collection
    n = backfill_collection(qdrant, collection="test_col", dry_run=False)

    assert n == 0
    qdrant.set_payload.assert_not_called()


def test_backfill_dry_run_makes_no_writes(db):
    from app.resources.metadata_db import MetadataDB
    song_slug = "a-x"
    MetadataDB.upsert_artist("a", "A", "test_col")
    MetadataDB.upsert_song(song_slug, "X", "a", "test_col")
    MetadataDB.upsert_sonic_descriptor(
        song_slug=song_slug,
        tags=[{"tag": "melancholic", "score": 0.8}],
    )

    qdrant = MagicMock()
    qdrant.scroll.return_value = (
        [MagicMock(id="point-1", payload={"artist": "A", "title": "X"})],
        None,
    )

    from scripts.backfill_sonic_payload import backfill_collection
    n = backfill_collection(qdrant, collection="test_col", dry_run=True)

    assert n == 1  # counts what _would_ be written
    qdrant.set_payload.assert_not_called()
