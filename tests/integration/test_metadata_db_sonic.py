"""Tests for Sonic Descriptor columns on songs table."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Patch DB_PATH to a temp file, reset singleton, init schema."""
    db_path = tmp_path / "metadata_test.db"
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_path)
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB._instance = None


def test_columns_present(temp_db):
    conn = MetadataDB.get()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)")}
    assert "sonic_tags_json" in cols
    assert "sonic_class" in cols
    assert "sonic_class_confidence" in cols
    assert "audio_signature" in cols


def test_upsert_and_get_sonic_descriptor(temp_db):
    MetadataDB.upsert_artist("radiohead", "Radiohead", "test_collection")
    MetadataDB.upsert_song("karma-police", "Karma Police", "radiohead", "test_collection")
    tags = [{"tag": "anxious", "score": 0.72}, {"tag": "atmospheric", "score": 0.68}]
    MetadataDB.upsert_sonic_descriptor(
        song_slug="karma-police",
        tags=tags,
        sonic_class="Indie melancholic",
        confidence=0.81,
    )
    desc = MetadataDB.get_sonic_descriptor("karma-police")
    assert desc["sonic_class"] == "Indie melancholic"
    assert desc["sonic_class_confidence"] == 0.81
    assert desc["tags"] == tags


def test_get_sonic_descriptor_returns_none_for_unknown(temp_db):
    desc = MetadataDB.get_sonic_descriptor("never-existed")
    assert desc is None


def test_upsert_sonic_descriptor_overwrites_previous(temp_db):
    MetadataDB.upsert_artist("a", "A", "c")
    MetadataDB.upsert_song("s", "S", "a", "c")
    MetadataDB.upsert_sonic_descriptor("s", tags=[{"tag": "warm", "score": 0.5}], sonic_class="X", confidence=0.4)
    MetadataDB.upsert_sonic_descriptor("s", tags=[{"tag": "cold", "score": 0.9}], sonic_class="Y", confidence=0.9)
    desc = MetadataDB.get_sonic_descriptor("s")
    assert desc["tags"] == [{"tag": "cold", "score": 0.9}]
    assert desc["sonic_class"] == "Y"
