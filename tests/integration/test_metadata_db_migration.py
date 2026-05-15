"""Migration tests: MusicBrainz scaffold columns are added idempotently."""

import sqlite3
from pathlib import Path

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def test_musicbrainz_columns_present_on_fresh_init(tmp_path, monkeypatch):
    """Fresh DB init should add the MusicBrainz columns."""
    db_path = tmp_path / "musix.db"
    monkeypatch.setattr(mod, "DB_PATH", db_path)

    MetadataDB._reset_for_tests()  # implemented in this task
    MetadataDB.init()

    cols = _columns(db_path, "songs")
    assert {"producers", "label", "samples_json", "mbid"}.issubset(cols)

    art_cols = _columns(db_path, "artists")
    assert "mbid" in art_cols


def test_musicbrainz_migration_idempotent(tmp_path, monkeypatch):
    """Running init() twice must not raise (no duplicate column errors)."""
    db_path = tmp_path / "musix.db"
    monkeypatch.setattr(mod, "DB_PATH", db_path)

    MetadataDB._reset_for_tests()
    MetadataDB.init()
    # Second init: should be a no-op for migrations.
    MetadataDB._reset_for_tests()
    MetadataDB.init()  # must not raise

    cols = _columns(db_path, "songs")
    assert {"producers", "label", "samples_json", "mbid"}.issubset(cols)


def test_migration_preserves_existing_data(tmp_path, monkeypatch):
    """Adding columns to a pre-existing songs row must not lose data."""
    db_path = tmp_path / "musix.db"
    monkeypatch.setattr(mod, "DB_PATH", db_path)

    # Phase 1: init without the migration (simulate pre-Plan-3 state).
    # The pre-Plan-3 songs table has the real columns but lacks the MB columns
    # (producers, label, samples_json, mbid) that this migration adds.
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE songs (
          song_key TEXT PRIMARY KEY,
          title TEXT,
          artist TEXT,
          collection_name TEXT
        )
    """)
    conn.execute("INSERT INTO songs (song_key, title, artist) VALUES ('s1', 'T', 'A')")
    conn.commit()
    conn.close()

    # Phase 2: run init() which should add columns without touching the row.
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT title, artist, producers, label, samples_json, mbid FROM songs").fetchone()
        assert row[0] == "T"
        assert row[1] == "A"
        assert row[2] is None  # producers — not populated
        assert row[3] is None  # label
        assert row[4] is None  # samples_json
        assert row[5] is None  # mbid
    finally:
        conn.close()
