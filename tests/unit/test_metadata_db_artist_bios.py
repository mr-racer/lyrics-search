"""Unit tests for artist_bios table + accessors."""
from app.resources.metadata_db import MetadataDB


def test_artist_bios_table_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.MetadataDB._instance", None)
    MetadataDB.init()
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)")]
    assert set(cols) == {"artist_slug", "collection_name", "lang", "bio_text", "generated_at"}


def test_artist_bios_pk(tmp_path, monkeypatch):
    """PRIMARY KEY is (artist_slug, collection_name, lang) — same shape as sonic_vibes."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.MetadataDB._instance", None)
    MetadataDB.init()
    conn = MetadataDB.get()
    pk_cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)") if r[5] > 0]
    assert pk_cols == ["artist_slug", "collection_name", "lang"]
