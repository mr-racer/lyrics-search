"""Unit tests for artist_bios table + accessors."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_artist_bios_table_exists():
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)")]
    assert set(cols) == {"artist_slug", "collection_name", "lang", "bio_text", "generated_at"}


def test_artist_bios_pk():
    """PRIMARY KEY is (artist_slug, collection_name, lang) — same shape as sonic_vibes."""
    conn = MetadataDB.get()
    pk_cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)") if r[5] > 0]
    assert pk_cols == ["artist_slug", "collection_name", "lang"]
