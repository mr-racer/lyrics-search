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


def test_set_and_get_artist_bio():
    MetadataDB.set_artist_bio("dua-lipa", "col_a", "en", "Born in London…")
    assert MetadataDB.get_artist_bio("dua-lipa", "col_a", "en") == "Born in London…"
    # other (slug, collection, lang) tuples isolated
    assert MetadataDB.get_artist_bio("dua-lipa", "col_a", "ru") is None
    assert MetadataDB.get_artist_bio("dua-lipa", "col_b", "en") is None


def test_set_artist_bio_upsert():
    """Same (slug, collection, lang) → overwrite, not duplicate row."""
    MetadataDB.set_artist_bio("x", "c", "en", "old")
    MetadataDB.set_artist_bio("x", "c", "en", "new")
    assert MetadataDB.get_artist_bio("x", "c", "en") == "new"
    conn = MetadataDB.get()
    n = conn.execute("SELECT COUNT(*) FROM artist_bios").fetchone()[0]
    assert n == 1


def test_delete_artist_bios():
    MetadataDB.set_artist_bio("a", "col_a", "en", "x")
    MetadataDB.set_artist_bio("b", "col_a", "en", "y")
    MetadataDB.set_artist_bio("c", "col_b", "en", "z")
    n = MetadataDB.delete_artist_bios("col_a")
    assert n == 2
    assert MetadataDB.get_artist_bio("a", "col_a", "en") is None
    # col_b row untouched
    assert MetadataDB.get_artist_bio("c", "col_b", "en") == "z"
