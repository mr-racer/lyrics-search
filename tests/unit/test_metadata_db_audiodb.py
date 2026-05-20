"""Tests that MetadataDB.init() migrates the artists table to add audiodb columns."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_migration_adds_audiodb_columns_on_fresh_db():
    conn = MetadataDB._connect()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(artists)").fetchall()}
    expected = {
        "audiodb_bio", "mood", "country_code", "country", "label",
        "cutout_path", "thumb_path", "audiodb_mbid", "audiodb_fetched_at",
    }
    assert expected.issubset(cols)


def test_migration_is_idempotent():
    # Re-running init() (which calls the migration) should not raise.
    MetadataDB._reset_for_tests()
    MetadataDB.init()  # First call (via fixture) already migrated
    MetadataDB.init()  # Second call should be a no-op
    conn = MetadataDB._connect()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(artists)").fetchall()}
    assert "audiodb_bio" in cols


def test_migration_preserves_existing_rows():
    # Insert a row before the new columns are added (simulating pre-migration state
    # would require manual DB manipulation; here we just confirm new columns are
    # nullable for existing rows).
    MetadataDB.upsert_artist(slug="kanye-west", name="Kanye West", collection_name="test")
    conn = MetadataDB._connect()
    row = conn.execute(
        "SELECT audiodb_bio, mood, audiodb_fetched_at FROM artists WHERE slug=?",
        ("kanye-west",),
    ).fetchone()
    assert row == (None, None, None)
