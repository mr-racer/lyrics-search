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


def test_upsert_audiodb_inserts_new_row():
    # No prior artist row; upsert should INSERT one with name=slug fallback.
    MetadataDB.upsert_artist_audiodb(
        slug="dua-lipa", collection_name="test",
        audiodb_bio="Dua Lipa is a singer.", mood="energetic",
        country_code="GB", country="London, UK", label="Warner",
        cutout_path="/covers/artists/abc.png",
        thumb_path="/covers/artists/def.png",
        audiodb_mbid="12345",
    )
    row = MetadataDB.get_artist_audiodb("dua-lipa", "test")
    assert row["audiodb_bio"] == "Dua Lipa is a singer."
    assert row["mood"] == "energetic"
    assert row["country_code"] == "GB"
    assert row["audiodb_mbid"] == "12345"
    assert row["audiodb_fetched_at"]  # ISO string, truthy


def test_upsert_audiodb_updates_existing_row():
    MetadataDB.upsert_artist(slug="kanye-west", name="Kanye West", collection_name="test")
    MetadataDB.upsert_artist_audiodb(
        slug="kanye-west", collection_name="test",
        audiodb_bio="bio v1", mood="introspective",
        country_code="US", country="Chicago, USA", label=None,
        cutout_path=None, thumb_path=None, audiodb_mbid=None,
    )
    # Update with new bio
    MetadataDB.upsert_artist_audiodb(
        slug="kanye-west", collection_name="test",
        audiodb_bio="bio v2", mood="introspective",
        country_code="US", country="Chicago, USA", label=None,
        cutout_path=None, thumb_path=None, audiodb_mbid=None,
    )
    row = MetadataDB.get_artist_audiodb("kanye-west", "test")
    assert row["audiodb_bio"] == "bio v2"


def test_get_audiodb_returns_none_when_no_row():
    assert MetadataDB.get_artist_audiodb("ghost-artist", "test") is None


def test_get_audiodb_returns_none_when_artist_has_no_audiodb_data():
    # Row exists from upsert_artist but audiodb_fetched_at is NULL — get returns dict
    # with all None values, NOT None itself, so the caller can distinguish missing-row
    # from never-fetched-row.
    MetadataDB.upsert_artist(slug="nobody", name="Nobody", collection_name="test")
    row = MetadataDB.get_artist_audiodb("nobody", "test")
    assert row is not None
    assert row["audiodb_bio"] is None
    assert row["audiodb_fetched_at"] is None


def test_upsert_audiodb_handles_cross_collection_slug_collision():
    """Same slug in two collections: the audiodb data overwrites in place rather than
    raising UNIQUE constraint failed: artists.slug."""
    # Collection A — slug exists with name set
    MetadataDB.upsert_artist(slug="multi-coll", name="Multi Coll", collection_name="A")
    # Collection B — same slug, audiodb enrichment should not raise
    MetadataDB.upsert_artist_audiodb(
        slug="multi-coll", collection_name="B",
        audiodb_bio="Bio from B", mood="reflective",
        country_code="US", country=None, label=None,
        cutout_path=None, thumb_path=None, audiodb_mbid=None,
    )
    # Row exists (the schema is single-PK, so it's the same row regardless of collection)
    row = MetadataDB.get_artist_audiodb("multi-coll", "A")
    # Note: the ON CONFLICT updates the row regardless of which collection it was queried
    # under, so the bio is "Bio from B" even when fetched with collection=A.
    assert row is not None
    assert row["audiodb_bio"] == "Bio from B"
