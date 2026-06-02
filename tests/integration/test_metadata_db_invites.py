"""Verify Phase A invites table exists after MetadataDB.init()."""
from app.resources.metadata_db import MetadataDB


def test_invites_table_exists_after_init():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='invites'"
    ).fetchone()
    assert row is not None


def test_invites_unused_partial_index_exists():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_invites_unused'"
    ).fetchone()
    assert row is not None
