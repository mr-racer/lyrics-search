"""Verify Phase A instance_config table exists and is single-row."""
import sqlite3
from app.resources.metadata_db import MetadataDB


def test_instance_config_table_exists_after_init():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='instance_config'"
    ).fetchone()
    assert row is not None


def test_instance_config_single_row_enforced():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.execute(
        "INSERT INTO instance_config (id, mode, created_at) VALUES (1, 'sharing', 1700000000)"
    )
    try:
        conn.execute(
            "INSERT INTO instance_config (id, mode, created_at) VALUES (2, 'server', 1700000001)"
        )
        conn.commit()
        raise AssertionError("expected IntegrityError — only id=1 row allowed")
    except sqlite3.IntegrityError:
        pass
    conn.execute("DELETE FROM instance_config")
    conn.commit()


def test_instance_config_mode_check_constraint():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    try:
        conn.execute(
            "INSERT INTO instance_config (id, mode, created_at) VALUES (1, 'invalid', 1700000000)"
        )
        conn.commit()
        raise AssertionError("expected IntegrityError on invalid mode")
    except sqlite3.IntegrityError:
        pass
    conn.execute("DELETE FROM instance_config")
    conn.commit()
