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


def test_get_instance_config_returns_none_when_empty():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.commit()
    assert MetadataDB.get_instance_config() is None


def test_set_and_get_instance_config():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.commit()
    MetadataDB.set_instance_config(mode="sharing", created_at=1700000000.0)
    row = MetadataDB.get_instance_config()
    assert row == {"mode": "sharing", "created_at": 1700000000.0}


def test_set_instance_config_rejects_second_write():
    """Per spec §4.2: mode is locked at first write. Second set_ raises."""
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.commit()
    MetadataDB.set_instance_config(mode="sharing", created_at=1700000000.0)
    try:
        MetadataDB.set_instance_config(mode="server", created_at=1700000100.0)
        raise AssertionError("expected IntegrityError on second instance_config write")
    except sqlite3.IntegrityError:
        pass
