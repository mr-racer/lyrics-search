"""Verify Phase A users table exists after MetadataDB.init()."""
import sqlite3
from app.resources.metadata_db import MetadataDB


def test_users_table_exists_after_init():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    assert row is not None, "users table should be created by MetadataDB.init()"


def test_users_email_unique_constraint():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    conn.execute(
        "INSERT INTO users (id, email, password_hash, role, created_at) "
        "VALUES ('id-1', 'owner@example.com', 'h', 'owner', 1700000000)"
    )
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) "
            "VALUES ('id-2', 'owner@example.com', 'h', 'member', 1700000001)"
        )
        conn.commit()
        raise AssertionError("expected IntegrityError on duplicate email")
    except sqlite3.IntegrityError:
        pass
    conn.execute("DELETE FROM users")
    conn.commit()


def test_users_role_check_constraint():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) "
            "VALUES ('id-x', 'x@y.z', 'h', 'admin', 1700000000)"
        )
        conn.commit()
        raise AssertionError("expected IntegrityError on invalid role")
    except sqlite3.IntegrityError:
        pass
    conn.execute("DELETE FROM users")
    conn.commit()
