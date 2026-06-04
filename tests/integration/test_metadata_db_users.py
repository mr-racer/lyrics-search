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


def test_create_user_and_get_by_email():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    conn.commit()
    MetadataDB.create_user(
        user_id="uid-1",
        email="alice@example.com",
        password_hash="$argon2id$...",
        role="owner",
        created_at=1700000000.0,
    )
    row = MetadataDB.get_user_by_email("alice@example.com")
    assert row is not None
    assert row["id"] == "uid-1"
    assert row["email"] == "alice@example.com"
    assert row["role"] == "owner"
    assert row["password_hash"] == "$argon2id$..."
    assert row["last_login_at"] is None


def test_create_user_normalizes_email_case():
    """Caller normalizes case; verify storage preserves what was passed."""
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    conn.commit()
    MetadataDB.create_user(
        user_id="uid-2", email="bob@example.com", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    # Lookup with same casing returns the row
    assert MetadataDB.get_user_by_email("bob@example.com") is not None
    # Lookup with different casing returns None (caller's responsibility to normalize)
    assert MetadataDB.get_user_by_email("BOB@example.com") is None


def test_get_user_by_id_round_trip():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    conn.commit()
    MetadataDB.create_user(
        user_id="uid-3", email="c@d.e", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    row = MetadataDB.get_user_by_id("uid-3")
    assert row is not None and row["email"] == "c@d.e"
    assert MetadataDB.get_user_by_id("missing") is None


def test_update_last_login_sets_timestamp():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM users")
    conn.commit()
    MetadataDB.create_user(
        user_id="uid-4", email="z@y.x", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    assert MetadataDB.get_user_by_id("uid-4")["last_login_at"] is None
    MetadataDB.update_last_login("uid-4", 1700001234.5)
    assert MetadataDB.get_user_by_id("uid-4")["last_login_at"] == 1700001234.5
