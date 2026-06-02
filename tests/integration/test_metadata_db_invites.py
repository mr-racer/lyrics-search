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


import sqlite3


def _seed_owner():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM invites")
    conn.execute("DELETE FROM users")
    MetadataDB.create_user(
        user_id="owner-1", email="o@x.y", password_hash="h",
        role="owner", created_at=1700000000.0,
    )


def test_create_invite_round_trip():
    _seed_owner()
    MetadataDB.create_invite(
        code="abc123XYZ_-x",
        created_by="owner-1",
        created_at=1700000000.0,
        expires_at=1700000000.0 + 7 * 86400,
    )
    row = MetadataDB.get_invite("abc123XYZ_-x")
    assert row is not None
    assert row["created_by"] == "owner-1"
    assert row["expires_at"] - row["created_at"] == 7 * 86400
    assert row["consumed_by"] is None
    assert row["consumed_at"] is None


def test_get_invite_unknown_code_returns_none():
    _seed_owner()
    assert MetadataDB.get_invite("nope") is None


def test_consume_invite_marks_row():
    _seed_owner()
    MetadataDB.create_user(
        user_id="member-1", email="m@x.y", password_hash="h",
        role="member", created_at=1700000100.0,
    )
    MetadataDB.create_invite(
        code="code1",
        created_by="owner-1",
        created_at=1700000000.0,
        expires_at=1700000000.0 + 86400,
    )
    MetadataDB.consume_invite("code1", consumed_by="member-1", consumed_at=1700000150.0)
    row = MetadataDB.get_invite("code1")
    assert row["consumed_by"] == "member-1"
    assert row["consumed_at"] == 1700000150.0


def test_list_invites_filters_consumed():
    _seed_owner()
    MetadataDB.create_user(
        user_id="m1", email="m1@x.y", password_hash="h",
        role="member", created_at=1700000100.0,
    )
    MetadataDB.create_invite("active", "owner-1", 1700000000.0, 1700100000.0)
    MetadataDB.create_invite("used",   "owner-1", 1700000010.0, 1700100010.0)
    MetadataDB.consume_invite("used", consumed_by="m1", consumed_at=1700000150.0)

    open_only = MetadataDB.list_invites(include_consumed=False)
    assert [r["code"] for r in open_only] == ["active"]

    everything = MetadataDB.list_invites(include_consumed=True)
    assert sorted(r["code"] for r in everything) == ["active", "used"]


def test_delete_invite_removes_row():
    _seed_owner()
    MetadataDB.create_invite("kill", "owner-1", 1700000000.0, 1700100000.0)
    assert MetadataDB.get_invite("kill") is not None
    MetadataDB.delete_invite("kill")
    assert MetadataDB.get_invite("kill") is None
