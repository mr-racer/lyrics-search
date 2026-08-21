"""Persistence of the per-account folder-indexing grant.

The grant has to survive on the user row rather than in instance settings,
because "who may index the server folder" is a property of an account, not of
the instance — that conflation is what let every member index /music.
"""
import time

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.auth_service import _row_to_user


@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    import app.resources.metadata_db as mod

    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "index_grant_test.db")
    MetadataDB._instance = None
    MetadataDB.init()
    yield
    MetadataDB.close()


def _make_user(uid: str = "u1", role: str = "member") -> str:
    MetadataDB.create_user(
        user_id=uid,
        email=f"{uid}@example.test",
        password_hash="hash",
        role=role,
        created_at=time.time(),
    )
    return uid


def test_a_fresh_account_has_no_index_grant():
    uid = _make_user()

    assert MetadataDB.get_user_by_id(uid)["index_root"] is None


def test_setting_a_grant_persists_it():
    uid = _make_user()

    assert MetadataDB.set_index_root(uid, "/music") is True
    assert MetadataDB.get_user_by_id(uid)["index_root"] == "/music"


def test_setting_an_empty_grant_revokes_it():
    uid = _make_user()
    MetadataDB.set_index_root(uid, "/music")

    MetadataDB.set_index_root(uid, "")

    assert MetadataDB.get_user_by_id(uid)["index_root"] is None


def test_setting_a_grant_on_a_missing_account_reports_failure():
    assert MetadataDB.set_index_root("ghost", "/music") is False


def test_the_grant_is_visible_by_email_lookup_too():
    uid = _make_user()
    MetadataDB.set_index_root(uid, "/music")

    assert MetadataDB.get_user_by_email(f"{uid}@example.test")["index_root"] == "/music"


def test_the_grant_reaches_the_user_model():
    uid = _make_user()
    MetadataDB.set_index_root(uid, "/music")

    user = _row_to_user(MetadataDB.get_user_by_id(uid))

    assert user.index_root == "/music"


def test_a_user_model_without_a_grant_exposes_none():
    uid = _make_user()

    user = _row_to_user(MetadataDB.get_user_by_id(uid))

    assert user.index_root is None
