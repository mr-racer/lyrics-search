"""E2E smoke for Phase D — cross-account data isolation.

Proves the server-derived collection (acct_<user.id>) keeps one account's data
invisible to another, end-to-end through the real reaction endpoints + a real
MetadataDB. The client NEVER supplies collection_name — the server derives it
from the JWT user, so two users hitting the same track_id land in different
collections.

The frontend half (per-user localStorage chat/search isolation) is covered by
tests/unit/test_localstorage_migration.mjs (run via node).
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import search as search_route
from app.resources.metadata_db import MetadataDB

_USER_A = SimpleNamespace(id="user-A", email="a@x", role="member")
_USER_B = SimpleNamespace(id="user-B", email="b@x", role="member")


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "iso.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    yield app
    MetadataDB._reset_for_tests()


def _login_as(app, user):
    app.dependency_overrides[search_route.get_current_user] = lambda: user


def test_reaction_does_not_leak_across_accounts(app_db):
    app = app_db
    c = TestClient(app)

    # ── User A likes T1 (no collection_name in the body — server derives it) ──
    _login_as(app, _USER_A)
    r = c.post("/api/v1/search/tracks/T1/reaction", json={"reaction": "like"})
    assert r.status_code == 200
    assert r.json()["collection_name"] == "acct_user-A"

    # User A reads their own reaction back
    r = c.get("/api/v1/search/tracks/T1/reaction")
    assert r.status_code == 200
    assert r.json()["reaction"] == "like"

    # ── User B asks about the SAME track id → sees nothing (different collection) ──
    _login_as(app, _USER_B)
    r = c.get("/api/v1/search/tracks/T1/reaction")
    assert r.status_code == 200
    assert r.json()["reaction"] != "like"
    assert r.json()["collection_name"] == "acct_user-B"

    # ── Sanity: B's read did not clobber A's data ──
    _login_as(app, _USER_A)
    r = c.get("/api/v1/search/tracks/T1/reaction")
    assert r.json()["reaction"] == "like"


def test_user_b_like_is_independent(app_db):
    """B liking the same track keeps A's and B's reactions independent."""
    app = app_db
    c = TestClient(app)

    _login_as(app, _USER_A)
    c.post("/api/v1/search/tracks/T1/reaction", json={"reaction": "like"})

    # B dislikes the same track id
    _login_as(app, _USER_B)
    c.post("/api/v1/search/tracks/T1/reaction", json={"reaction": "dislike"})

    # Each account sees only its own reaction
    r = c.get("/api/v1/search/tracks/T1/reaction")
    assert r.json()["reaction"] == "dislike"

    _login_as(app, _USER_A)
    r = c.get("/api/v1/search/tracks/T1/reaction")
    assert r.json()["reaction"] == "like"
