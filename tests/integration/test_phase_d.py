"""Phase D — cross-account collection derivation & data isolation.

Consolidated from:
  - test_e2e_phase_d_isolation.py            -> TestPhaseDIsolation
  - test_phase_d_recommend_derive_collection -> TestPhaseDRecommendDeriveCollection

Both halves prove the server derives the per-account Qdrant collection
(acct_<user.id>) from the JWT user and ignores / does not require any
client-supplied collection name, so one account's data stays invisible to
another.

The frontend half (per-user localStorage chat/search isolation) is covered by
tests/unit/test_localstorage_migration.mjs (run via node).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import recommend as rec_route
from app.api.routes import search as search_route
from app.domain.models import (
    AutoplayQueueDiagnostics,
    AutoplayQueueResponse,
)
from app.resources.metadata_db import MetadataDB
from app.services import autoplay_service, stream_service

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


def _autoplay_response(seed_id: str = "X") -> AutoplayQueueResponse:
    return AutoplayQueueResponse(
        seed_track_id=seed_id,
        tracks=[],
        diagnostics=AutoplayQueueDiagnostics(
            candidates_fetched=0,
            dropped_excluded=0,
            dropped_disliked=0,
            dropped_diversity=0,
            returned=0,
        ),
    )


def _stream_response() -> dict:
    return {"tracks": [], "diagnostics": {}}


class TestPhaseDIsolation:
    """E2E smoke for Phase D — cross-account data isolation.

    Proves the server-derived collection (acct_<user.id>) keeps one account's
    data invisible to another, end-to-end through the real reaction endpoints +
    a real MetadataDB. The client NEVER supplies collection_name — the server
    derives it from the JWT user, so two users hitting the same track_id land in
    different collections.
    """

    def test_reaction_does_not_leak_across_accounts(self, app_db):
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

    def test_user_b_like_is_independent(self, app_db):
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


class TestPhaseDRecommendDeriveCollection:
    """Phase D-soft: /recommend/* endpoints derive collection from JWT, ignore supplied value."""

    # -----------------------------------------------------------------------
    # /recommend/autoplay-queue
    # -----------------------------------------------------------------------

    def test_autoplay_ignores_supplied_collection(self):
        """Endpoint must use acct_<user.id> regardless of the supplied collection param."""
        from app.services import autoplay_service

        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[rec_route.get_current_user] = lambda: fixed
        captured = {}

        def fake_next_queue(**kwargs):
            captured.update(kwargs)
            return _autoplay_response("X")

        with patch.object(autoplay_service, "next_queue", side_effect=fake_next_queue):
            with TestClient(app) as c:
                c.app.state.db_client = MagicMock()
                c.get("/api/v1/recommend/autoplay-queue?collection=acct_BAD&seed_track_id=X")

        assert captured["collection_name"] == "acct_user-A"

    def test_autoplay_derives_collection_when_no_collection_supplied(self):
        """When collection is omitted entirely, still derives from JWT."""
        from app.services import autoplay_service

        fixed = SimpleNamespace(id="user-B", email="b@x")
        app = create_app()
        app.dependency_overrides[rec_route.get_current_user] = lambda: fixed
        captured = {}

        def fake_next_queue(**kwargs):
            captured.update(kwargs)
            return _autoplay_response("Y")

        with patch.object(autoplay_service, "next_queue", side_effect=fake_next_queue):
            with TestClient(app) as c:
                c.app.state.db_client = MagicMock()
                c.get("/api/v1/recommend/autoplay-queue?seed_track_id=Y")

        assert captured["collection_name"] == "acct_user-B"

    # -----------------------------------------------------------------------
    # /recommend/stream/next
    # -----------------------------------------------------------------------

    def test_stream_next_derives_collection_from_jwt(self):
        """The stream endpoint takes no collection param at all (D-hard) — the
        service must be called with acct_<user.id>."""
        fixed = SimpleNamespace(id="user-C", email="c@x")
        app = create_app()
        app.dependency_overrides[rec_route.get_current_user] = lambda: fixed
        captured = {}

        def fake_next_chunk(**kwargs):
            captured.update(kwargs)
            return _stream_response()

        with patch.object(stream_service, "next_chunk", side_effect=fake_next_chunk), \
             patch.object(rec_route.MetadataDB, "get_stream_liked_share", return_value=None):
            with TestClient(app) as c:
                c.app.state.db_client = MagicMock()
                c.get("/api/v1/recommend/stream/next?session_id=s1")

        assert captured["collection_name"] == "acct_user-C"

    # -----------------------------------------------------------------------
    # /recommend/sonic-sibling
    # -----------------------------------------------------------------------

    def test_sonic_sibling_still_returns_501_with_optional_collection(self):
        """Relaxed collection param must not change the 501 behaviour."""
        fixed = SimpleNamespace(id="user-E", email="e@x")
        app = create_app()
        app.dependency_overrides[rec_route.get_current_user] = lambda: fixed

        with TestClient(app) as c:
            resp = c.get("/api/v1/recommend/sonic-sibling?track_id=t1&collection=whatever")

        assert resp.status_code == 501

    def test_sonic_sibling_returns_501_without_collection(self):
        """collection param is now optional — omitting it must still give 501."""
        fixed = SimpleNamespace(id="user-F", email="f@x")
        app = create_app()
        app.dependency_overrides[rec_route.get_current_user] = lambda: fixed

        with TestClient(app) as c:
            resp = c.get("/api/v1/recommend/sonic-sibling?track_id=t1")

        assert resp.status_code == 501
