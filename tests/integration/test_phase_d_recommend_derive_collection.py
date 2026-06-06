"""Phase D-soft: /recommend/* endpoints derive collection from JWT, ignore supplied value."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import recommend as rec_route
from app.services import autoplay_service, personalization_service
from app.domain.models import (
    AutoplayQueueDiagnostics,
    AutoplayQueueResponse,
    ForYouSeedResponse,
)


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


def _for_you_response() -> ForYouSeedResponse:
    return ForYouSeedResponse(
        seed_track_id=None,
        track=None,
        source="random",
        collection_name=None,
    )


# ---------------------------------------------------------------------------
# /recommend/autoplay-queue
# ---------------------------------------------------------------------------

def test_autoplay_ignores_supplied_collection():
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


def test_autoplay_derives_collection_when_no_collection_supplied():
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


# ---------------------------------------------------------------------------
# /recommend/for-you-seed
# ---------------------------------------------------------------------------

def test_for_you_seed_ignores_supplied_collection():
    """Endpoint must use acct_<user.id> regardless of the supplied collection param."""
    from app.services import personalization_service

    fixed = SimpleNamespace(id="user-C", email="c@x")
    app = create_app()
    app.dependency_overrides[rec_route.get_current_user] = lambda: fixed
    captured = {}

    def fake_pick(**kwargs):
        captured.update(kwargs)
        return _for_you_response()

    with patch.object(personalization_service, "pick_for_you_seed", side_effect=fake_pick):
        with TestClient(app) as c:
            c.app.state.db_client = MagicMock()
            c.get("/api/v1/recommend/for-you-seed?collection=acct_WRONG")

    assert captured["collection_name"] == "acct_user-C"


def test_for_you_seed_derives_collection_when_no_collection_supplied():
    """When collection is omitted entirely, still derives from JWT."""
    from app.services import personalization_service

    fixed = SimpleNamespace(id="user-D", email="d@x")
    app = create_app()
    app.dependency_overrides[rec_route.get_current_user] = lambda: fixed
    captured = {}

    def fake_pick(**kwargs):
        captured.update(kwargs)
        return _for_you_response()

    with patch.object(personalization_service, "pick_for_you_seed", side_effect=fake_pick):
        with TestClient(app) as c:
            c.app.state.db_client = MagicMock()
            c.get("/api/v1/recommend/for-you-seed")

    assert captured["collection_name"] == "acct_user-D"


# ---------------------------------------------------------------------------
# /recommend/sonic-sibling
# ---------------------------------------------------------------------------

def test_sonic_sibling_still_returns_501_with_optional_collection():
    """Relaxed collection param must not change the 501 behaviour."""
    fixed = SimpleNamespace(id="user-E", email="e@x")
    app = create_app()
    app.dependency_overrides[rec_route.get_current_user] = lambda: fixed

    with TestClient(app) as c:
        resp = c.get("/api/v1/recommend/sonic-sibling?track_id=t1&collection=whatever")

    assert resp.status_code == 501


def test_sonic_sibling_returns_501_without_collection():
    """collection param is now optional — omitting it must still give 501."""
    fixed = SimpleNamespace(id="user-F", email="f@x")
    app = create_app()
    app.dependency_overrides[rec_route.get_current_user] = lambda: fixed

    with TestClient(app) as c:
        resp = c.get("/api/v1/recommend/sonic-sibling?track_id=t1")

    assert resp.status_code == 501
