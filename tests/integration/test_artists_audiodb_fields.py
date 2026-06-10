"""Integration: GET /artists/{slug} includes audiodb fields when present in DB."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import artists as art_route
from app.resources.metadata_db import MetadataDB

# Phase D-soft: the collection is derived from the JWT user, never from the
# client. Override get_current_user with a fixed user so the derived collection
# is deterministic and seed the audiodb row under it.
_FIXED_USER = SimpleNamespace(id="user-A", email="a@x")
_DERIVED = "acct_user-A"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    _app = create_app()
    _app.dependency_overrides[art_route.get_current_user] = lambda: _FIXED_USER
    # Stub a db_client so the route reaches the aggregate-building code (no 503).
    # scroll returns no points → partial aggregate; the DB-side audiodb fields
    # are what we assert.
    db = MagicMock()
    db.qdrant.scroll.return_value = ([], None)
    _app.state.db_client = db
    c = TestClient(_app, raise_server_exceptions=False)
    yield c
    MetadataDB._reset_for_tests()


def test_artists_endpoint_returns_audiodb_fields(client):
    # Pre-populate the DB rows directly, under the DERIVED collection.
    MetadataDB.upsert_artist(slug="kanye-west", name="Kanye West", collection_name=_DERIVED)
    MetadataDB.upsert_artist_audiodb(
        slug="kanye-west", collection_name=_DERIVED,
        audiodb_bio="Kanye Omari West...",
        mood="introspective",
        country_code="US",
        country="Chicago, USA",
        label="Roc-A-Fella",
        cutout_path="/covers/artists/abc.png",
        thumb_path="/covers/artists/def.png",
        audiodb_mbid="mbid-123",
    )

    # ?collection=test is supplied but IGNORED — the server derives acct_user-A,
    # which is where we seeded the audiodb row.
    resp = client.get("/api/v1/artists/kanye-west?collection=test")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("mood") == "introspective"
    assert body.get("country_code") == "US"
    assert body.get("country") == "Chicago, USA"
    assert body.get("label") == "Roc-A-Fella"
    assert body.get("cutout_path") == "/covers/artists/abc.png"
    assert body.get("thumb_path") == "/covers/artists/def.png"
    assert body.get("audiodb_mbid") == "mbid-123"
