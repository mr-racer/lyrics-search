"""Integration test: /metadata/tracks/{id}/facts prefers refined over originals.

D-soft: the endpoint derives collection from JWT; we pin a fixed user
(id="track-facts-user") so the derived collection is "acct_track-facts-user"
and seed all facts under that collection name.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import metadata as meta_route
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client

_FIXED_USER = SimpleNamespace(id="track-facts-user", email="tf@x")
_DERIVED_COLLECTION = "acct_track-facts-user"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()

    qdrant = MagicMock()
    qdrant.retrieve.return_value = [MagicMock(payload={"title": "Foo", "artist": "Bar"})]
    db_stub = MagicMock()
    db_stub.qdrant = qdrant
    app.state.db_client = db_stub

    # Override auth so the derived collection is deterministic
    app.dependency_overrides[meta_route.get_current_user] = lambda: _FIXED_USER

    # Seed original song + artist facts under the DERIVED collection.
    # get_song_facts_key("Bar", "Foo") -> "bar-foo"  (hyphens, not "::")
    conn = MetadataDB._connect()
    conn.execute(
        "INSERT INTO artists (slug, name, collection_name) VALUES ('bar', 'Bar', ?)",
        (_DERIVED_COLLECTION,),
    )
    conn.execute(
        "INSERT INTO songs (slug, title, artist_slug, collection_name) "
        "VALUES ('bar-foo', 'Foo', 'bar', ?)",
        (_DERIVED_COLLECTION,),
    )
    conn.execute(
        "INSERT INTO song_facts (song_slug, fact) VALUES (?, ?), (?, ?)",
        ("bar-foo", "Original song fact 1", "bar-foo", "Original song fact 2"),
    )
    conn.execute("INSERT INTO artist_facts (artist_slug, fact) VALUES (?, ?)",
                 ("bar", "Original artist fact"))
    conn.commit()

    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    app.dependency_overrides.clear()
    MetadataDB._reset_for_tests()


def test_facts_returns_originals_without_refined(client):
    resp = client.get(
        "/api/v1/metadata/tracks/t1/facts",
        params={"collection": "music", "lang": "en"},
    )
    body = resp.json()
    assert "Original song fact 1" in body["song_facts"]
    assert body["artist_facts"] == ["Original artist fact"]


def test_facts_returns_refined_when_present(client):
    # Refined facts are keyed by song_slug (get_song_facts_key(artist, title)),
    # not by track_id. The mocked Qdrant payload has artist="Bar", title="Foo"
    # → song_slug = "bar-foo".
    MetadataDB.set_refined_facts(
        scope="song", scope_key="bar-foo", collection_name=_DERIVED_COLLECTION, lang="en",
        refined=["Refined and sharper"],
    )
    resp = client.get(
        "/api/v1/metadata/tracks/t1/facts",
        params={"collection": "music", "lang": "en"},
    )
    body = resp.json()
    assert body["song_facts"] == ["Refined and sharper"]
    # Artist refinement absent → fall back to original
    assert body["artist_facts"] == ["Original artist fact"]


def test_facts_empty_refined_does_not_fall_back(client):
    """Explicit empty list from refined_facts must override originals."""
    MetadataDB.set_refined_facts(
        scope="song", scope_key="bar-foo", collection_name=_DERIVED_COLLECTION, lang="en",
        refined=[],
    )
    resp = client.get(
        "/api/v1/metadata/tracks/t1/facts",
        params={"collection": "music", "lang": "en"},
    )
    body = resp.json()
    assert body["song_facts"] == []  # not original
