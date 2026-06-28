"""Integration tests for metadata API endpoints."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.api.routes import metadata as meta_route
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client

# Fixed user whose derived collection is always "acct_user-meta-test"
_FIXED_USER = SimpleNamespace(id="user-meta-test", email="meta@x")
_DERIVED = "acct_user-meta-test"


def _make_app_with_fixed_user():
    """Create a fresh app with get_current_user overridden to _FIXED_USER."""
    app = create_app()
    app.dependency_overrides[meta_route.get_current_user] = lambda: _FIXED_USER
    return app


class TestMetadataAPI:
    def test_get_artist_facts_empty(self):
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/metadata/artists/the-weeknd/facts",
                params={"collection": "test"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

    def test_add_artist_fact(self):
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            # collection field is now optional — omit it to verify D-soft relaxation
            resp = c.post(
                "/api/v1/metadata/artists/test-artist/facts",
                json={"fact": "Test fact via API"},
            )
            assert resp.status_code == 201
            assert resp.json() == {"ok": True}

    def test_get_artist_facts_after_add(self):
        """Fact written by POST is visible via GET because both derive the same
        collection from the JWT — supplied collection param is ignored."""
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            # POST with a stale collection param — should be ignored
            c.post(
                "/api/v1/metadata/artists/readme-artist/facts",
                json={"fact": "README fact", "collection": "old_col"},
            )
            # GET with a different stale collection param — also ignored
            resp = c.get(
                "/api/v1/metadata/artists/readme-artist/facts",
                params={"collection": "other_old_col"},
            )
            assert resp.status_code == 200
            assert "README fact" in resp.json()

    def test_get_song_facts_empty(self):
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/metadata/songs/some-song/facts",
                params={"collection": "test"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

    def test_add_song_fact(self):
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/metadata/songs/test-song/facts",
                json={"fact": "Song trivia"},
            )
            assert resp.status_code == 201
            assert resp.json() == {"ok": True}

    def test_get_song_facts_after_add(self):
        """Fact written by POST is visible via GET — both use the derived collection."""
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            c.post(
                "/api/v1/metadata/songs/readme-song/facts",
                json={"fact": "Written in 2020", "collection": "rs_col"},
            )
            resp = c.get(
                "/api/v1/metadata/songs/readme-song/facts",
                params={"collection": "rs_col"},
            )
            assert resp.status_code == 200
            assert "Written in 2020" in resp.json()

    def test_facts_scoped_by_user(self):
        """Facts added by user-A are NOT visible to user-B (different derived collections)."""
        from app.resources.metadata_db import MetadataDB

        user_a = SimpleNamespace(id="user-scope-A", email="a@x")
        user_b = SimpleNamespace(id="user-scope-B", email="b@x")

        app = create_app()

        # Seed a fact as user-A
        app.dependency_overrides[meta_route.get_current_user] = lambda: user_a
        with TestClient(app) as c:
            c.post(
                "/api/v1/metadata/artists/scoped-artist/facts",
                json={"fact": "Col A fact"},
            )
        app.dependency_overrides.clear()

        # Read as user-B — should NOT see user-A's fact
        app.dependency_overrides[meta_route.get_current_user] = lambda: user_b
        with TestClient(app) as c:
            resp = c.get("/api/v1/metadata/artists/scoped-artist/facts")
            assert resp.status_code == 200
            assert resp.json() == []
        app.dependency_overrides.clear()

    def test_add_fact_collection_optional(self):
        """POST without collection in body succeeds — collection is now optional (D-soft)."""
        app = _make_app_with_fixed_user()
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/metadata/artists/artist-x/facts",
                json={"fact": "No collection"},
            )
            assert resp.status_code == 201

    def test_metadata_endpoints_ignore_supplied_collection(self):
        """D-soft: supplied collection param is ignored; server derives from JWT."""
        app = create_app()
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[meta_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            for path in [
                "/api/v1/metadata/artists/some-slug/facts?collection=acct_BAD",
                "/api/v1/metadata/songs/some-key/facts?collection=acct_BAD",
                "/api/v1/metadata/random-facts?collection=acct_BAD",
            ]:
                resp = c.get(path)
                assert resp.status_code < 500, path
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /metadata/tracks/{id}/facts prefers refined over originals.
#
# D-soft: the endpoint derives collection from JWT; we pin a fixed user
# (id="track-facts-user") so the derived collection is "acct_track-facts-user"
# and seed all facts under that collection name.
# (merged from test_api_metadata_facts_with_refined.py)
# ---------------------------------------------------------------------------

_TRACK_FACTS_USER = SimpleNamespace(id="track-facts-user", email="tf@x")
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
    app.dependency_overrides[meta_route.get_current_user] = lambda: _TRACK_FACTS_USER

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


class TestFactsWithRefined:
    def test_facts_returns_originals_without_refined(self, client):
        resp = client.get(
            "/api/v1/metadata/tracks/t1/facts",
            params={"collection": "music", "lang": "en"},
        )
        body = resp.json()
        assert "Original song fact 1" in body["song_facts"]
        assert body["artist_facts"] == ["Original artist fact"]

    def test_facts_returns_refined_when_present(self, client):
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

    def test_facts_empty_refined_does_not_fall_back(self, client):
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


# ---------------------------------------------------------------------------
# /metadata/tracks (batch) + /metadata/tracks/{id} — the player's queue-enrich
# endpoints. They MUST return per-participant artist_refs so a collaboration
# links each artist to their own page; a previous version omitted them, so the
# frontend fell back to slugifying the whole raw tag ("Eminem, Nate Dogg" ->
# "eminem,-nate-dogg") and the second artist 404'd.
# ---------------------------------------------------------------------------
class TestTrackMetadataArtistRefs:
    def _app_with_track(self, payload: dict):
        app = _make_app_with_fixed_user()
        pt = SimpleNamespace(id="t1", payload=payload)
        fake_qdrant = MagicMock()
        fake_qdrant.retrieve.return_value = [pt]
        app.state.db_client = MagicMock(qdrant=fake_qdrant)
        return app

    def test_batch_returns_split_artist_refs(self):
        app = self._app_with_track({
            "title": "Till I Collapse",
            "artist": "Eminem, Nate Dogg",
            "album": "The Eminem Show",
            "year": 2002,
        })
        c = TestClient(app)
        resp = c.get("/api/v1/metadata/tracks", params={"ids": "t1"})
        assert resp.status_code == 200
        track = resp.json()[0]
        assert [r["slug"] for r in track["artist_refs"]] == ["eminem", "nate-dogg"]
        assert track["primary_artist_slug"] == "eminem"
        # album/year must survive so the player can show them
        assert track["album"] == "The Eminem Show"
        assert track["year"] == 2002

    def test_single_returns_split_artist_refs(self):
        app = self._app_with_track({
            "title": "Till I Collapse",
            "artist": "Eminem, Nate Dogg",
        })
        c = TestClient(app)
        resp = c.get("/api/v1/metadata/tracks/t1")
        assert resp.status_code == 200
        track = resp.json()
        assert [r["slug"] for r in track["artist_refs"]] == ["eminem", "nate-dogg"]
        assert track["primary_artist_slug"] == "eminem"
