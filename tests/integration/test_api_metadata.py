"""Integration tests for metadata API endpoints."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import metadata as meta_route
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
