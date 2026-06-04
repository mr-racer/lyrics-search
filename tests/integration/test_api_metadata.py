"""Integration tests for metadata API endpoints."""

from fastapi.testclient import TestClient

from app.api.main import create_app
from ._auth_helper import authenticate_test_client


class TestMetadataAPI:
    def test_get_artist_facts_empty(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get(
                "/api/v1/metadata/artists/the-weeknd/facts",
                params={"collection": "test"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

    def test_add_artist_fact(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/metadata/artists/test-artist/facts",
                json={"fact": "Test fact via API", "collection": "api_col"},
            )
            assert resp.status_code == 201
            assert resp.json() == {"ok": True}

    def test_get_artist_facts_after_add(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.post(
                "/api/v1/metadata/artists/readme-artist/facts",
                json={"fact": "README fact", "collection": "r_col"},
            )
            resp = c.get(
                "/api/v1/metadata/artists/readme-artist/facts",
                params={"collection": "r_col"},
            )
            assert resp.status_code == 200
            assert "README fact" in resp.json()

    def test_get_song_facts_empty(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get(
                "/api/v1/metadata/songs/some-song/facts",
                params={"collection": "test"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

    def test_add_song_fact(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/metadata/songs/test-song/facts",
                json={"fact": "Song trivia", "collection": "s_col"},
            )
            assert resp.status_code == 201
            assert resp.json() == {"ok": True}

    def test_get_song_facts_after_add(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
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

    def test_facts_scoped_by_collection(self):
        """Facts added to one collection should not appear in another."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.post(
                "/api/v1/metadata/artists/scoped-artist/facts",
                json={"fact": "Col A fact", "collection": "col_a"},
            )
            resp = c.get(
                "/api/v1/metadata/artists/scoped-artist/facts",
                params={"collection": "col_b"},
            )
            assert resp.status_code == 200
            assert resp.json() == []

    def test_add_fact_missing_collection(self):
        """POST without collection in body should fail validation."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/metadata/artists/artist-x/facts",
                json={"fact": "No collection"},
            )
            assert resp.status_code == 422
