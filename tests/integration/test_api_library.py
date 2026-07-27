"""Integration tests for library API endpoints.

Consolidated from the former ``test_api_library_*.py`` modules. Each former module
is represented as a ``Test*`` class below:

- albums            → ``TestAlbums``
- delete_track      → ``TestDeleteTrack``
- featured_artist   → ``TestFeaturedArtist``
- liked_songs       → ``TestLikedSongs``
- listening_stats   → ``TestListeningStats``
- mode_gates        → ``TestIndexModeGate`` / ``TestSharingIndexSanityCheck`` /
                       ``TestModeGateMatrix``
- rediscover        → ``TestRediscover``
- sonic_facets      → ``TestSonicFacets``
- year_facets       → ``TestYearFacets``
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.api.dependencies import get_current_user
from app.api.routes import library as lib_route
from app.domain.models import User
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


class TestLibraryCollections:
    def test_collections_returns_no_data_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/collections")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["collections"] == []


class TestLibraryStats:
    def test_stats_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["total_tracks"] == 0


class TestLibraryTopPairs:
    def test_top_pairs_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/top-pairs")
            assert resp.status_code == 200
            data = resp.json()
            assert data["qdrant_available"] is False
            assert data["similar"] == []

    def test_top_pairs_for_track_unavailable_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/top-pairs/some-track-id")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is False
            assert data["similar"] == []
            assert data["dissimilar"] == []


class TestLibraryBrowse:
    def test_browse_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse", params={"q": "test"})
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_omits_query_returns_empty_when_qdrant_down(self):
        """When q is omitted and Qdrant is unavailable, return 200 with empty list."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_browse_rejects_short_query(self):
        """q with less than 2 characters should be rejected (min_length=2)."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/browse", params={"q": "a"})
            assert resp.status_code == 422


class TestLibraryIndex:
    def test_index_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": "/music", "collection_name": "test"},
            )
            assert resp.status_code == 503


class TestLibraryStatus:
    def test_status_503_when_service_unavailable(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/status")
            assert resp.status_code == 503


class TestLibraryDeleteCollection:
    def test_delete_collection_is_gone_410(self):
        """Phase D removed self-serve collection delete — the old path now 410s
        and points callers at the admin wipe endpoint."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.delete("/api/v1/library/collection/test")
            assert resp.status_code == 410
            assert "wipe" in resp.json()["detail"]


# ── Phase D (D-soft): collection derived from JWT, supplied param ignored ──────

def test_library_endpoints_ignore_supplied_collection():
    """Every collection-aware GET tolerates a bogus supplied collection_name:
    the server derives the collection from the JWT and never 5xxs on it."""
    from app.api.routes import library as lib_route
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        for path in [
            "/api/v1/library/browse?collection_name=acct_BAD",
            "/api/v1/library/random?collection_name=acct_BAD",
            "/api/v1/library/albums?collection_name=acct_BAD",
            "/api/v1/library/liked-songs?collection_name=acct_BAD",
            "/api/v1/library/rediscover?collection_name=acct_BAD",
            "/api/v1/library/listening-stats?collection_name=acct_BAD",
            "/api/v1/library/stats?collection_name=acct_BAD",
            "/api/v1/library/top-pairs?collection_name=acct_BAD",
        ]:
            resp = c.get(path)
            assert resp.status_code < 500, path
    app.dependency_overrides.clear()


def test_library_echoes_derived_collection_not_supplied():
    """The response echo reflects the JWT-derived collection, not the param."""
    from app.api.routes import library as lib_route
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.state.db_client = None  # force the graceful-empty echo branch
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    with TestClient(app) as c:
        resp = c.get("/api/v1/library/liked-songs?collection_name=acct_BAD")
        assert resp.status_code == 200
        assert resp.json()["collection_name"] == "acct_user-A"
    app.dependency_overrides.clear()


class TestLibrarySettingsRename:
    def test_settings_uses_derived_collection(self):
        from app.api.routes import library as lib_route
        from types import SimpleNamespace
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/settings")
            assert resp.status_code == 200
            body = resp.json()
            assert body["collection_name"] == "acct_user-A"
            assert body["ai_enabled"] is True  # no row → AI-on default
        app.dependency_overrides.clear()

    def test_old_settings_path_is_410(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/collections/whatever/settings")
            assert resp.status_code == 410
            assert "settings" in resp.json()["detail"].lower()

    def test_ai_enabled_uses_derived_collection(self):
        from app.api.routes import library as lib_route
        from types import SimpleNamespace
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            resp = c.patch("/api/v1/library/ai-enabled", json={"enabled": False})
            assert resp.status_code == 200
            assert resp.json()["collection_name"] == "acct_user-A"
            assert resp.json()["ai_enabled"] is False
        app.dependency_overrides.clear()

    def test_old_ai_enabled_path_is_410(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.patch(
                "/api/v1/library/collections/whatever/ai-enabled",
                json={"enabled": False},
            )
            assert resp.status_code == 410


# ── GET /library/albums (from test_api_library_albums.py) ─────────────────────

class TestAlbums:
    def test_albums_returns_empty_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/albums", params={"collection_name": "music_explorer"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["albums"] == []
            assert body["qdrant_available"] is False

    def test_albums_accepts_sort_param(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            for s in ("alphabetical", "year_desc", "year_asc", "track_count_desc"):
                resp = c.get("/api/v1/library/albums",
                             params={"collection_name": "x", "sort": s})
                assert resp.status_code == 200


# ── Producer credits + album label filter (SQLite-only, Qdrant not needed) ────

def _login_credits(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


def _seed_credit_tracks(collection: str) -> None:
    MetadataDB.upsert_track_metadata(collection, "t1", {
        "title": "Runaway", "artist": "Kanye West", "album": "MBDTF",
        "artist_slugs": ["kanye-west"], "producer": "Kanye West, Emile",
        "label": "GOOD Music, Def Jam", "duration": 200,
    })
    MetadataDB.upsert_track_metadata(collection, "t2", {
        "title": "Power", "artist": "Kanye West", "album": "MBDTF",
        "artist_slugs": ["kanye-west"], "producer": "kanye west",
        "label": "def jam", "duration": 200,
    })
    MetadataDB.upsert_track_metadata(collection, "t3", {
        "title": "Accordion", "artist": "Madvillain", "album": "Madvillainy",
        "artist_slugs": ["madvillain"], "producer": "Madlib",
        "label": "Stones Throw", "duration": 120,
    })


class TestProducerCredits:
    def test_resolve_returns_artist_slug_and_counts(self):
        app = create_app()
        _login_credits(app, "alice")
        _seed_credit_tracks("acct_alice")
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/producers/resolve", params={"track_id": "t1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["collection_name"] == "acct_alice"
            got = {p["name"]: p for p in body["producers"]}
            assert set(got) == {"Kanye West", "Emile"}
            assert got["Kanye West"]["artist_slug"] == "kanye-west"
            assert got["Kanye West"]["produced_count"] == 1   # t2 only, t1 excluded
            assert got["Emile"]["artist_slug"] is None
            assert got["Emile"]["produced_count"] == 0

    def test_resolve_unknown_track_yields_empty(self):
        app = create_app()
        _login_credits(app, "alice")
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/producers/resolve", params={"track_id": "nope"})
            assert resp.status_code == 200
            assert resp.json()["producers"] == []

    def test_producer_tracks_matches_case_insensitively(self):
        app = create_app()
        _login_credits(app, "alice")
        _seed_credit_tracks("acct_alice")
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/producers/tracks", params={"name": "KANYE WEST"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["producer"] == "KANYE WEST"
            assert [t["track_id"] for t in body["tracks"]] == ["t2", "t1"]

    def test_producer_tracks_requires_name(self):
        app = create_app()
        _login_credits(app, "alice")
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/producers/tracks")
            assert resp.status_code == 422

    def test_no_cross_account_leak(self):
        """Bob's resolve must not see Alice's produced tracks."""
        app = create_app()
        _login_credits(app, "bob")
        _seed_credit_tracks("acct_alice")
        MetadataDB.upsert_track_metadata("acct_bob", "b1", {
            "title": "Solo", "artist": "Bob", "producer": "Kanye West",
        })
        with TestClient(app) as c:
            resp = c.get("/api/v1/library/producers/resolve", params={"track_id": "b1"})
            got = {p["name"]: p for p in resp.json()["producers"]}
            assert got["Kanye West"]["produced_count"] == 0
            assert got["Kanye West"]["artist_slug"] is None


class TestSampleResolve:
    def test_resolve_matches_and_keeps_alignment(self):
        app = create_app()
        _login_credits(app, "alice")
        _seed_credit_tracks("acct_alice")
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/samples/resolve", json={
                "items": [
                    "Kanye West — Runaway",       # in the library → match
                    "James Brown — Funky Drummer",  # not here → null
                ],
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["collection_name"] == "acct_alice"
            assert body["matches"][0]["track_id"] == "t1"
            assert body["matches"][0]["cover_art_path"] is None
            assert body["matches"][1] is None

    def test_no_cross_account_leak(self):
        app = create_app()
        _login_credits(app, "bob")
        _seed_credit_tracks("acct_alice")
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/samples/resolve",
                          json={"items": ["Kanye West — Runaway"]})
            assert resp.status_code == 200
            assert resp.json()["matches"] == [None]

    def test_empty_items(self):
        app = create_app()
        _login_credits(app, "alice")
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/samples/resolve", json={"items": []})
            assert resp.status_code == 200
            assert resp.json()["matches"] == []


class TestAlbumLabelFilter:
    def test_albums_carry_labels_and_label_param_filters(self):
        app = create_app()
        _login_credits(app, "alice")
        _seed_credit_tracks("acct_alice")
        with TestClient(app) as c:
            # The route needs a non-None db_client; the SQLite fast-path never
            # touches Qdrant once track_metadata rows exist.
            c.app.state.db_client = SimpleNamespace(qdrant=MagicMock())

            resp = c.get("/api/v1/library/albums")
            assert resp.status_code == 200
            by_title = {a["album_title"]: a for a in resp.json()["albums"]}
            assert by_title["MBDTF"]["labels"] == ["Def Jam", "GOOD Music"]
            assert by_title["Madvillainy"]["labels"] == ["Stones Throw"]

            resp = c.get("/api/v1/library/albums", params={"label": "def jam"})
            assert resp.status_code == 200
            assert [a["album_title"] for a in resp.json()["albums"]] == ["MBDTF"]

            resp = c.get("/api/v1/library/albums", params={"label": "Nonexistent"})
            assert resp.json()["albums"] == []


# ── DELETE /library/tracks/{track_id} (from test_api_library_delete_track.py) ──

@pytest.fixture
def _server_mode(clean_metadata_db, tmp_path, monkeypatch):
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    media_root = tmp_path / "media"
    monkeypatch.setattr("app.services.uploads_service.media_root_default", lambda: media_root)
    monkeypatch.setattr(
        "app.services.uploads_service.quarantine_root_default", lambda: media_root / "_quarantine",
    )
    return media_root


def _login_delete(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


class TestDeleteTrack:
    def test_deletes_qdrant_point_and_file(self, _server_mode):
        media_file = _server_mode / "acct_alice" / "audio" / "deadbeef.flac"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"fake audio")

        fake_qdrant = MagicMock()
        fake_qdrant.retrieve.return_value = [MagicMock(
            payload={"file_path": str(media_file), "title": "T", "artist": "A"},
        )]
        fake_db = MagicMock()
        fake_db.qdrant = fake_qdrant

        app = create_app()
        _login_delete(app, "acct_alice")
        with TestClient(app) as c:
            app.state.db_client = fake_db
            resp = c.delete("/api/v1/library/tracks/track_xyz")
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"ok": True, "track_id": "track_xyz"}

        fake_qdrant.delete.assert_called_once()
        assert not media_file.exists()

    def test_cross_account_blocked(self, _server_mode):
        # Track belongs to Bob (file path under acct_bob); Alice must not delete it.
        media_file = _server_mode / "acct_bob" / "audio" / "deadbeef.flac"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"x")

        fake_qdrant = MagicMock()
        fake_qdrant.retrieve.return_value = [MagicMock(payload={"file_path": str(media_file)})]
        fake_db = MagicMock()
        fake_db.qdrant = fake_qdrant

        app = create_app()
        _login_delete(app, "acct_alice")
        with TestClient(app) as c:
            app.state.db_client = fake_db
            resp = c.delete("/api/v1/library/tracks/track_xyz")
            assert resp.status_code == 404
        assert media_file.exists()
        fake_qdrant.delete.assert_not_called()


# ── GET /library/featured-artist (from test_api_library_featured_artist.py) ────

def _pt_featured(tid, artist, album="Al", year=2000, cover="/c.jpg"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {"title": f"T{tid}", "artist": artist, "album": album,
                  "year": year, "duration": 200.0, "cover_art_path": cover, "genre": "rock"}
    return pt


@pytest.fixture
def featured_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    pts = [_pt_featured("1", "Radiohead"), _pt_featured("2", "Radiohead"), _pt_featured("3", "Portishead")]
    qdrant = MagicMock()
    qdrant.scroll.return_value = (pts, None)
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None


class TestFeaturedArtist:
    def test_featured_artist_deterministic_per_date(self, featured_client):
        a = featured_client.get("/api/v1/library/featured-artist",
                                params={"collection_name": "c", "date": "2026-05-24"})
        b = featured_client.get("/api/v1/library/featured-artist",
                                params={"collection_name": "c", "date": "2026-05-24"})
        assert a.status_code == 200 and b.status_code == 200
        assert a.json()["slug"] == b.json()["slug"]
        assert a.json()["name"] in {"Radiohead", "Portishead"}

    def test_featured_artist_qdrant_down_returns_404(self, featured_client):
        app.state.db_client = None
        resp = featured_client.get("/api/v1/library/featured-artist", params={"collection_name": "c"})
        assert resp.status_code in (404, 503)


# ── GET /library/liked-songs (from test_api_library_liked_songs.py) ───────────

class TestLikedSongs:
    def test_liked_songs_returns_empty_when_qdrant_down(self):
        fixed = SimpleNamespace(id="user-A", email="a@x")
        app = create_app()
        app.state.db_client = None  # force qdrant-down branch
        app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
        with TestClient(app) as c:
            # Supplied collection_name is intentionally bogus — it must be ignored.
            resp = c.get("/api/v1/library/liked-songs", params={"collection_name": "x"})
            assert resp.status_code == 200
            assert resp.json() == {"tracks": [], "collection_name": "acct_user-A"}
        app.dependency_overrides.clear()


# ── GET /library/listening-stats (from test_api_library_listening_stats.py) ───

class TestListeningStats:
    def test_listening_stats_empty_response_when_qdrant_down(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/listening-stats",
                         params={"collection_name": "x", "lang": "en"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["total_seconds_listened"] == 0
            assert body["top_track"] is None
            assert body["top_artist"] is None
            assert body["peak_hour"] is None

    def test_listening_stats_rejects_unknown_lang(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/library/listening-stats",
                         params={"collection_name": "x", "lang": "fr"})
            assert resp.status_code == 422


# ── Mode-gate matrix (from test_api_library_mode_gates.py) ────────────────────

def _setup_mode(mode: str) -> None:
    MetadataDB.set_instance_config(mode=mode, created_at=1700000000.0)


def _login_mode(app, uid: str = "acct_alice", role: str = "member") -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role=role, created_at=0.0,
    )


class TestIndexModeGate:
    def test_index_allowed_in_sharing_mode(self, tmp_path):
        _setup_mode("sharing")
        app = create_app()
        _login_mode(app)
        with TestClient(app) as c:
            # Real folder so the folder-exists sanity check passes; we only assert
            # the mode gate let us through (200 started or 503 service-down).
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code in (200, 503), resp.text

    def test_index_member_403_in_server_mode(self, tmp_path):
        """Members must not point the indexer at arbitrary host folders —
        they upload via POST /library/upload instead."""
        _setup_mode("server")
        app = create_app()
        _login_mode(app, role="member")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code == 403

    def test_index_owner_allowed_in_server_mode(self, tmp_path):
        """The owner runs the instance host, so folder indexing (wizard +
        settings) must work in server mode too — not only via uploads."""
        _setup_mode("server")
        app = create_app()
        _login_mode(app, uid="acct_owner", role="owner")
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": str(tmp_path), "collection_name": "test"},
            )
            assert resp.status_code in (200, 503), resp.text

    def test_upload_404s_in_sharing_mode(self):
        _setup_mode("sharing")
        app = create_app()
        _login_mode(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload",
                files={"file": ("song.flac", b"fake", "audio/flac")},
            )
            assert resp.status_code == 404

    def test_batch_commit_404s_in_sharing_mode(self):
        _setup_mode("sharing")
        app = create_app()
        _login_mode(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/upload/batch-commit",
                json={"upload_ids": ["x"]},
            )
            assert resp.status_code == 404


class TestSharingIndexSanityCheck:
    def test_missing_folder_returns_400(self):
        _setup_mode("sharing")
        app = create_app()
        _login_mode(app)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/library/index",
                json={"folder_path": "/nonexistent/path/xyz", "collection_name": "x"},
            )
            assert resp.status_code == 400
            assert "does not exist" in resp.json()["detail"].lower()


class TestModeGateMatrix:
    """Pin the surface: each endpoint reachable in exactly one mode."""

    SERVER_ONLY = [
        ("POST", "/api/v1/library/upload", "file"),
        ("GET", "/api/v1/library/upload/some-id", None),
        ("POST", "/api/v1/library/upload/batch-commit", {"upload_ids": ["x"]}),
        ("DELETE", "/api/v1/library/tracks/some-track", None),
    ]
    # NOTE: /library/index is no longer sharing-only — in server mode it is
    # owner-only (see TestIndexModeGate) so the wizard can index a host folder.

    @pytest.mark.parametrize("method,path,body", SERVER_ONLY)
    def test_server_only_404s_in_sharing(self, method, path, body):
        _setup_mode("sharing")
        app = create_app()
        _login_mode(app)  # bypass auth so the 404 is unambiguously from the mode gate
        with TestClient(app) as c:
            kwargs = {}
            if body == "file":
                kwargs["files"] = {"file": ("x.flac", b"x", "audio/flac")}
            elif isinstance(body, dict):
                kwargs["json"] = body
            resp = c.request(method, path, **kwargs)
            assert resp.status_code == 404, f"{method} {path}: {resp.status_code} {resp.text}"


# ── GET /library/rediscover (from test_api_library_rediscover.py) ─────────────

_DERIVED = "acct_user-A"


def _pt_redisc(tid, title="T", artist="A"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {"title": title, "artist": artist, "album": "Al",
                  "year": 1999, "duration": 200.0, "cover_art_path": f"/c/{tid}.jpg",
                  "genre": "rock"}
    return pt


@pytest.fixture
def rediscover_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([_pt_redisc("t1"), _pt_redisc("t2"), _pt_redisc("t3")], None)
    qdrant.retrieve.side_effect = lambda **kw: [_pt_redisc(kw["ids"][0])]
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    fixed = SimpleNamespace(id="user-A", email="a@x")
    app.dependency_overrides[lib_route.get_current_user] = lambda: fixed
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()
    MetadataDB._reset_for_tests()
    app.state.db_client = None


class TestRediscover:
    def test_rediscover_prefers_never_played(self, rediscover_client):
        # Record the play under the DERIVED collection so recency lines up with the
        # collection the endpoint actually queries (D-soft ignores the param).
        MetadataDB.record_playback_event(session_id="s", collection_name=_DERIVED,
                                         track_id="t1", played_sec=150.0, total_dur=200.0)
        # Supplied collection_name is bogus on purpose — it must be ignored.
        resp = rediscover_client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["never_played"] is True
        assert body["track"]["track_id"] in {"t2", "t3"}

    def test_rediscover_qdrant_down_returns_empty(self, rediscover_client):
        app.state.db_client = None
        resp = rediscover_client.get("/api/v1/library/rediscover", params={"collection_name": "c"})
        assert resp.status_code == 200
        assert resp.json()["track"] is None

    def test_rediscover_missing_collection_now_optional(self, rediscover_client):
        """D-soft relaxed the previously-required collection_name param: omitting it
        no longer 422s — the collection is derived from the JWT."""
        resp = rediscover_client.get("/api/v1/library/rediscover")
        assert resp.status_code == 200


# ── GET /library/sonic-facets (from test_api_library_sonic_facets.py) ─────────

@pytest.fixture
def sonic_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    c = TestClient(app)
    authenticate_test_client(c, app)
    yield c
    MetadataDB._reset_for_tests()


class TestSonicFacets:
    def test_sonic_facets_endpoint_returns_shape_for_empty_db(self, sonic_client):
        r = sonic_client.get("/api/v1/library/sonic-facets")
        assert r.status_code == 200
        body = r.json()
        assert body == {"tags": []}

    def test_sonic_facets_endpoint_reflects_db_state(self, sonic_client):
        MetadataDB.upsert_artist("a", "A", "test_col")
        MetadataDB.upsert_song("a-1", "T1", "a", "test_col")
        MetadataDB.upsert_sonic_descriptor(
            song_slug="a-1",
            tags=[{"tag": "melancholic", "score": 0.8}],
        )
        r = sonic_client.get("/api/v1/library/sonic-facets")
        assert r.status_code == 200
        body = r.json()
        assert body["tags"] == [{"value": "melancholic", "count": 1}]

    def test_sonic_facets_endpoint_respects_top_k_param(self, sonic_client):
        MetadataDB.upsert_artist("a", "A", "test_col")
        for i in range(20):
            slug = f"a-{i}"
            MetadataDB.upsert_song(slug, f"T{i}", "a", "test_col")
            MetadataDB.upsert_sonic_descriptor(
                song_slug=slug,
                tags=[{"tag": f"tag-{i}", "score": 0.5}],
            )
        r = sonic_client.get("/api/v1/library/sonic-facets?top_k=5")
        assert r.status_code == 200
        assert len(r.json()["tags"]) == 5


# ── GET /library/year-facets (from test_api_library_year_facets.py) ───────────
# The endpoint reads year_range from Qdrant payload (not SQLite), so tests mock
# db_client.qdrant following the pattern in test_backfill_sonic_payload.py.

def _col(name: str) -> MagicMock:
    obj = MagicMock()
    obj.name = name
    return obj


def _point(yr: str | None) -> MagicMock:
    p = MagicMock()
    p.payload = {"year_range": yr} if yr else {}
    return p


@pytest.fixture
def year_client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    app = create_app()
    tc = TestClient(app)
    authenticate_test_client(tc, app)
    yield app, tc
    MetadataDB._reset_for_tests()


def _attach_qdrant(app, collections: list[str], pages: dict[str, list]):
    qdrant = MagicMock()
    qdrant.get_collections.return_value.collections = [_col(n) for n in collections]

    def scroll_side(collection_name, limit, with_payload, with_vectors, offset=None):
        return (pages.get(collection_name, []), None)

    qdrant.scroll.side_effect = scroll_side
    db_client = MagicMock()
    db_client.qdrant = qdrant
    app.state.db_client = db_client


class TestYearFacets:
    def test_year_facets_endpoint_returns_shape_for_empty_db(self, year_client):
        app, tc = year_client
        _attach_qdrant(app, ["test_col"], {"test_col": []})
        r = tc.get("/api/v1/library/year-facets")
        assert r.status_code == 200
        assert r.json() == {"year_ranges": []}

    def test_year_facets_endpoint_reflects_qdrant_state(self, year_client):
        app, tc = year_client
        _attach_qdrant(
            app,
            ["test_col"],
            {"test_col": [_point("1990-1994"), _point("1990-1994"), _point("2000-2004")]},
        )
        r = tc.get("/api/v1/library/year-facets")
        assert r.status_code == 200
        data = r.json()
        assert data["year_ranges"][0] == {"value": "1990-1994", "count": 2}
        assert data["year_ranges"][1] == {"value": "2000-2004", "count": 1}

    def test_year_facets_endpoint_respects_top_k_param(self, year_client):
        app, tc = year_client
        pts = [_point(f"{1960 + i}-{1964 + i}") for i in range(20)]
        _attach_qdrant(app, ["test_col"], {"test_col": pts})
        r = tc.get("/api/v1/library/year-facets?top_k=5")
        assert r.status_code == 200
        assert len(r.json()["year_ranges"]) == 5

    def test_year_facets_endpoint_filters_by_collection_name(self, year_client):
        app, tc = year_client
        _attach_qdrant(
            app,
            ["col_a", "col_b"],
            {
                "col_a": [_point("1970-1974")],
                "col_b": [_point("2010-2014")],
            },
        )
        r = tc.get("/api/v1/library/year-facets?collection_name=col_b")
        assert r.status_code == 200
        yr = {item["value"]: item["count"] for item in r.json()["year_ranges"]}
        assert "1970-1974" not in yr
        assert yr["2010-2014"] == 1

    def test_year_facets_endpoint_returns_empty_without_db_client(self, year_client):
        app, tc = year_client
        app.state.db_client = None
        r = tc.get("/api/v1/library/year-facets")
        assert r.status_code == 200
        assert r.json() == {"year_ranges": []}
