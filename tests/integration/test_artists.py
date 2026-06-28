"""Integration tests for artist endpoints & artist enumeration.

Consolidates:
  - test_api_artists.py                  -> TestApiArtists
  - test_artist_aggregate_collabs.py     -> TestArtistAggregateCollabs
  - test_artists_audiodb_fields.py       -> TestArtistsAudiodbFields
  - test_distinct_artist_slugs_explode.py-> TestDistinctArtistSlugsExplode
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.api.routes import artists as art_route
from app.resources.metadata_db import MetadataDB
from app.services.artist_split import split_artists
from app.services.library_service import LibraryService
from ._auth_helper import authenticate_test_client

# Phase D-soft: server derives the collection from the JWT user, never from the
# client. Override get_current_user with a fixed user so the derived collection
# is deterministic ("acct_user-A") and seed all collection-scoped data under it.
_FIXED_USER = SimpleNamespace(id="user-A", email="a@x")
_DERIVED = "acct_user-A"


def _collab_pt(tid, artist, slugs, primary, album="Al"):
    pt = MagicMock()
    pt.id = tid
    pt.payload = {
        "title": f"T{tid}", "artist": artist, "artists": split_artists(artist),
        "artist_slugs": slugs, "primary_artist_slug": primary,
        "album": album, "year": 2008, "duration": 200.0,
        "cover_art_path": "/c.jpg", "genre": "rap",
    }
    return pt


def _single_artist_client(tmp_path, monkeypatch, pts):
    """A client whose Qdrant returns exactly `pts` (in order) for any scroll."""
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    qdrant = MagicMock()
    qdrant.scroll.return_value = (pts, None)
    db = MagicMock(); db.qdrant = qdrant
    app.state.db_client = db
    c = TestClient(app)
    authenticate_test_client(c, app)
    return c


def _distinct_pt(artist, slugs, names):
    pt = MagicMock()
    pt.payload = {"artist": artist, "artist_slugs": slugs, "artists": names}
    return pt


class TestApiArtists:
    """Integration tests for GET /artists/{slug}."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        app = create_app()
        app.dependency_overrides[art_route.get_current_user] = lambda: _FIXED_USER

        # Stub db_client with a qdrant mock that returns canned points
        class FakeQdrant:
            def __init__(self, points):
                self._points = points
            def scroll(self, collection_name, limit, offset, with_payload, with_vectors, scroll_filter=None):
                if offset is None:
                    return list(self._points), None
                return [], None
            def get_collections(self):
                m = MagicMock(); m.collections = [MagicMock(name="x")]
                return m

        class FakePoint:
            def __init__(self, id, payload):
                self.id = id
                self.payload = payload

        points = [
            FakePoint("t1", {"artist": "Dua Lipa", "title": "Physical", "album": "Future Nostalgia",
                              "year": 2020, "duration": 195.0, "file_path": "/a.flac",
                              "cover_art_path": "/covers/c1.jpg", "genre": "pop"}),
            FakePoint("t2", {"artist": "Dua Lipa", "title": "Levitating", "album": "Future Nostalgia",
                              "year": 2020, "duration": 203.0, "file_path": "/b.flac",
                              "cover_art_path": "/covers/c2.jpg"}),
            FakePoint("t3", {"artist": "Other", "title": "x", "album": "y",
                              "year": 2010, "duration": 120.0, "file_path": "/c.flac"}),
        ]
        db = MagicMock()
        db.qdrant = FakeQdrant(points)
        app.state.db_client = db
        c = TestClient(app)
        yield c
        MetadataDB._reset_for_tests()

    def test_get_artist_not_found(self, client):
        r = client.get("/api/v1/artists/no-such?collection=col_a")
        assert r.status_code == 404

    def test_get_artist_aggregates_tracks_and_albums(self, client):
        # Seed artist + facts under the DERIVED collection (server ignores ?collection).
        conn = MetadataDB.get()
        conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                      ("dua-lipa", "Dua Lipa", _DERIVED))
        conn.commit()
        MetadataDB.add_artist_facts_batch("dua-lipa", _DERIVED, ["fact1", "fact2"], source="test")
        r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
        assert r.status_code == 200
        body = r.json()
        assert body["slug"] == "dua-lipa"
        assert body["name"] == "Dua Lipa"
        assert body["track_count"] == 2  # 'Other' artist's track excluded
        assert body["album_count"] == 1
        assert body["albums"][0]["title"] == "Future Nostalgia"
        assert len(body["albums"][0]["tracks"]) == 2
        assert body["facts"] == ["fact1", "fact2"]
        assert body["bio"] is None  # not indexed

    def test_get_artist_includes_bio_when_indexed(self, client):
        conn = MetadataDB.get()
        conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                      ("dua-lipa", "Dua Lipa", _DERIVED))
        conn.commit()
        MetadataDB.set_artist_bio("dua-lipa", _DERIVED, "en", "Indie-pop, London.")
        r = client.get("/api/v1/artists/dua-lipa?collection=col_a&lang=en")
        assert r.status_code == 200
        assert r.json()["bio"] == "Indie-pop, London."

    def test_get_artist_decade_range_from_year_span(self, client):
        conn = MetadataDB.get()
        conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                      ("dua-lipa", "Dua Lipa", _DERIVED))
        conn.commit()
        r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
        body = r.json()
        # All tracks 2020 → single decade
        assert body["decade_range"] == "2020s"

    def test_track_coercion_handles_messy_payload_values(self):
        """Real Qdrant payloads sometimes carry hyphen-range strings ('154-179')
        in the duration field — must not 500 the endpoint."""
        from app.api.routes.artists import _coerce_float, _coerce_year

        # duration coercion
        assert _coerce_float(195.0) == 195.0
        assert _coerce_float(195) == 195.0
        assert _coerce_float("195") == 195.0
        assert _coerce_float("195.5") == 195.5
        assert _coerce_float("154-179") == 166.5  # average of range
        assert _coerce_float(None) == 0.0
        assert _coerce_float("") == 0.0
        assert _coerce_float("garbage") == 0.0
        assert _coerce_float("154-") == 154.0  # trailing dash → use what we have

        # year coercion
        assert _coerce_year(2020) == 2020
        assert _coerce_year("2020") == 2020
        assert _coerce_year("2018-2020") == 2018  # first valid
        assert _coerce_year(None) is None
        assert _coerce_year("") is None
        assert _coerce_year(0) is None  # zero treated as missing
        assert _coerce_year("garbage") is None

    def test_get_artist_album_liked_track_count(self, client):
        """Albums carry the number of liked tracks (drives the gold-star marker)."""
        conn = MetadataDB.get()
        conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                      ("dua-lipa", "Dua Lipa", _DERIVED))
        conn.commit()
        # Like both album tracks under the derived collection; a like on the other
        # artist's track (t3) must not bleed into Dua Lipa's count.
        MetadataDB.set_reaction("t1", _DERIVED, "like")
        MetadataDB.set_reaction("t2", _DERIVED, "like")
        MetadataDB.set_reaction("t3", _DERIVED, "like")
        r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
        assert r.status_code == 200
        album = r.json()["albums"][0]
        assert album["liked_track_count"] == 2

    def test_get_artist_liked_track_count_zero_without_reactions(self, client):
        conn = MetadataDB.get()
        conn.execute("INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
                      ("dua-lipa", "Dua Lipa", _DERIVED))
        conn.commit()
        r = client.get("/api/v1/artists/dua-lipa?collection=col_a")
        assert r.json()["albums"][0]["liked_track_count"] == 0

    def test_get_artist_ignores_supplied_collection(self, client):
        """D-soft: even when the client passes ?collection=acct_BAD, the server uses
        the JWT-derived collection (acct_user-A) when building the aggregate."""
        from unittest.mock import patch

        captured: dict = {}

        def fake_build(db, collection, slug, lang):
            captured["collection"] = collection
            from app.domain.models import ArtistAggregate
            return ArtistAggregate(slug=slug, name="X", track_count=0, album_count=0)

        with patch.object(art_route, "build_artist_aggregate", side_effect=fake_build):
            r = client.get("/api/v1/artists/some-slug?collection=acct_BAD")
        assert r.status_code == 200
        assert captured["collection"] == _DERIVED


class TestArtistAggregateCollabs:
    """Opening an artist surfaces their collaborations; non-participants excluded."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        pts = [
            _collab_pt("1", "Kanye West", ["kanye-west"], "kanye-west"),
            _collab_pt("2", "Kanye West, Sia", ["kanye-west", "sia"], "kanye-west"),
            _collab_pt("3", "Drake, Kanye West", ["drake", "kanye-west"], "drake"),
            _collab_pt("4", "Taylor Swift", ["taylor-swift"], "taylor-swift"),  # must be excluded
        ]
        qdrant = MagicMock()
        # MagicMock ignores scroll_filter and returns all points — the client-side
        # membership guard is what filters here (and on un-backfilled real data).
        qdrant.scroll.return_value = (pts, None)
        db = MagicMock(); db.qdrant = qdrant
        app.state.db_client = db
        c = TestClient(app)
        authenticate_test_client(c, app)
        yield c
        MetadataDB._reset_for_tests()
        app.state.db_client = None

    def test_collabs_included_nonparticipant_excluded(self, client):
        resp = client.get("/api/v1/artists/kanye-west", params={"collection": "c"})
        assert resp.status_code == 200
        data = resp.json()
        titles = {t["title"] for al in data["albums"] for t in al["tracks"]}
        assert {"T1", "T2", "T3"} <= titles
        assert "T4" not in titles
        assert data["track_count"] == 3

    def test_feat_role_visible_via_primary_slug(self, client):
        resp = client.get("/api/v1/artists/kanye-west", params={"collection": "c"})
        tracks = {t["title"]: t for al in resp.json()["albums"] for t in al["tracks"]}
        assert tracks["T2"]["primary_artist_slug"] == "kanye-west"
        assert tracks["T3"]["primary_artist_slug"] == "drake"

    def test_artist_name_is_canonical_not_raw_collab_tag(self, tmp_path, monkeypatch):
        # Regression: the page title must be the canonical participant, never the raw
        # collab tag of whatever track scrolls first. The FIRST point for dua-lipa is
        # a collaboration — the name must still resolve to "Dua Lipa".
        c = _single_artist_client(tmp_path, monkeypatch, [
            _collab_pt("1", "Dua Lipa x Angele", ["dua-lipa", "angele"], "dua-lipa"),
            _collab_pt("2", "Dua Lipa", ["dua-lipa"], "dua-lipa"),
        ])
        try:
            resp = c.get("/api/v1/artists/dua-lipa", params={"collection": "c"})
            assert resp.status_code == 200
            assert resp.json()["name"] == "Dua Lipa"
        finally:
            MetadataDB._reset_for_tests()
            app.state.db_client = None

    def test_featured_artist_name_taken_from_their_participant(self, tmp_path, monkeypatch):
        # The feat's page (angele) must show "Angele", not the whole "Dua Lipa x
        # Angele" tag, even though every one of their tracks is a collaboration.
        c = _single_artist_client(tmp_path, monkeypatch, [
            _collab_pt("1", "Dua Lipa x Angele", ["dua-lipa", "angele"], "dua-lipa"),
        ])
        try:
            resp = c.get("/api/v1/artists/angele", params={"collection": "c"})
            assert resp.status_code == 200
            assert resp.json()["name"] == "Angele"
        finally:
            MetadataDB._reset_for_tests()
            app.state.db_client = None


class TestArtistsAudiodbFields:
    """Integration: GET /artists/{slug} includes audiodb fields when present in DB."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
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

    def test_artists_endpoint_returns_audiodb_fields(self, client):
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


class TestDistinctArtistSlugsExplode:
    """Distinct-artist enumeration explodes collaborations into individuals."""

    def test_collab_string_replaced_by_individuals(self):
        pts = [
            _distinct_pt("Dua Lipa, Angele", ["dua-lipa", "angele"], ["Dua Lipa", "Angele"]),
            _distinct_pt("Dua Lipa", ["dua-lipa"], ["Dua Lipa"]),
        ]
        qdrant = MagicMock()
        qdrant.scroll.return_value = (pts, None)
        result = LibraryService.list_distinct_artist_slugs(
            qdrant_client=qdrant, collection_name="c",
        )
        slugs = {s for s, _ in result}
        assert "dua-lipa" in slugs
        assert "angele" in slugs
        assert "dua-lipa-angele" not in slugs

    def test_fallback_when_payload_lacks_slugs(self):
        # Un-backfilled point: only raw `artist`. Must still explode via splitter.
        pt = MagicMock()
        pt.payload = {"artist": "Calvin Harris & Dua Lipa"}
        qdrant = MagicMock()
        qdrant.scroll.return_value = ([pt], None)
        result = LibraryService.list_distinct_artist_slugs(
            qdrant_client=qdrant, collection_name="c",
        )
        slugs = {s for s, _ in result}
        assert "calvin-harris" in slugs
        assert "dua-lipa" in slugs

    def test_fallback_alias_display_name_is_artist_name_not_slug(self):
        pt = MagicMock()
        pt.payload = {"artist": "Ye"}
        qdrant = MagicMock()
        qdrant.scroll.return_value = ([pt], None)
        result = LibraryService.list_distinct_artist_slugs(
            qdrant_client=qdrant, collection_name="c",
        )
        slug_to_name = dict(result)
        assert "kanye-west" in slug_to_name
        assert slug_to_name["kanye-west"] == "Ye"
