"""Consolidated integration tests for the search subsystem.

Merged from (now removed):
  - test_api_search.py            -> TestSearchAPI, TestPhaseDSoft
  - test_search_filters_decade.py -> TestFiltersDecade
  - test_search_filters_sonic.py  -> TestFiltersSonic
  - test_search_points_to_hits.py -> TestPointsToHits
  - test_catalog_route.py         -> TestCatalogRoute
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import models
from qdrant_client.models import ScoredPoint

from app.api.main import create_app
from app.domain.models import SearchFilters, TrackHit
from app.resources.qdrant_utils import invalidate_light_cache
from app.services import catalog_search_service
from app.services.search_service import SearchService
from ._auth_helper import authenticate_test_client

pytestmark = pytest.mark.integration


# ── shared helpers ──────────────────────────────────────────────────────────

def _build(filters: SearchFilters):
    """Build a Qdrant filter from SearchFilters without running __init__."""
    svc = SearchService.__new__(SearchService)  # bypass __init__
    return svc._build_qdrant_filter_models(filters)


# ── from test_api_search.py ─────────────────────────────────────────────────

class TestSearchAPI:
    def test_search_503_when_service_unavailable(self):
        """POST /api/v1/search/ returns 503 when the search service is down.

        Force search_service=None so this is deterministic regardless of whether
        Qdrant happens to be reachable in the dev env (the lifespan would
        otherwise wire a real SearchService when Qdrant is up)."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            app.state.search_service = None
            resp = c.post(
                "/api/v1/search/",
                json={"query": "test", "mode": "text"},
            )
            assert resp.status_code == 503

    def test_search_returns_results_when_service_available(self):
        """POST /api/v1/search/ returns hits when SearchService is mocked."""
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={
                    "query": "hello",
                    "mode": "text",
                    "limit": 5,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "hits" in data
            assert data["query"] == "hello"
            assert data["mode"] == "text"

    def test_search_with_audio_mode(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "beat", "mode": "audio"},
            )
            assert resp.status_code == 200
            assert resp.json()["mode"] == "audio"

    def test_search_with_hybrid_mode(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "vibe", "mode": "hybrid"},
            )
            assert resp.status_code == 200
            assert resp.json()["mode"] == "hybrid"

    def test_search_with_filters(self):
        app = create_app()
        mock_service = MagicMock(spec=SearchService)

        async def mock_search(*_args, **_kwargs):
            return []

        mock_service.search = mock_search

        with TestClient(app) as c:
            authenticate_test_client(c, app)
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={
                    "query": "pop",
                    "mode": "text",
                    "filters": {"genre": "Pop", "year_ranges": ["2000-2009"]},
                },
            )
            assert resp.status_code == 200

    def test_search_request_validation_missing_query(self):
        """Empty query should fail validation."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.post(
                "/api/v1/search/",
                json={"mode": "text"},
            )
            assert resp.status_code == 422

    def test_search_models_text_endpoint(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/search/models/text")
            assert resp.status_code == 200

    def test_search_models_loaded_endpoint(self):
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            resp = c.get("/api/v1/search/models/loaded")
            assert resp.status_code == 200
            data = resp.json()
            assert "text_models" in data
            assert "clap_available" in data

    def test_search_models_loaded_reports_the_retrieval_legs(self):
        """``retrieval_status()`` documents itself as "surfaced by
        GET /search/models/loaded" — and for a while it simply was not, so a
        stack that had lost its sparse leg answered this endpoint as healthy.
        The counters matter as much as the booleans: a leg whose weights are
        resident but whose every encode dies is invisible without them."""
        app = create_app()
        with TestClient(app) as c:
            authenticate_test_client(c, app)
            data = c.get("/api/v1/search/models/loaded").json()
            assert "retrieval" in data
            legs = data["retrieval"]
            assert {"dense", "sparse", "cross_encoder", "failed",
                    "encode_failures", "sparse_oom_retries"} <= set(legs)
            assert {"sparse", "cross_encoder"} <= set(legs["encode_failures"])


# ── Phase D-soft tests ────────────────────────────────────────────────────────

class TestPhaseDSoft:
    """D-soft: endpoints derive collection from JWT, ignore client-supplied value."""

    def test_search_ignores_client_supplied_collection_name(self):
        """D-soft: even when client passes collection_name in body, server uses
        the JWT-derived value."""
        from unittest.mock import MagicMock
        from app.api.main import create_app
        from app.services.search_service import SearchService
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        mock_service = MagicMock(spec=SearchService)
        received = {}

        async def mock_search(**kwargs):
            received.update(kwargs)
            return []

        mock_service.search = mock_search
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        with TestClient(app) as c:
            c.app.state.search_service = mock_service
            resp = c.post(
                "/api/v1/search/",
                json={"query": "x", "mode": "text", "collection_name": "acct_user-B"},
            )
        assert resp.status_code == 200
        assert received["collection_name"] == "acct_user-A"

    def test_stream_uses_derived_collection(self):
        """D-soft: GET stream ignores ?collection_name and uses JWT-derived value."""
        from unittest.mock import MagicMock
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fake_db = MagicMock()
        fake_db.qdrant.retrieve.return_value = []
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        # The stream endpoint authenticates via get_user_for_stream (Bearer OR
        # ?st= stream token), not the blanket get_current_user gate.
        app.dependency_overrides[search_route.get_user_for_stream] = lambda: fixed_user

        with TestClient(app) as c:
            c.app.state.db_client = fake_db
            c.get("/api/v1/search/tracks/T1/stream?collection_name=acct_user-B")
        assert fake_db.qdrant.retrieve.call_args.kwargs["collection_name"] == "acct_user-A"

    def test_reaction_set_ignores_client_supplied_collection_name(self):
        """D-soft: POST reaction ignores collection_name in body, uses JWT-derived value."""
        from unittest.mock import MagicMock, patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_set_reaction(track_id, collection_name, reaction):
            captured["collection_name"] = collection_name

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.set_reaction", side_effect=fake_set_reaction):
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/search/tracks/T1/reaction",
                    json={"collection_name": "acct_user-B", "reaction": "like"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert captured["collection_name"] == "acct_user-A"
        assert data["collection_name"] == "acct_user-A"

    def test_reaction_get_uses_derived_collection(self):
        """D-soft: GET reaction ignores ?collection_name query param, uses JWT-derived value."""
        from unittest.mock import patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_get_reaction(track_id, collection_name):
            captured["collection_name"] = collection_name
            return "like"

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.get_reaction", side_effect=fake_get_reaction):
            with TestClient(app) as c:
                resp = c.get(
                    "/api/v1/search/tracks/T1/reaction?collection_name=acct_user-B",
                )
        assert resp.status_code == 200
        data = resp.json()
        assert captured["collection_name"] == "acct_user-A"
        assert data["collection_name"] == "acct_user-A"

    def test_reaction_post_without_collection_name_is_accepted(self):
        """D-soft: a client that DROPS collection_name from the body must NOT
        get a 422 — the field is optional now (frontend stops sending it at the
        same deploy as D-soft). Server still derives + uses acct_<user.id>."""
        from unittest.mock import patch
        from app.api.main import create_app
        from app.api.routes import search as search_route
        from types import SimpleNamespace

        app = create_app()
        fixed_user = SimpleNamespace(id="user-A", email="a@x")
        app.dependency_overrides[search_route.get_current_user] = lambda: fixed_user

        captured = {}

        def fake_set_reaction(track_id, collection_name, reaction):
            captured["collection_name"] = collection_name

        with patch("app.resources.metadata_db.MetadataDB.init"), \
             patch("app.resources.metadata_db.MetadataDB.set_reaction", side_effect=fake_set_reaction):
            with TestClient(app) as c:
                resp = c.post(
                    "/api/v1/search/tracks/T1/reaction",
                    json={"reaction": "like"},  # NO collection_name
                )
        assert resp.status_code == 200
        assert captured["collection_name"] == "acct_user-A"


# ── from test_search_filters_decade.py ──────────────────────────────────────

class TestFiltersDecade:
    """Tests for year_ranges (OR) filter wiring."""

    def test_no_decade_filter_does_not_emit_year_range_condition(self):
        f = _build(SearchFilters(artist="A"))
        assert f is not None
        keys = {c.key for c in f.must}
        assert "year_range" not in keys

    def test_single_decade_uses_match_any_with_one_value(self):
        f = _build(SearchFilters(year_ranges=["1990-1999"]))
        assert f is not None
        year_conds = [c for c in f.must if c.key == "year_range"]
        assert len(year_conds) == 1
        match = year_conds[0].match
        assert isinstance(match, models.MatchAny)
        assert list(match.any) == ["1990-1999"]

    def test_multiple_decades_use_single_match_any_for_or_semantics(self):
        f = _build(SearchFilters(year_ranges=["1990-1999", "2000-2009", "2010-2019"]))
        assert f is not None
        year_conds = [c for c in f.must if c.key == "year_range"]
        assert len(year_conds) == 1
        match = year_conds[0].match
        assert isinstance(match, models.MatchAny)
        assert set(match.any) == {"1990-1999", "2000-2009", "2010-2019"}

    def test_decade_combines_with_artist_and_sonic_tags(self):
        f = _build(SearchFilters(
            artist="A",
            year_ranges=["1990-1999"],
            sonic_tags=["melancholic"],
        ))
        assert f is not None
        keys = [c.key for c in f.must]
        assert keys.count("artist") == 1
        assert keys.count("year_range") == 1
        assert keys.count("sonic_tags") == 1

    def test_empty_year_ranges_list_does_not_emit_condition(self):
        f = _build(SearchFilters(artist="A", year_ranges=[]))
        assert f is not None
        keys = {c.key for c in f.must}
        assert keys == {"artist"}


# ── from test_search_filters_sonic.py ───────────────────────────────────────

class TestFiltersSonic:
    """Tests for sonic_tags (AND) filter wiring.

    We don't need a live Qdrant — Qdrant's filter-matching is its contract,
    not ours. We assert the structure of the Filter object SearchService produces."""

    def test_no_sonic_filter_returns_filter_only_with_legacy(self):
        f = _build(SearchFilters(artist="Beach House"))
        assert f is not None
        keys = {c.key for c in f.must}
        assert "artist" in keys
        assert "sonic_tags" not in keys

    def test_sonic_tags_emit_one_must_condition_per_tag_for_and_semantics(self):
        f = _build(SearchFilters(sonic_tags=["melancholic", "lo-fi"]))
        assert f is not None
        tag_conds = [c for c in f.must if c.key == "sonic_tags"]
        assert len(tag_conds) == 2
        matched_values = {c.match.value for c in tag_conds}
        assert matched_values == {"melancholic", "lo-fi"}

    def test_sonic_tags_combine_with_legacy_artist(self):
        f = _build(SearchFilters(
            artist="Beach House",
            sonic_tags=["melancholic"],
        ))
        assert f is not None
        keys = [c.key for c in f.must]
        assert keys.count("artist") == 1
        assert keys.count("sonic_tags") == 1

    def test_empty_sonic_tags_list_does_not_emit_conditions(self):
        """Empty sonic_tags=[] must be treated as 'no filter'."""
        f = _build(SearchFilters(artist="A", sonic_tags=[]))
        assert f is not None
        keys = {c.key for c in f.must}
        assert keys == {"artist"}


# ── from test_search_points_to_hits.py ──────────────────────────────────────

class TestPointsToHits:
    """Integration tests for SearchService._points_to_hits() with real ScoredPoint objects."""

    def _make_service(self) -> SearchService:
        """Create a SearchService without init (no Qdrant needed)."""
        return SearchService.__new__(SearchService)

    def test_empty_points(self):
        svc = self._make_service()
        result = svc._points_to_hits([], matched_on="lyrics")
        assert result == []

    def test_single_point_converted(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="abc123",
            version=1,
            score=0.85,
            payload={
                "title": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "year": 2020,
                "genre": "Pop",
                "duration": 200,
                "lyrics": "These are test lyrics that are long enough to be meaningful here",
                "file_path": "/test/file.flac",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert len(result) == 1
        hit = result[0]
        assert isinstance(hit, TrackHit)
        assert hit.track.title == "Test Song"
        assert hit.track.artist == "Test Artist"
        assert hit.track.album == "Test Album"
        assert hit.track.year == 2020
        assert hit.score == 0.85
        assert hit.matched_on == "lyrics"

    def test_year_parsed_from_year_range_string(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "year_range": "2010-2019",
                "duration": 180,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.year == 2010

    def test_duration_parsed_from_numeric(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 245,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.duration_sec == 245.0

    def test_lyrics_normalized(self):
        """Lyrics newlines are replaced with spaces."""
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "Line one\nLine two\nLine three of lyrics that is long enough to check",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert "\n" not in (result[0].lyrics or "")

    def test_empty_lyrics_becomes_none(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].lyrics is None

    def test_artist_facts_attached_when_collection_set(self, initialized_db):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "The Weeknd",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits(
            [point], matched_on="lyrics", collection_name="test_collection"
        )
        assert result[0].artist_facts is not None

    def test_song_facts_attached_when_collection_set(self, initialized_db):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "Blinding Lights",
                "artist": "The Weeknd",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits(
            [point], matched_on="lyrics", collection_name="test_collection"
        )
        assert result[0].song_facts is not None

    def test_missing_optional_payload_fields(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 0,
                "lyrics": "",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="lyrics")
        assert result[0].track.album is None
        assert result[0].track.year is None
        assert result[0].track.genre is None

    def test_matched_on_audio(self):
        svc = self._make_service()
        point = ScoredPoint(
            id="x",
            version=1,
            score=0.5,
            payload={
                "title": "S",
                "artist": "A",
                "duration": 200,
                "lyrics": "Some lyrics text that is long enough to pass the threshold for checking",
                "file_path": "/f",
            },
            vector={},
        )
        result = svc._points_to_hits([point], matched_on="audio")
        assert result[0].matched_on == "audio"


# ── from test_catalog_route.py ──────────────────────────────────────────────
# Integration tests for GET /api/v1/search/catalog (fake Qdrant injected — the
# engine is unit-tested separately; this checks route wiring + auth + response).

class _P:
    def __init__(self, i, p):
        self.id, self.payload = i, p


class _Count:
    def __init__(self, n):
        self.count = n


class _FakeQdrant:
    def __init__(self, pts):
        self._pts = pts

    def scroll(self, collection_name, limit, with_payload, with_vectors, offset=None):
        if offset:
            return [], None
        return [_P(i, p) for i, p in self._pts], None

    def count(self, collection_name, exact=True):
        return _Count(len(self._pts))


class _FakeDb:
    def __init__(self, q):
        self.qdrant = q


def _pt(tid, title, artist, album):
    slug = artist.lower().replace(" ", "-")
    return (tid, {
        "title": title, "artist": artist, "album": album,
        "artists": [artist], "artist_slugs": [slug], "primary_artist_slug": slug,
        "cover_art_path": None, "file_path": f"/m/{tid}.mp3", "duration": 200.0,
    })


POINTS = [
    _pt("t1", "Bohemian Rhapsody", "Queen", "A Night at the Opera"),
    _pt("t2", "Time", "Pink Floyd", "The Dark Side of the Moon"),
    _pt("t3", "Money", "Pink Floyd", "The Dark Side of the Moon"),
]


def _client():
    app = create_app()
    c = TestClient(app)
    authenticate_test_client(c, app)
    app.state.db_client = _FakeDb(_FakeQdrant(POINTS))
    invalidate_light_cache()
    catalog_search_service.invalidate()
    return c


class TestCatalogRoute:
    def test_catalog_short_query_returns_empty(self):
        r = _client().get("/api/v1/search/catalog?q=a")
        assert r.status_code == 200
        assert r.json() == []

    def test_catalog_returns_song_hit(self):
        r = _client().get("/api/v1/search/catalog?q=bohem")
        assert r.status_code == 200
        data = r.json()
        assert data and data[0]["type"] == "song"
        assert data[0]["title"] == "Bohemian Rhapsody"
        assert data[0]["track_id"] == "t1"

    def test_catalog_album_query_returns_album_entity(self):
        r = _client().get("/api/v1/search/catalog?q=dark side of the moon")
        assert r.status_code == 200
        data = r.json()
        assert data and data[0]["type"] == "album"
        assert "dark side of the moon" in data[0]["album"].lower()

    def test_catalog_requires_auth(self):
        app = create_app()
        # Seed auth_service (+owner) via a throwaway client, then hit it WITHOUT a token.
        authenticate_test_client(TestClient(app), app)
        app.state.db_client = _FakeDb(_FakeQdrant(POINTS))
        r = TestClient(app).get("/api/v1/search/catalog?q=bohem")
        assert r.status_code == 401
