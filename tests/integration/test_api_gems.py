"""Integration tests for the lyric-gems API surface.

Covers the three server touchpoints: per-track gem chips, the gem→tracks
mini-playlist source, and the lyric_gems entry in the ai-index registry
(the side-effect-import pattern unit tests can't see)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import metadata as meta_route
from app.resources.metadata_db import MetadataDB

pytestmark = pytest.mark.integration

_FIXED_USER = SimpleNamespace(id="user-gems-test", email="gems@x")
_DERIVED = "acct_user-gems-test"


def _make_app():
    app = create_app()
    app.dependency_overrides[meta_route.get_current_user] = lambda: _FIXED_USER
    return app


def _seed_gem(track_id="11111111-1111-1111-1111-111111111111"):
    MetadataDB.init()
    MetadataDB.set_track_gems(track_id, _DERIVED, [{
        "kind": "capsule", "canonical": "pager", "display": "pager",
        "quote": "Got you a beeper to feel important",
        "detail": {"item": "beeper", "ru": "пейджер"}, "score": 1.0,
    }, {
        "kind": "namedrop", "canonical": "Dr. Dre", "display": "Dr. Dre",
        "quote": "bustin' with Dr. Dre",
        "detail": {"artist_slug": "dr-dre"}, "score": 0.9,
    }])
    return track_id


class TestTrackGemsEndpoint:
    def test_empty_for_unknown_track(self):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/metadata/tracks/does-not-exist/gems")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_returns_seeded_gems_namedrop_first(self):
        tid = _seed_gem()
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get(f"/api/v1/metadata/tracks/{tid}/gems")
            assert resp.status_code == 200
            gems = resp.json()
            assert [g["kind"] for g in gems] == ["namedrop", "capsule"]
            assert gems[0]["detail"]["artist_slug"] == "dr-dre"

    def test_ru_lang_localizes_display(self):
        tid = _seed_gem()
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get(f"/api/v1/metadata/tracks/{tid}/gems", params={"lang": "ru"})
            capsule = next(g for g in resp.json() if g["kind"] == "capsule")
            assert capsule["display"] == "пейджер"
            # canonical stays the stable original-name key
            assert capsule["canonical"] == "pager"


class TestGemTracksEndpoint:
    def test_lists_tracks_carrying_the_entity(self):
        tid = _seed_gem()
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/library/gems/tracks",
                params={"kind": "namedrop", "canonical": "Dr. Dre"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["kind"] == "namedrop"
            assert any(t["track_id"] == tid for t in body["tracks"])

    def test_empty_entity(self):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get(
                "/api/v1/library/gems/tracks",
                params={"kind": "popculture", "canonical": "Nobody"},
            )
            assert resp.status_code == 200
            assert resp.json()["tracks"] == []


class TestRandomFactsExcludeLyricCommentary:
    """The home strip promises facts about a song, so anything that is really
    a comment on a lyric line stays out of it: gems (chips in the track
    drawer) and Genius line notes."""

    def _seed_editorial_fact(self):
        MetadataDB.add_song_fact(
            "eminem-forgot-about-dre", _DERIVED,
            "Recorded in a single night after a studio argument.",
            source="songfacts.com",
            artist_name="Eminem", title="Forgot About Dre",
        )

    def test_gems_never_reach_the_strip(self):
        _seed_gem()
        self._seed_editorial_fact()
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/metadata/random-facts", params={"limit": 5})
            assert resp.status_code == 200
            facts = [f["fact"] for f in resp.json()]
            assert facts == ["Recorded in a single night after a studio argument."]

    def test_genius_line_notes_never_reach_the_strip(self):
        self._seed_editorial_fact()
        MetadataDB.add_song_fact(
            "eminem-stan", _DERIVED,
            "Lyrics string: My tea's gone cold. Fact: the line opens Dido's «Thank You».",
            source="genius.com", category="genius_annotation",
            artist_name="Eminem", title="Stan",
        )
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/metadata/random-facts", params={"limit": 5})
            facts = [f["fact"] for f in resp.json()]
            assert facts == ["Recorded in a single night after a studio argument."]


class TestLyricGemsRegistry:
    def test_task_is_registered(self):
        """The ai-index route knows lyric_gems (side-effect import wiring)."""
        from app.services import ai_indexing_service
        from app.services import ai_tasks  # noqa: F401 — side effect: register

        assert "lyric_gems" in ai_indexing_service._registry

    def test_unknown_task_still_404s(self):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/library/ai-index/nope", json={"lang": "en"})
            assert resp.status_code == 404
