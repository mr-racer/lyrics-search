"""Integration tests for GET /recommend/stream/next (design §10).

Temp SQLite + deterministic fake Qdrant. The fake collection holds two sonic
«islands» (cluster A around e0=[1,0,…], cluster B around e1=[0,1,0,…]) so
session adaptation is observable: like A-tracks → stream leans A.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client

DIM = 8
N_PER_CLUSTER = 12


def _vec(cluster: str, i: int) -> np.ndarray:
    """Unit vectors: cluster a → near e0, cluster b → near e1."""
    base = np.zeros(DIM, dtype=np.float32)
    base[0 if cluster == "a" else 1] = 1.0
    base[2 + (i % 3)] = 0.15 + 0.02 * i   # small per-track variation
    return base / np.linalg.norm(base)


class _Point:
    def __init__(self, tid, vector, payload):
        self.id = tid
        self.vector = {"clap": vector.tolist()}
        self.payload = payload
        self.score = None


class FakeQdrant:
    """Cosine-faithful in-memory stand-in for the three calls the stream uses."""

    def __init__(self):
        self.points: dict[str, _Point] = {}
        for cluster in ("a", "b"):
            for i in range(N_PER_CLUSTER):
                tid = f"{cluster}{i}"
                axes = {ax: (0.5 if cluster == "a" else -0.5) for ax in AXIS_NAMES}
                payload = {
                    "title": f"T{tid}", "artist": f"artist_{tid}",
                    "album": None, "year": 2020, "genre": "g",
                    "duration": 200.0, "file_path": f"/{tid}.mp3",
                    "cover_art_path": None, "sonic_axes": axes,
                }
                self.points[tid] = _Point(tid, _vec(cluster, i), payload)

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self.points[t] for t in ids if t in self.points]

    def search(self, collection_name, query_vector, limit, with_payload=True):
        q = np.asarray(query_vector[1], dtype=np.float32)
        q = q / np.linalg.norm(q)
        scored = []
        for p in self.points.values():
            v = np.asarray(p.vector["clap"], dtype=np.float32)
            cos = float(q @ v)
            hit = _Point(p.id, v, p.payload)
            hit.score = cos
            scored.append(hit)
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def scroll(self, collection_name, limit, with_payload=True, with_vectors=False, offset=None):
        pts = list(self.points.values())
        return pts, None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    fake = FakeQdrant()
    from unittest.mock import MagicMock
    db = MagicMock(); db.qdrant = fake
    app.state.db_client = db
    # ai-playlist needs a search_service on app.state; the AI tests patch the
    # service function itself, so a MagicMock is never actually searched.
    app.state.search_service = MagicMock()
    c = TestClient(app)
    authenticate_test_client(c, app)
    c.fake_qdrant = fake
    yield c
    MetadataDB._reset_for_tests()
    app.state.db_client = None
    app.state.search_service = None


def _owner_collection(client) -> str:
    """The collection the JWT derives — acct_<owner id>."""
    me = client.get("/api/v1/auth/me").json()
    return f"acct_{me['user']['id'] if 'user' in me else me['id']}"


def _post_event(client, track, session="s1", played=190.0, dur=200.0, interacted=True):
    r = client.post("/api/v1/playback/events", json={
        "session_id": session, "track_id": track,
        "played_sec": played, "total_dur": dur, "interacted": interacted,
    })
    assert r.status_code == 200


def _like(client, coll, track):
    MetadataDB.set_reaction(track, coll, "like")


def _dislike(client, coll, track):
    MetadataDB.set_reaction(track, coll, "dislike")


class TestStreamNext:
    def test_returns_tracks_with_pools_and_diagnostics(self, client):
        coll = _owner_collection(client)
        for i in range(3):
            _post_event(client, f"a{i}")
        _like(client, coll, "a0")

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s2", "n": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["tracks"]) == 3
        assert {t["pool"] for t in body["tracks"]} <= {"anchor", "explore", "liked"}
        assert body["diagnostics"]["anchors"]
        # every track carries full metadata for the player
        for t in body["tracks"]:
            assert t["title"] and t["artist"] and t["file_path"]

    def test_session_signals_adapt_queue_toward_cluster(self, client):
        """«15 событий → очередь адаптировалась»: full listens of B-cluster in
        the session must surface B-cluster anchor candidates."""
        coll = _owner_collection(client)
        # long-term history: cluster A
        for i in range(5):
            _post_event(client, f"a{i}", session="old")
        # current session: 15 strong B signals
        for i in range(10):
            _post_event(client, f"b{i}", session="live")
        for i in range(5):
            _like(client, coll, f"b{i+5}")

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "live", "n": 3, "liked_share": 0.0})
        body = resp.json()
        # w_s ramps with session signals but caps at 0.5 — long-term always
        # keeps an equal vote (anti-trap), so a wild session can't take over.
        assert body["diagnostics"]["w_session"] == 0.5
        # Session taste still leads the anchor set: the strongest anchor must
        # be a session (B-cluster) track thanks to the ×(1+2·w_s) boost…
        assert body["diagnostics"]["anchors"][0]["track_id"].startswith("b")
        # …and unheard B-cluster tracks surface in the queue. Long-term (A)
        # anchors legitimately remain — the design blends, never erases.
        anchor_tracks = [t for t in body["tracks"] if t["pool"] == "anchor"]
        assert anchor_tracks, "expected anchor-pool tracks"
        assert any(t["track_id"].startswith("b") for t in anchor_tracks)

    def test_disliked_track_never_served(self, client):
        """«Дизлайк исчезает немедленно» — жёсткий фильтр."""
        coll = _owner_collection(client)
        for i in range(4):
            _post_event(client, f"a{i}")
        _dislike(client, coll, "a5")

        for attempt in range(5):
            resp = client.get("/api/v1/recommend/stream/next",
                              params={"session_id": f"s{attempt}", "n": 3})
            served = {t["track_id"] for t in resp.json()["tracks"]}
            assert "a5" not in served

    def test_liked_share_full_returns_only_liked(self, client):
        coll = _owner_collection(client)
        for i in range(3):
            _post_event(client, f"a{i}", session="old")
        for tid in ("a3", "a4", "a5", "b1", "b2"):
            _like(client, coll, tid)

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s9", "n": 3, "liked_share": 1.0})
        body = resp.json()
        assert len(body["tracks"]) == 3
        assert all(t["pool"] == "liked" for t in body["tracks"])

    def test_small_collection_topup_still_fills_chunk(self, client):
        """Undersupply path: almost everything excluded → chunk still fills
        from what remains (ступенчатое смягчение, не недобор)."""
        coll = _owner_collection(client)
        _post_event(client, "a0")
        _like(client, coll, "a1")
        # dislike everything in cluster b — shrink the candidate space hard
        for i in range(N_PER_CLUSTER):
            _dislike(client, coll, f"b{i}")

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s3", "n": 3})
        body = resp.json()
        assert len(body["tracks"]) >= 2  # filled despite half the library disliked
        assert all(not t["track_id"].startswith("b") for t in body["tracks"])

    def test_cold_start_returns_exploration(self, client):
        """No history at all → pure exploration, not an empty stream."""
        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "fresh", "n": 3, "liked_share": 0.0})
        body = resp.json()
        assert len(body["tracks"]) == 3
        assert all(t["pool"] == "explore" for t in body["tracks"])

    def test_exclude_ids_respected(self, client):
        for i in range(3):
            _post_event(client, f"a{i}")
        resp = client.get("/api/v1/recommend/stream/next", params={
            "session_id": "s4", "n": 3,
            "exclude_ids": "a3,a4,a5,a6,a7,a8,a9,a10,a11",
        })
        served = {t["track_id"] for t in resp.json()["tracks"]}
        assert served.isdisjoint({f"a{i}" for i in range(3, 12)})

    def test_db_down_returns_503(self, client):
        app.state.db_client = None
        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s5"})
        assert resp.status_code == 503


class TestStreamSettings:
    def test_put_persists_and_get_uses_it(self, client):
        coll = _owner_collection(client)
        r = client.put("/api/v1/recommend/stream/settings", json={"liked_share": 0.7})
        assert r.status_code == 200
        assert MetadataDB.get_stream_liked_share(coll) == pytest.approx(0.7)

    def test_put_validates_range(self, client):
        assert client.put("/api/v1/recommend/stream/settings",
                          json={"liked_share": 1.5}).status_code == 422


class TestProfileAndAxisPlaylist:
    def test_profile_returns_axes_islands_confidence(self, client):
        coll = _owner_collection(client)
        MetadataDB.set_axis_norm_stats(coll, {
            "version": __import__("app.resources.clap_features", fromlist=["axis_version"]).axis_version(),
            "n": 100,
            "mean": {a: 0.0 for a in AXIS_NAMES},
            "std": {a: 1.0 for a in AXIS_NAMES},
        })
        for i in range(3):
            _post_event(client, f"a{i}")
        _like(client, coll, "a0")

        resp = client.get("/api/v1/recommend/profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_signals"] == 4
        assert body["islands"], "expected at least one island"
        assert body["islands"][0]["tracks"][0]["title"]
        assert set(body["axes"]) == set(AXIS_NAMES)
        assert body["portrait"] is None  # LLM enrichment not generated yet

    def test_axis_playlist_endpoint(self, client):
        coll = _owner_collection(client)
        MetadataDB.set_axis_norm_stats(coll, {
            "version": __import__("app.resources.clap_features", fromlist=["axis_version"]).axis_version(),
            "n": 100,
            "mean": {a: 0.0 for a in AXIS_NAMES},
            "std": {a: 1.0 for a in AXIS_NAMES},
        })
        resp = client.post("/api/v1/recommend/axis-playlist",
                           json={"targets": {"energy": 0.5}, "limit": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["tracks"]) == 5
        assert all(t["pool"] == "axis" for t in body["tracks"])
        # cluster a payloads sit at +0.5 on every axis → they match the target
        assert all(t["track_id"].startswith("a") for t in body["tracks"])

    def test_axis_playlist_without_stats_is_empty(self, client):
        resp = client.post("/api/v1/recommend/axis-playlist",
                           json={"targets": {}, "limit": 5})
        assert resp.status_code == 200
        assert resp.json()["tracks"] == []
        assert resp.json()["diagnostics"]["reason"] == "no_axis_stats"

    def test_profile_attaches_fresh_llm_enrichment(self, client):
        """A cached enrichment matching the current islands hash is served;
        islands get their LLM names attached by representative track_id."""
        from app.services.recsys_ai_service import PROFILE_ENRICH_KIND, profile_source_hash

        coll = _owner_collection(client)
        for i in range(3):
            _post_event(client, f"a{i}")
        _like(client, coll, "a0")

        islands = client.get("/api/v1/recommend/profile").json()["islands"]
        assert islands
        rep = islands[0]["track_id"]
        MetadataDB.set_recsys_llm_text(
            coll, PROFILE_ENRICH_KIND, "ru", profile_source_hash(islands),
            {"portrait": "Портрет.", "island_names": {rep: "Ночной синтвейв"}},
        )

        body = client.get("/api/v1/recommend/profile", params={"lang": "ru"}).json()
        assert body["portrait"] == "Портрет."
        assert body["islands"][0]["name"] == "Ночной синтвейв"
        # other language has no cache → no portrait
        assert client.get("/api/v1/recommend/profile",
                          params={"lang": "en"}).json()["portrait"] is None


class TestAIEndpointsRegistered:
    """Per [[feedback_registry_pattern_needs_e2e]]: hit the live routes, do not
    trust unit coverage — a missing include_router fails only here. LLM calls
    are patched; we verify routing + request/response wiring."""

    def test_ai_playlist_route_wired(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        canned = {
            "title": "Тест",
            "steps": [{"tool": "clap_search", "query": "q", "found": 1}],
            "tracks": [{"track_id": "a0", "title": "T", "artist": "A",
                        "duration": 100.0, "file_path": "/a0.mp3", "tool": "clap_search",
                        "reason": "ок"}],
        }
        with patch.object(rec_route.recsys_ai_service, "ai_playlist",
                          new=AsyncMock(return_value=canned)) as m:
            resp = client.post("/api/v1/recommend/ai-playlist",
                               json={"prompt": "дождь и кофе", "lang": "ru"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Тест"
        assert body["tracks"][0]["reason"] == "ок"
        assert body["tracks"][0]["source_tool"] == "clap_search"
        assert m.call_args.kwargs["collection_name"].startswith("acct_")

    def test_ai_enrich_route_wired(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        canned = {"portrait": "П.", "island_names": {"x": "Имя"}, "islands": []}
        with patch.object(rec_route.recsys_ai_service, "enrich_profile",
                          new=AsyncMock(return_value=canned)):
            resp = client.post("/api/v1/recommend/profile/ai-enrich",
                               json={"lang": "ru"})
        assert resp.status_code == 200
        assert resp.json() == {"portrait": "П.", "island_names": {"x": "Имя"}}

    def test_ai_playlist_llm_failure_returns_502(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        with patch.object(rec_route.recsys_ai_service, "ai_playlist",
                          new=AsyncMock(side_effect=RuntimeError("llm down"))):
            resp = client.post("/api/v1/recommend/ai-playlist",
                               json={"prompt": "что-нибудь"})
        assert resp.status_code == 502


class TestSimilar:
    def test_returns_neighbors_with_seed_excluded(self, client):
        resp = client.get("/api/v1/recommend/similar",
                          params={"track_id": "a0", "limit": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["seed_track_id"] == "a0"
        ids = [t["track_id"] for t in body["tracks"]]
        assert len(ids) == 5
        assert "a0" not in ids
        # same-cluster tracks dominate (cosine-faithful fake)
        assert sum(1 for t in ids if t.startswith("a")) >= 4
        assert all(t["anchor_track_id"] == "a0" for t in body["tracks"])

    def test_disliked_neighbor_filtered(self, client):
        coll = _owner_collection(client)
        MetadataDB.set_reaction("a1", coll, "dislike")
        resp = client.get("/api/v1/recommend/similar",
                          params={"track_id": "a0", "limit": 10})
        assert "a1" not in {t["track_id"] for t in resp.json()["tracks"]}

    def test_unknown_seed_returns_empty(self, client):
        resp = client.get("/api/v1/recommend/similar",
                          params={"track_id": "ghost", "limit": 5})
        assert resp.status_code == 200
        assert resp.json()["tracks"] == []
