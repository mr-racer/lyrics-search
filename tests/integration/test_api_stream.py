"""Audio stream endpoints — consolidated suite.

Merged from:
  * test_api_stream_auth.py -> class TestStreamAuth
      Audio stream endpoint auth: must accept ?st=<stream token> because
      <audio> elements cannot send an Authorization header. These tests assert
      gate behavior only (Qdrant may be down in CI — anything that is not 401
      means the request got PAST auth).
  * test_api_stream_next.py -> classes TestStreamNext, TestStreamRoundReset,
      TestStreamSettings, TestProfileAndAxisPlaylist, TestAIEndpointsRegistered,
      TestTasteVibeRoute, TestSimilar
      Integration tests for GET /recommend/stream/next (design §10) and the
      surrounding recommend surfaces. Temp SQLite + deterministic fake Qdrant.
      The fake collection holds two sonic «islands» (cluster A around
      e0=[1,0,…], cluster B around e1=[0,1,0,…]) so session adaptation is
      observable: like A-tracks → stream leans A.

The two sources defined a same-named but different-bodied module-level `client`
fixture. The recsys (next) fixture is kept at module scope; the auth fixture is
preserved as a class-scoped override inside TestStreamAuth.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app, create_app
from app.resources.clap_features import AXIS_NAMES, AXIS_PROMPTS
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService
from ._auth_helper import authenticate_test_client

DIM = 8
N_PER_CLUSTER = 12

JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"

STREAM_PATH = "/api/v1/search/tracks/00000000-0000-0000-0000-000000000000/stream"


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

    def query_points(self, collection_name, query, using, limit, with_payload=True):
        # Modern qdrant-client: vector goes in `query`, named vector in `using`,
        # result wraps hits in .points (legacy .search() was removed upstream).
        from types import SimpleNamespace
        q = np.asarray(query, dtype=np.float32)
        q = q / np.linalg.norm(q)
        scored = []
        for p in self.points.values():
            v = np.asarray(p.vector["clap"], dtype=np.float32)
            cos = float(q @ v)
            hit = _Point(p.id, v, p.payload)
            hit.score = cos
            scored.append(hit)
        scored.sort(key=lambda h: h.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])

    def scroll(self, collection_name, limit, with_payload=True, with_vectors=False, offset=None):
        pts = list(self.points.values())
        return pts, None

    def count(self, collection_name, exact=True):
        from types import SimpleNamespace
        return SimpleNamespace(count=len(self.points))


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


def _backdate_event(coll, track, session, hours_ago, played=190.0, dur=200.0):
    """Insert a playback event with an explicit PAST played_at. The API always
    stamps CURRENT_TIMESTAMP, so any «old» history (needed to exercise the relax
    pass and the anti-repeat floor) must be written directly."""
    conn = MetadataDB._connect()
    conn.execute(
        "INSERT INTO playback_events "
        "(session_id, collection_name, track_id, played_sec, total_dur, "
        " skipped_early, interacted, played_at) "
        "VALUES (?,?,?,?,?,0,1, datetime('now', ?))",
        (session, coll, track, played, dur, f"-{hours_ago} hours"),
    )
    conn.commit()


def _login_token(c) -> str:
    r = c.post("/api/v1/auth/login",
               json={"email": "owner@example.com", "password": "ownerpass12345"})
    return r.json()["token"]


def _stream_token(c) -> str:
    tok = _login_token(c)
    r = c.post("/api/v1/auth/stream-token",
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    return r.json()["token"]


class TestStreamAuth:
    """Audio stream endpoint auth: must accept ?st=<stream token> because
    <audio> elements cannot send an Authorization header.

    These tests assert gate behavior only (Qdrant may be down in CI — anything
    that is not 401 means the request got PAST auth).
    """

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
        app = create_app()
        with TestClient(app) as c:
            conn = MetadataDB.get()
            conn.execute("DELETE FROM instance_config")
            conn.execute("DELETE FROM users")
            conn.commit()
            MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
            auth = AuthService(jwt_secret=JWT_SECRET)
            auth.create_owner(email="owner@example.com", password="ownerpass12345")
            yield c

    def test_stream_401_without_any_credentials(self, client):
        assert client.get(STREAM_PATH).status_code == 401

    def test_stream_passes_gate_with_st_query(self, client):
        st = _stream_token(client)
        r = client.get(f"{STREAM_PATH}?st={st}")
        assert r.status_code != 401, f"?st= rejected: {r.status_code} {r.text[:200]}"

    def test_stream_passes_gate_with_bearer_header(self, client):
        tok = _login_token(client)
        r = client.get(STREAM_PATH, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code != 401

    def test_stream_401_with_login_token_in_st(self, client):
        """Full login tokens must not be usable in URLs."""
        tok = _login_token(client)
        assert client.get(f"{STREAM_PATH}?st={tok}").status_code == 401

    def test_other_search_routes_still_gated(self, client):
        """Carving /stream out of the blanket gate must not open anything else."""
        r = client.post("/api/v1/search/", json={"query": "x", "mode": "text"})
        assert r.status_code == 401


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
        assert {t["pool"] for t in body["tracks"]} <= {"fresh", "familiar", "liked"}
        assert body["diagnostics"]["clusters"]["positive"]
        # every track carries full metadata for the player
        for t in body["tracks"]:
            assert t["title"] and t["artist"] and t["file_path"]
        # uniform full listens only → contributions indistinguishable → no badge
        assert body["session_adaptation"] is None

    def test_fire_lights_the_session_adaptation_badge(self, client):
        """A session fire makes contributions distinguishable: the chunk ships
        «подстроились под твой вайб» with the fired track's metadata."""
        for i in range(3):
            _post_event(client, f"a{i}", session="s_fire")
        r = client.post("/api/v1/recommend/taste-signal", json={
            "session_id": "s_fire", "track_id": "a1", "kind": "fire"})
        assert r.status_code == 200

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s_fire", "n": 3})
        body = resp.json()
        adapt = body["session_adaptation"]
        assert adapt["active"] is True
        assert adapt["tracks"][0]["track_id"] == "a1"
        assert adapt["tracks"][0]["title"]

    def test_taste_signal_state_reflects_latest_reaction(self, client):
        """GET /taste-signal/state returns the newest reaction per track (water
        cancels an older fire), fresh signals are locked, unknown tracks omitted."""
        client.post("/api/v1/recommend/taste-signal", json={
            "session_id": "s1", "track_id": "hot", "kind": "fire"})
        # water over an older fire on the same track → latest-wins
        client.post("/api/v1/recommend/taste-signal", json={
            "session_id": "s1", "track_id": "hot", "kind": "fire"})
        client.post("/api/v1/recommend/taste-signal", json={
            "session_id": "s1", "track_id": "hot", "kind": "water"})

        resp = client.get("/api/v1/recommend/taste-signal/state",
                          params={"track_ids": "hot,never_touched"})
        assert resp.status_code == 200
        states = resp.json()["states"]
        assert set(states) == {"hot"}                 # untouched track omitted
        assert states["hot"]["kind"] == "water"       # water superseded the fire
        assert states["hot"]["contribution"] == pytest.approx(1.0, abs=1e-2)
        assert states["hot"]["locked"] is True        # fresh → still charged >50%

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
                          params={"session_id": "live", "n": 3, "liked_share": 0.5})
        body = resp.json()
        # 2026-08-03 §4.5: the session LEADS once it has signals — the long-term
        # seed is down to its floor, not holding half the vote as it used to.
        assert body["diagnostics"]["w_long"] == pytest.approx(0.15)
        # The strongest pull cluster is built from this session's B-cluster
        # listening, and unheard B tracks surface in the queue.
        positive = body["diagnostics"]["clusters"]["positive"]
        assert positive and positive[0]["track_id"].startswith("b")
        assert any(t["track_id"].startswith("b") for t in body["tracks"])

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

    def test_orphaned_liked_track_not_served(self, client):
        """Re-indexing mints fresh uuid4 ids, orphaning old likes in SQLite.
        The liked pool must drop ids that no longer resolve in Qdrant instead of
        emitting an empty-payload «—» track that 404s on /stream and /lyrics."""
        coll = _owner_collection(client)
        for i in range(3):
            _post_event(client, f"a{i}", session="old")
        # a real like + a stale like whose track no longer exists in Qdrant
        _like(client, coll, "a3")
        _like(client, coll, "ghost_reindexed")

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s_orphan", "n": 3, "liked_share": 1.0})
        body = resp.json()
        served = {t["track_id"] for t in body["tracks"]}
        assert "ghost_reindexed" not in served
        # no leaked placeholder track in the chunk
        for t in body["tracks"]:
            assert t["title"] != "—"
            assert t["file_path"], "served track must resolve to a playable file"

    def test_liked_share_full_returns_only_liked(self, client):
        coll = _owner_collection(client)
        # Favorites are COMPUTED from deep listens; backdated past the 8h
        # cooldown and the just-heard floor so they are eligible again.
        for tid in ("a3", "a4", "a5", "b1", "b2"):
            _backdate_event(coll, tid, "old", hours_ago=20)

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s9", "n": 3, "liked_share": 1.0})
        body = resp.json()
        assert len(body["tracks"]) == 3
        # 2026-08-03 §6: the «ЛЮБИМОЕ» end is the not-fresh side — computed
        # favorites plus already-heard neighbours, never an unheard track.
        assert all(t["pool"] != "fresh" for t in body["tracks"])
        assert any(t["pool"] == "liked" for t in body["tracks"])

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
        assert all(t["pool"] == "fresh" for t in body["tracks"])

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


class TestStreamRoundReset:
    """«Бесконечный круг» (design 2026-06-14): a session that has heard everything
    replays instead of going empty, the just-heard floor holds, vibe is carried."""

    def test_exhausted_session_replays_instead_of_empty(self, client):
        """«Долго слушал → Поток пуст» regression: one session plays the WHOLE
        library, yet the next request still returns a full chunk (via fallback)."""
        for cluster in ("a", "b"):
            for i in range(N_PER_CLUSTER):
                _post_event(client, f"{cluster}{i}", session="marathon")
        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "marathon", "n": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["tracks"]) == 3
        assert all(t["file_path"] for t in body["tracks"])

    def test_relax_replays_old_but_never_the_floor(self, client):
        """Whole library played in one session, most of it long ago: the relax pass
        serves older tracks while the anti-repeat floor keeps the just-heard out."""
        coll = _owner_collection(client)
        for cluster in ("a", "b"):
            for i in range(N_PER_CLUSTER):
                _backdate_event(coll, f"{cluster}{i}", "S", hours_ago=120)  # 5 days
        # a0, a1 heard minutes ago → inside the 30-min floor
        _backdate_event(coll, "a0", "S", hours_ago=0.1)
        _backdate_event(coll, "a1", "S", hours_ago=0.1)

        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "S", "n": 3, "liked_share": 0.5})
        body = resp.json()
        served = {t["track_id"] for t in body["tracks"]}
        assert len(served) == 3                    # filled from replay, not empty
        assert served.isdisjoint({"a0", "a1"})     # floor holds across the round
        assert body["diagnostics"]["relaxed"] is True

    def test_rare_extreme_goes_short_rather_than_replay(self, client):
        """2026-08-03 §6: at the «РЕДКОЕ» extreme freshness outranks непустота.
        A library heard end to end five days ago has nothing unheard left, so
        the chunk comes back short and says so — it does NOT quietly replay."""
        coll = _owner_collection(client)
        for cluster in ("a", "b"):
            for i in range(N_PER_CLUSTER):
                _backdate_event(coll, f"{cluster}{i}", "S2", hours_ago=120)

        body = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "S2", "n": 3,
                                  "liked_share": 0.0}).json()
        assert body["tracks"] == []
        assert body["diagnostics"]["fresh_exhausted"] is True
        assert body["diagnostics"]["relaxed"] is False

    def test_vibe_carried_into_replay(self, client):
        """Session blend (w_s) must survive into the relaxed round — the replay
        still follows the mood the listener drifted into this session."""
        coll = _owner_collection(client)
        for cluster in ("a", "b"):
            for i in range(N_PER_CLUSTER):
                _backdate_event(coll, f"{cluster}{i}", "S", hours_ago=120)
        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "S", "n": 3, "liked_share": 0.5})
        # 24 session signals push the long-term seed to its floor — the replay
        # follows the mood the listener drifted into, not the all-time profile.
        assert resp.json()["diagnostics"]["w_long"] == pytest.approx(0.15)

    def test_diagnostics_round_increments_after_a_full_cycle(self, client):
        """round = ceil(total session plays / library size): a 2nd lap reads as 2."""
        coll = _owner_collection(client)
        for cluster in ("a", "b"):              # 24 distinct plays
            for i in range(N_PER_CLUSTER):
                _backdate_event(coll, f"{cluster}{i}", "cyc", hours_ago=2)
        for i in range(6):                       # +6 repeats → 30 events over 24 tracks
            _backdate_event(coll, f"a{i}", "cyc", hours_ago=0.1)
        resp = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "cyc", "n": 3})
        assert resp.json()["diagnostics"]["round"] == 2   # ceil(30 / 24)


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
        # Islands feed on fires + ≥85% completions (hearts gone). a0's fire
        # supersedes a0's own completion (2026-08-03 §2.1), so it is 1 fire +
        # the 2 remaining completions = 3 signals.
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name=coll, track_id="a0", kind="fire")

        resp = client.get("/api/v1/recommend/profile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_signals"] == 3
        assert body["islands"], "expected at least one island"
        assert body["islands"][0]["tracks"][0]["title"]
        # Vibes (the fast mood layer) ship in the same response; the fresh
        # a-cluster signals must also form at least one vibe.
        assert body["vibes"], "expected at least one vibe"
        assert body["vibes"][0]["tracks"][0]["title"]
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

    def test_axis_playlist_without_stats_is_empty(self, client, monkeypatch):
        # An empty stats table is no longer enough to reach this branch: the
        # bundled norm reference in clap_features stands in for a collection
        # that was never indexed. Take that away too, or the axis term stays
        # live and the endpoint happily ranks.
        monkeypatch.setattr(
            "app.resources.clap_features.load_axis_norm_reference", lambda: None,
        )
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
        # `headline` — the 2-4 word name for this taste — rides along with the
        # portrait; the route passes the enrichment through untouched.
        assert resp.json() == {"portrait": "П.", "island_names": {"x": "Имя"},
                               "headline": None}

    def test_ai_playlist_llm_failure_returns_502(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        with patch.object(rec_route.recsys_ai_service, "ai_playlist",
                          new=AsyncMock(side_effect=RuntimeError("llm down"))):
            resp = client.post("/api/v1/recommend/ai-playlist",
                               json={"prompt": "что-нибудь"})
        assert resp.status_code == 502


class TestTasteVibeRoute:
    """GET /recommend/taste-vibe: on a cache miss with AI available the phrase
    must be generated SYNCHRONOUSLY (not deferred to a background task) so the
    hero shows the AI line on the first request. Service funcs are patched, so
    the fake Qdrant is never actually read."""

    def test_generates_synchronously_when_ai_available(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        with patch.object(rec_route.recsys_ai_service, "taste_vibe_cached_or_fallback",
                          return_value={"phrase": "детерминированная",
                                        "source": "fallback", "needs_generation": True}), \
             patch.object(rec_route.recsys_ai_service, "generate_taste_vibe",
                          new=AsyncMock(return_value={"phrase": "AI вайб", "source": "ai"})) as gen, \
             patch.object(rec_route.settings_service, "ai_available", return_value=True):
            resp = client.get("/api/v1/recommend/taste-vibe", params={"lang": "ru"})

        assert resp.status_code == 200
        assert resp.json() == {"phrase": "AI вайб", "source": "ai"}
        gen.assert_awaited_once()

    def test_deterministic_when_ai_unavailable(self, client):
        from unittest.mock import AsyncMock, patch
        from app.api.routes import recommend as rec_route

        with patch.object(rec_route.recsys_ai_service, "taste_vibe_cached_or_fallback",
                          return_value={"phrase": "детерминированная",
                                        "source": "fallback", "needs_generation": True}), \
             patch.object(rec_route.recsys_ai_service, "generate_taste_vibe",
                          new=AsyncMock()) as gen, \
             patch.object(rec_route.settings_service, "ai_available", return_value=False):
            resp = client.get("/api/v1/recommend/taste-vibe", params={"lang": "ru"})

        assert resp.status_code == 200
        assert resp.json() == {"phrase": "детерминированная", "source": "fallback"}
        gen.assert_not_awaited()


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


class TestTasteSignalRoute:
    """POST /recommend/taste-signal — огонёк/вода journal write + wave steer."""

    def test_fire_recorded_and_surfaces_in_diagnostics(self, client):
        coll = _owner_collection(client)
        for i in range(3):
            _post_event(client, f"a{i}", session="s1")
        r = client.post("/api/v1/recommend/taste-signal",
                        json={"session_id": "s1", "track_id": "a0", "kind": "fire"})
        assert r.status_code == 200
        assert isinstance(r.json()["id"], int)
        # persisted in the per-account journal
        sigs = {(t, k) for t, k, _ in MetadataDB.get_taste_signals(coll, session_id="s1")}
        assert ("a0", "fire") in sigs
        # and steers the wave: the fired track is pulled into a positive cluster
        body = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s1", "n": 3}).json()
        positive = body["diagnostics"]["clusters"]["positive"]
        assert positive, "a fresh fire must produce at least one pull cluster"

    def test_water_recorded(self, client):
        r = client.post("/api/v1/recommend/taste-signal",
                        json={"session_id": "s1", "track_id": "a0", "kind": "water"})
        assert r.status_code == 200

    def test_water_mutes_the_track_for_days(self, client):
        """2026-08-03 §3: water is a real debuff — the track stops being served
        outright for WATER_MUTE_DAYS, not just softly demoted for four hours."""
        for i in range(3):
            _post_event(client, f"a{i}", session="s2")
        client.post("/api/v1/recommend/taste-signal",
                    json={"session_id": "s2", "track_id": "a0", "kind": "water"})
        body = client.get("/api/v1/recommend/stream/next",
                          params={"session_id": "s2", "n": 5}).json()
        assert body["diagnostics"]["n_muted"] >= 1
        assert all(t["track_id"] != "a0" for t in body["tracks"])

    def test_invalid_kind_rejected(self, client):
        r = client.post("/api/v1/recommend/taste-signal",
                        json={"session_id": "s1", "track_id": "a0", "kind": "love"})
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post("/api/v1/recommend/taste-signal",
                        headers={"Authorization": "Bearer not-a-real-token"},
                        json={"session_id": "s1", "track_id": "a0", "kind": "fire"})
        assert r.status_code == 401
