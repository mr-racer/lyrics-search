"""Unit tests for recsys_ai_service.

Consolidates:
  * test_recsys_ai_service.py        -> enrichment cache + plan/execute/select +
                                         taste-vibe (kept as classes)
  * test_recsys_ai_enrich_headline.py -> TestAiEnrichHeadline
  * test_recsys_library_search.py     -> TestLibrarySearch
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import catalog_search_service, recsys_ai_service
from app.services import recsys_ai_service as svc
from app.services.recsys_ai_service import (
    _validate_plan,
    ai_playlist,
    enrich_profile,
    get_cached_enrichment,
    profile_source_hash,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _island(rep, members):
    return {
        "track_id": rep, "weight": 1.0,
        "tracks": [{"track_id": m, "title": f"T{m}", "artist": "A",
                    "cover_art_path": None} for m in members],
    }


class TestSourceHash:
    def test_stable_across_member_order(self):
        a = [_island("x", ["x", "y"]), _island("z", ["z"])]
        b = [_island("z", ["z"]), _island("x", ["y", "x"])]
        assert profile_source_hash(a) == profile_source_hash(b)

    def test_changes_on_membership_change(self):
        a = [_island("x", ["x", "y"])]
        b = [_island("x", ["x", "y", "new"])]
        assert profile_source_hash(a) != profile_source_hash(b)


class TestEnrichmentCache:
    def test_roundtrip_and_staleness(self):
        islands = [_island("x", ["x", "y"])]
        MetadataDB.set_recsys_llm_text(
            "col", svc.PROFILE_ENRICH_KIND, "ru",
            profile_source_hash(islands),
            {"portrait": "П", "island_names": {"x": "Остров"}},
        )
        hit = get_cached_enrichment("col", "ru", islands)
        assert hit["portrait"] == "П"
        # different lang → miss
        assert get_cached_enrichment("col", "en", islands) is None
        # taste drifted → stale → miss
        drifted = [_island("x", ["x", "y", "new"])]
        assert get_cached_enrichment("col", "ru", drifted) is None


class TestEnrichProfile:
    @pytest.mark.asyncio
    async def test_generates_validates_and_caches(self):
        islands = [_island("x", ["x", "y"]), _island("z", ["z"])]
        profile = {"islands": islands, "axes": None, "confidence": 0.5, "n_signals": 7,
                   "axis_stats_source": None}
        llm_out = {
            "portrait": "  Любит контрасты.  ",
            "island_names": {"x": "Ночной синтвейв", "z": " Грязный бум-бэп ",
                             "ghost": "должно отфильтроваться"},
        }
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "ask_llm", new=AsyncMock(return_value=llm_out)):
            out = await enrich_profile(qdrant_client=object(), collection_name="col", lang="ru")

        assert out["portrait"] == "Любит контрасты."
        assert out["island_names"] == {"x": "Ночной синтвейв", "z": "Грязный бум-бэп"}
        # persisted and readable through the staleness-checked getter
        assert get_cached_enrichment("col", "ru", islands)["portrait"] == "Любит контрасты."

    @pytest.mark.asyncio
    async def test_empty_profile_skips_llm(self):
        profile = {"islands": [], "axes": None, "confidence": 0.0, "n_signals": 0,
                   "axis_stats_source": None}
        mock_llm = AsyncMock()
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "ask_llm", new=mock_llm):
            out = await enrich_profile(qdrant_client=object(), collection_name="col")
        assert out == {"portrait": None, "island_names": {}, "islands": []}
        mock_llm.assert_not_awaited()


class TestValidatePlan:
    def test_drops_bad_tools_and_caps(self):
        title, actions = _validate_plan({
            "title": "Мой плейлист",
            "actions": [
                {"tool": "clap_search", "query": "calm jazz", "limit": 999},
                {"tool": "rm_rf", "query": "evil"},
                {"tool": "library_search", "query": ""},
                {"tool": "similar_tracks", "query": "Artist Song"},
            ],
        })
        assert title == "Мой плейлист"
        assert [a["tool"] for a in actions] == ["clap_search", "similar_tracks"]
        assert actions[0]["limit"] == svc.MAX_ACTION_LIMIT

    def test_empty_plan(self):
        title, actions = _validate_plan({})
        assert title == "Playlist" and actions == []


def _hit(tid, title=None, artist="A"):
    track = SimpleNamespace(
        track_id=tid, title=title or f"T{tid}", artist=artist, album=None,
        year=2020, genre="g", duration_sec=200.0, file_path=f"/{tid}.mp3",
        cover_art_path=None,
    )
    return SimpleNamespace(track=track, score=0.9)


def _trackdict(tid, title=None, artist="A"):
    """A catalog tracks-mode result dict (what search_catalog_tracks returns)."""
    return {
        "track_id": tid, "title": title or f"T{tid}", "artist": artist,
        "album": None, "year": 2020, "genre": "g", "duration": 200.0,
        "file_path": f"/{tid}.mp3", "cover_art_path": None,
    }


class FakeSearchService:
    def __init__(self):
        self.calls = []

    async def search(self, query, mode="text", limit=10, collection_name=None, **kw):
        self.calls.append({"query": query, "mode": mode, "limit": limit})
        if mode == "audio":
            return [_hit("au1"), _hit("au2")]
        if mode == "hybrid":
            return [_hit("hy1"), _hit("au1")]   # au1 dups across tools
        return [_hit("seed")]                    # text mode resolves the seed


class TestAIPlaylist:
    @pytest.mark.asyncio
    async def test_plan_execute_select_happy_path(self):
        plan = {"title": "Дождливый вечер",
                "actions": [{"tool": "clap_search", "query": "rainy calm", "limit": 5},
                            {"tool": "library_search", "query": "rain lyrics", "limit": 5}]}
        selection = {"picks": [{"n": 2, "reason": "мягкий вокал"},
                               {"n": 1, "reason": "дождливый бит"},
                               {"n": 99, "reason": "мимо"}]}
        fake_search = FakeSearchService()
        # library_search now routes to the catalog tracks engine (by name → songs).
        with patch.object(svc, "ask_llm", new=AsyncMock(side_effect=[plan, selection])), \
             patch.object(catalog_search_service, "search_catalog_tracks",
                          return_value=[_trackdict("hy1"), _trackdict("au1")]):
            out = await ai_playlist(
                search_service=fake_search, qdrant_client=object(),
                collection_name="col", prompt="дождь и кофе", lang="ru", limit=5,
            )

        assert out["title"] == "Дождливый вечер"
        assert [s["tool"] for s in out["steps"]] == ["clap_search", "library_search"]
        # dedup: au1 found by both tools → 3 unique candidates (au1, au2, hy1)
        assert out["steps"][0]["found"] == 2 and out["steps"][1]["found"] == 2
        # selection mapped by index over the deduped order, invalid n=99 dropped
        assert [t["track_id"] for t in out["tracks"]] == ["au2", "au1"]
        assert out["tracks"][0]["reason"] == "мягкий вокал"

    @pytest.mark.asyncio
    async def test_selection_failure_falls_back_to_candidates(self):
        plan = {"title": "X", "actions": [{"tool": "clap_search", "query": "q", "limit": 5}]}
        with patch.object(svc, "ask_llm",
                          new=AsyncMock(side_effect=[plan, RuntimeError("llm down")])):
            out = await ai_playlist(
                search_service=FakeSearchService(), qdrant_client=object(),
                collection_name="col", prompt="что-нибудь", limit=5,
            )
        assert [t["track_id"] for t in out["tracks"]] == ["au1", "au2"]
        assert all(t["reason"] is None for t in out["tracks"])

    @pytest.mark.asyncio
    async def test_degenerate_plan_falls_back_to_clap_search(self):
        plan = {"title": "X", "actions": [{"tool": "hack", "query": "evil"}]}
        selection = {"picks": [{"n": 1, "reason": "ok"}]}
        fake_search = FakeSearchService()
        with patch.object(svc, "ask_llm", new=AsyncMock(side_effect=[plan, selection])):
            out = await ai_playlist(
                search_service=fake_search, qdrant_client=object(),
                collection_name="col", prompt="просто музыка", limit=5,
            )
        assert fake_search.calls[0]["mode"] == "audio"  # wish reused as CLAP query
        assert out["tracks"][0]["track_id"] == "au1"

    @pytest.mark.asyncio
    async def test_similar_tracks_resolves_seed_then_queries(self):
        plan = {"title": "X",
                "actions": [{"tool": "similar_tracks", "query": "Artist Song", "limit": 4}]}
        selection = {"picks": [{"n": 1, "reason": "похоже"}]}
        sim_result = {"tracks": [SimpleNamespace(
            track_id="sim1",
            payload={"title": "S", "artist": "B", "duration": 100.0,
                     "file_path": "/s.mp3"},
        )]}
        fake_search = FakeSearchService()
        with patch.object(svc, "ask_llm", new=AsyncMock(side_effect=[plan, selection])), \
             patch.object(svc.stream_service, "similar_tracks",
                          return_value=sim_result) as sim_mock:
            out = await ai_playlist(
                search_service=fake_search, qdrant_client=object(),
                collection_name="col", prompt="похожее на Artist Song", limit=5,
            )
        assert fake_search.calls[0]["mode"] == "text"   # seed resolution
        assert sim_mock.call_args.kwargs["seed_track_id"] == "seed"
        assert out["tracks"][0]["track_id"] == "sim1"


# ── Taste vibe (For-You hero phrase) ────────────────────────────────────────

def _vibe_island(rep, artists):
    return {
        "track_id": rep, "weight": 1.0,
        "tracks": [{"track_id": f"{rep}-{i}", "title": f"T{i}", "artist": a,
                    "cover_art_path": None} for i, a in enumerate(artists)],
    }


class FakeQdrant:
    """Minimal qdrant stub: retrieve(ids) → points carrying a payload dict
    ({"artist", "genre"}) per id."""
    def __init__(self, by_id):
        self._by_id = by_id

    def retrieve(self, collection_name, ids, with_payload=None, with_vectors=False):
        return [SimpleNamespace(id=tid, payload=self._by_id.get(tid) or {})
                for tid in ids if tid in self._by_id]


class TestVibeSourceHash:
    def test_stable_across_order(self):
        islands = [_vibe_island("x", ["A", "B"])]
        assert (svc._vibe_source_hash(islands, [{"artist": "Burial"}, {"artist": "Bonobo"}])
                == svc._vibe_source_hash(islands, [{"artist": "Bonobo"}, {"artist": "Burial"}]))

    def test_changes_with_rotation(self):
        islands = [_vibe_island("x", ["A"])]
        assert (svc._vibe_source_hash(islands, [{"artist": "Burial"}])
                != svc._vibe_source_hash(islands, [{"artist": "Aphex Twin"}]))


class TestDeterministicVibe:
    def test_blends_recent_and_long_term(self):
        prof = {"islands": [_vibe_island("x", ["Boards of Canada", "Aphex Twin"])]}
        out = svc.deterministic_taste_vibe(prof, [{"artist": "Burial"}], "ru")
        assert out["source"] == "fallback"
        assert "Burial" in out["phrase"] and "Boards of Canada" in out["phrase"]

    def test_two_artist_form_when_no_recent(self):
        prof = {"islands": [_vibe_island("x", ["Boards of Canada", "Aphex Twin"])]}
        out = svc.deterministic_taste_vibe(prof, [], "en")
        assert out["source"] == "fallback"
        assert "Boards of Canada" in out["phrase"] and "Aphex Twin" in out["phrase"]

    def test_cold_start_returns_none(self):
        out = svc.deterministic_taste_vibe({"islands": []}, [], "ru")
        assert out == {"phrase": None, "source": None}


class TestRecentRotation:
    def test_orders_by_recency_dedups_and_resolves_artists(self):
        # get_playback_signals is chronological (oldest→newest); recent_rotation
        # reverses it, so t3 is the most recent.
        signals = [{"track_id": "t1"}, {"track_id": "t2"}, {"track_id": "t3"}]
        reactions = [("t4", "like", "2026"), ("t5", "dislike", "2026")]
        by_id = {
            "t3": {"artist": "Burial", "genre": "dubstep"},
            "t2": {"artist": "Burial"},
            "t1": {"artist": "Aphex Twin", "genre": "idm"},
            "t4": {"artist": "Bonobo", "genre": "downtempo"},
        }
        with patch.object(svc.MetadataDB, "get_playback_signals", return_value=signals), \
             patch.object(svc.MetadataDB, "get_reactions_with_updated_at", return_value=reactions):
            out = svc.recent_rotation(FakeQdrant(by_id), "col")
        # Burial (t3) first, then Aphex Twin (t1), then liked Bonobo (t4); each
        # carries its genre (missing genre → None). t2's duplicate Burial is
        # dropped; the disliked t5 never enters.
        assert out == [
            {"artist": "Burial", "genre": "dubstep"},
            {"artist": "Aphex Twin", "genre": "idm"},
            {"artist": "Bonobo", "genre": "downtempo"},
        ]

    def test_empty_when_no_signals(self):
        with patch.object(svc.MetadataDB, "get_playback_signals", return_value=[]), \
             patch.object(svc.MetadataDB, "get_reactions_with_updated_at", return_value=[]):
            assert svc.recent_rotation(FakeQdrant({}), "col") == []


class TestTasteVibeCachedOrFallback:
    def test_cache_hit_returns_ai_phrase(self):
        islands = [_vibe_island("x", ["A", "B"])]
        profile = {"islands": islands}
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "recent_rotation", return_value=[{"artist": "Burial"}]):
            MetadataDB.set_recsys_llm_text(
                "col", svc.TASTE_VIBE_KIND, "ru",
                svc._vibe_source_hash(islands, [{"artist": "Burial"}]),
                {"phrase": "Полночный дрейф", "source": "ai"},
            )
            out = svc.taste_vibe_cached_or_fallback(
                qdrant_client=object(), collection_name="col", lang="ru")
        assert out == {"phrase": "Полночный дрейф", "source": "ai", "needs_generation": False}

    def test_cache_miss_returns_fallback_needs_generation(self):
        islands = [_vibe_island("x", ["Boards of Canada"])]
        profile = {"islands": islands}
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "recent_rotation", return_value=[]):
            out = svc.taste_vibe_cached_or_fallback(
                qdrant_client=object(), collection_name="col", lang="en")
        assert out["source"] == "fallback" and out["needs_generation"] is True
        assert "Boards of Canada" in out["phrase"]

    def test_cold_start_no_generation(self):
        with patch.object(svc.stream_service, "long_term_profile", return_value={"islands": []}), \
             patch.object(svc, "recent_rotation", return_value=[]):
            out = svc.taste_vibe_cached_or_fallback(
                qdrant_client=object(), collection_name="col", lang="ru")
        assert out == {"phrase": None, "source": None, "needs_generation": False}


class TestGenerateTasteVibe:
    @pytest.mark.asyncio
    async def test_generates_validates_and_caches(self):
        islands = [_vibe_island("x", ["A", "B"])]
        profile = {"islands": islands, "axes": None}
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "recent_rotation", return_value=[{"artist": "Burial"}]), \
             patch.object(svc, "ask_llm", new=AsyncMock(return_value='  "Полночный дрейф между эмбиентом и брейкбитом"  ')):
            out = await svc.generate_taste_vibe(qdrant_client=object(), collection_name="col", lang="ru")
        assert out["source"] == "ai"
        assert out["phrase"] == "Полночный дрейф между эмбиентом и брейкбитом"
        # persisted under the matching islands+rotation hash
        cached = svc.get_cached_taste_vibe("col", "ru", islands, [{"artist": "Burial"}])
        assert cached["phrase"] == out["phrase"]

    @pytest.mark.asyncio
    async def test_empty_profile_skips_llm(self):
        mock_llm = AsyncMock()
        with patch.object(svc.stream_service, "long_term_profile", return_value={"islands": []}), \
             patch.object(svc, "ask_llm", new=mock_llm):
            out = await svc.generate_taste_vibe(qdrant_client=object(), collection_name="col")
        assert out == {"phrase": None, "source": None}
        mock_llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_deterministic(self):
        islands = [_vibe_island("x", ["Boards of Canada"])]
        profile = {"islands": islands, "axes": None}
        with patch.object(svc.stream_service, "long_term_profile", return_value=profile), \
             patch.object(svc, "recent_rotation", return_value=[]), \
             patch.object(svc, "ask_llm", new=AsyncMock(side_effect=RuntimeError("llm down"))):
            out = await svc.generate_taste_vibe(qdrant_client=object(), collection_name="col", lang="en")
        assert out["source"] == "fallback"
        assert "Boards of Canada" in out["phrase"]


# ── Headline surfacing (from test_recsys_ai_enrich_headline.py) ──────────────

class TestAiEnrichHeadline:
    @pytest.mark.asyncio
    async def test_enrich_profile_surfaces_headline(self):
        fake_profile = {
            "islands": [
                {"track_id": "t1", "weight": 2.0,
                 "tracks": [{"track_id": "t1", "title": "A", "artist": "X", "cover_art_path": None}]},
            ],
            "axes": {"energy": {"z": 0.5, "level": "high"}},
            "n_signals": 10,
        }
        llm_json = {
            "portrait": "A concrete portrait.",
            "island_names": {"t1": "Ночной синтвейв"},
            "headline": "Поздний неон",
        }
        with patch.object(recsys_ai_service.stream_service, "long_term_profile", return_value=fake_profile), \
             patch.object(recsys_ai_service, "ask_llm", new=AsyncMock(return_value=llm_json)), \
             patch.object(recsys_ai_service.MetadataDB, "set_recsys_llm_text") as set_text:
            result = await recsys_ai_service.enrich_profile(
                qdrant_client=object(), collection_name="acct_test", lang="ru",
            )

        assert result["headline"] == "Поздний неон"
        assert result["portrait"] == "A concrete portrait."
        # headline must be persisted in the cached content blob
        stored_content = set_text.call_args.args[4]
        assert stored_content["headline"] == "Поздний неон"


# ── library_search routing (from test_recsys_library_search.py) ──────────────

class TestLibrarySearch:
    """The recsys 'library_search' tool must route to the catalog tracks engine
    (by name → songs), not the semantic hybrid search."""

    def test_library_search_uses_catalog_tracks(self, monkeypatch):
        captured = {}

        def fake_tracks(qdrant, collection, query, limit=20):
            captured["args"] = (collection, query, limit)
            return [{
                "track_id": "x1", "title": "Time", "artist": "Pink Floyd",
                "album": "DSOTM", "year": 1973, "genre": "rock", "duration": 412.0,
                "file_path": "/m/x1.mp3", "cover_art_path": None,
            }]

        monkeypatch.setattr(catalog_search_service, "search_catalog_tracks", fake_tracks)

        class _NoSemanticSearch:
            async def search(self, *a, **k):  # pragma: no cover - must not run
                raise AssertionError("library_search must not call semantic search")

        action = {"tool": "library_search", "query": "pink floyd", "limit": 15}
        out = asyncio.run(recsys_ai_service._execute_action(
            action,
            search_service=_NoSemanticSearch(),
            qdrant_client=object(),
            collection_name="acct_x",
        ))

        assert captured["args"] == ("acct_x", "pink floyd", 15)
        assert out and out[0]["track_id"] == "x1"
        assert out[0]["file_path"] == "/m/x1.mp3"
        assert out[0]["tool"] == "library_search"
