"""Unit tests for recsys_ai_service: enrichment cache + plan/execute/select."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import recsys_ai_service as svc
from app.services.recsys_ai_service import (
    _validate_plan,
    ai_playlist,
    enrich_profile,
    get_cached_enrichment,
    profile_source_hash,
)


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
        with patch.object(svc, "ask_llm", new=AsyncMock(side_effect=[plan, selection])):
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
