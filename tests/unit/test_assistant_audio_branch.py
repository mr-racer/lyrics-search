"""Finding tracks by how they sound.

The branch is arithmetic on top of two model calls, and neither is a judgement:
the planner already decided the intent and the filters, and this rewrites the
user's description of sound into CLAP's dialect. What is worth pinning is
everything AFTER that — four searches, dedupe, reciprocal rank, the artist
filter, and the fact that no third model call writes the caption.
"""

from __future__ import annotations

from app.domain.models import TrackHit, TrackMetadata
from app.services.assistant.branches.audio import AudioBranch, _rrf
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Filters, Plan
from app.services.assistant.events import AgentSink
from app.services.assistant.timing import Timings


def _hit(track_id, title, artist, year=2005, score=0.5):
    return TrackHit(
        track=TrackMetadata(track_id=track_id, title=title, artist=artist,
                            duration_sec=200.0, file_path=f"/m/{track_id}.mp3",
                            year=year),
        score=score, matched_on="audio")


class _Service:
    """Returns a different ordering per query, so RRF has something to fuse."""

    def __init__(self, per_query):
        self.per_query = per_query
        self.calls = []

    async def search(self, query, *, mode="text", limit=10, **kw):
        self.calls.append({"query": query, "mode": mode, "limit": limit, **kw})
        return list(self.per_query.get(query, self.per_query.get("*", [])))


class _LLM:
    def __init__(self, queries):
        self.queries = queries
        self.calls = 0

    async def ask_list(self, messages, **kw):
        self.calls += 1
        return self.queries


class _Agent:
    def __init__(self, service, llm, cfg=None):
        self.cfg = cfg or AgentConfig(lang="ru")
        self.sink = AgentSink()
        self.timings = Timings()
        self.search_service = service
        self.llm = llm
        self.collection_name = "acct_1"

    def library_artist(self, raw):
        # The playlist branch's resolution, stubbed: «Сейд» → the library spelling.
        return {"Сейд": "Sade"}.get(raw, raw)


def _plan(style="спокойные", artist=None, era=None):
    return Plan(intent="audio_search",
                filters=Filters(style=style, artist=artist, era=era),
                web_queries=[], ce_query="")


PROMPTS = ["This song is a slow soul track with soft dynamics",
           "This song is a warm timbre soul track with sparse texture",
           "This song is a mellow soul song with analogue production",
           "This song is a soul song with an intimate close vocal"]


class TestRephrasing:
    async def test_four_prompts_reach_four_searches(self):
        service = _Service({"*": [_hit("t1", "A", "Sade")]})
        agent = _Agent(service, _LLM(PROMPTS))
        result = await AudioBranch(agent).run("спокойные песни", _plan())
        assert len(service.calls) == 4
        assert {c["mode"] for c in service.calls} == {"audio"}
        assert result.queries == PROMPTS

    async def test_the_artist_never_enters_the_prompt(self):
        """Inside a CLAP prompt a name drags the vector towards that artist's
        most typical track; as a filter it is exact and free."""
        service = _Service({"*": [_hit("t1", "A", "Sade")]})
        agent = _Agent(service, _LLM(PROMPTS))
        await AudioBranch(agent).run("спокойные песни Сейд", _plan(artist="Сейд"))
        assert all("Sade" not in c["query"] and "Сейд" not in c["query"]
                   for c in service.calls)

    async def test_a_failed_rephrasing_still_searches(self):
        """One prompt is worse than four; no answer is worse than one."""
        service = _Service({"*": [_hit("t1", "A", "Sade")]})
        agent = _Agent(service, _LLM(None))
        result = await AudioBranch(agent).run("спокойные", _plan())
        assert len(service.calls) == 1
        assert result.tracks

    async def test_the_count_is_configurable(self):
        service = _Service({"*": [_hit("t1", "A", "Sade")]})
        agent = _Agent(service, _LLM(PROMPTS), cfg=AgentConfig(clap_queries=2))
        await AudioBranch(agent).run("спокойные", _plan())
        assert len(service.calls) == 2


class TestFusion:
    async def test_a_track_every_prompt_liked_wins(self):
        """RRF over positions, not raw CLAP scores: four prompts are four points
        in the text space and their score scales are not comparable."""
        agreed, lucky = _hit("t1", "Agreed", "X"), _hit("t2", "Lucky", "X")
        per_query = {PROMPTS[0]: [lucky, agreed],
                     PROMPTS[1]: [agreed, lucky],
                     PROMPTS[2]: [agreed, lucky],
                     PROMPTS[3]: [agreed, lucky]}
        agent = _Agent(_Service(per_query), _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan())
        assert [h.track.track_id for h in result.tracks] == ["t1", "t2"]

    async def test_a_track_is_returned_once(self):
        service = _Service({"*": [_hit("t1", "A", "X"), _hit("t1", "A", "X")]})
        agent = _Agent(service, _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan())
        assert [h.track.track_id for h in result.tracks] == ["t1"]

    async def test_the_result_is_capped(self):
        hits = [_hit(f"t{i}", f"S{i}", "X") for i in range(20)]
        agent = _Agent(_Service({"*": hits}), _LLM(PROMPTS),
                       cfg=AgentConfig(clap_result_count=5))
        result = await AudioBranch(agent).run("q", _plan())
        assert len(result.tracks) == 5

    def test_rrf_rewards_agreement_over_a_single_first_place(self):
        merged = dict(_rrf([["a", "b"], ["b", "a"], ["b", "a"]], k=60))
        assert merged["b"] > merged["a"]


class TestFilters:
    async def test_the_artist_filter_keeps_a_feat_credit(self):
        service = _Service({"*": [_hit("t1", "A", "Sade feat. Nas"),
                                  _hit("t2", "B", "Radiohead")]})
        agent = _Agent(service, _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan(artist="Сейд"))
        assert [h.track.track_id for h in result.tracks] == ["t1"]

    async def test_the_era_filter_applies(self):
        service = _Service({"*": [_hit("t1", "Old", "X", year=1985),
                                  _hit("t2", "New", "X", year=2021)]})
        agent = _Agent(service, _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan(era=(2020, 2029)))
        assert [h.track.track_id for h in result.tracks] == ["t2"]

    async def test_a_filtered_run_asks_for_a_deeper_pool(self):
        """The filter runs after the search, so a pool cut to size first would
        come back empty the moment a filter is set."""
        service = _Service({"*": [_hit("t1", "A", "Sade")]})
        agent = _Agent(service, _LLM(PROMPTS))
        await AudioBranch(agent).run("q", _plan(artist="Sade"))
        plain = _Service({"*": [_hit("t1", "A", "Sade")]})
        await AudioBranch(_Agent(plain, _LLM(PROMPTS))).run("q", _plan())
        assert service.calls[0]["limit"] > plain.calls[0]["limit"]


class TestNoJudgement:
    async def test_the_caption_costs_no_model_call(self):
        """One call for the rephrasing and nothing else — there is nothing here
        for a model to validate, and a round trip for one sentence costs
        seconds."""
        llm = _LLM(PROMPTS)
        agent = _Agent(_Service({"*": [_hit("t1", "A", "X")]}), llm)
        result = await AudioBranch(agent).run("спокойные песни", _plan())
        assert llm.calls == 1
        assert result.comment

    async def test_tracks_carry_no_invented_reason(self):
        agent = _Agent(_Service({"*": [_hit("t1", "A", "X")]}), _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan())
        assert all(getattr(h, "reason", None) is None for h in result.tracks)

    async def test_nothing_found_says_so(self):
        agent = _Agent(_Service({"*": []}), _LLM(PROMPTS))
        result = await AudioBranch(agent).run("q", _plan())
        assert result.tracks == []
        assert result.comment
