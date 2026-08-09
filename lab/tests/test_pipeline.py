"""The parts of the pipeline that decide things: the iterate/stop veto, the
citation gate, and the numbered pack.

These are the rules that keep a small model from steering the run, so they are
tested directly rather than through a live call.
"""

import pytest

from lab.agent.config import AgentConfig
from lab.agent.llm import extract_json_object
from lab.agent.models import Evidence, ResolvedTrack, TrackRef
from lab.agent.pipeline import (GeneralBranch, PlaylistBranch, _fallback_answer,
                                _pack, _strip, _valid_citations)
from lab.agent.reasons import clean_reason
from lab.agent.retrieval.types import Fact


class _Stub:
    """Enough of an Assistant/sources/fetcher for the pure decision logic."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sink = None
        self.hub = None


def _branch(cls=GeneralBranch, **overrides):
    cfg = AgentConfig(**overrides)
    agent = _Stub(cfg)
    branch = cls.__new__(cls)
    branch.agent = agent
    branch.cfg = cfg
    branch.sink = None
    branch.sources = None
    branch.fetcher = None
    branch.chunks = []
    branch.retriever = None
    branch.used_queries = []
    return branch


class TestIterationVeto:
    """The model judges meaning; code judges budget — and overrules either way."""

    def test_a_confident_model_with_strong_context_stops(self):
        stop, why = _branch()._should_stop(
            iterations=1, sufficient=True, best_prob=0.8, new_chunks=5)
        assert stop is True
        assert "strong" in why

    def test_a_confident_model_with_weak_context_is_overruled(self):
        """The model cannot see the probabilities. It accepts almost any text
        as an answer; the cross-encoder says this text is not about the
        question."""
        stop, why = _branch()._should_stop(
            iterations=1, sufficient=True, best_prob=0.21, new_chunks=5)
        assert stop is False
        assert "0.21" in why

    def test_an_unsatisfied_model_out_of_budget_is_overruled(self):
        stop, why = _branch(general_max_iterations=2)._should_stop(
            iterations=2, sufficient=False, best_prob=0.9, new_chunks=5)
        assert stop is True
        assert "budget" in why

    def test_a_round_that_found_nothing_new_ends_the_run(self):
        """Otherwise a model that keeps asking for more spends the whole budget
        re-reading the same pages."""
        stop, why = _branch(general_max_iterations=5)._should_stop(
            iterations=2, sufficient=False, best_prob=0.9, new_chunks=0)
        assert stop is True
        assert "nothing new" in why

    def test_the_first_round_finding_nothing_new_still_continues(self):
        """On round one there is nothing to compare against — an empty first
        gather should be retried, not treated as exhaustion."""
        stop, _ = _branch()._should_stop(
            iterations=1, sufficient=False, best_prob=0.5, new_chunks=0)
        assert stop is False


class TestCitations:
    def test_valid_numbers_survive(self):
        assert _valid_citations([1, 3], 4) == [1, 3]

    def test_out_of_range_numbers_are_dropped(self):
        """A model citing [7] against a five-item pack is inventing a source."""
        assert _valid_citations([1, 7, 0, -2], 5) == [1]

    def test_duplicates_collapse(self):
        assert _valid_citations([2, 2, 2], 3) == [2]

    def test_strings_that_are_numbers_are_accepted(self):
        assert _valid_citations(["1", "2"], 2) == [1, 2]

    def test_garbage_yields_nothing(self):
        assert _valid_citations("all of them", 3) == []
        assert _valid_citations(None, 3) == []


class TestPack:
    def test_facts_come_first_and_numbering_is_continuous(self):
        facts = [Fact(row_id=1, kind="song", slug="s", text="fact one"),
                 Fact(row_id=2, kind="artist", slug="a", text="fact two")]
        from lab.agent.models import Chunk
        chunks = [(Chunk(id=0, path=["A"], body="chunk one", url="u1"), 0.7)]
        pack = _pack(facts, chunks)
        assert [e.n for e in pack] == [1, 2, 3]
        assert [e.kind for e in pack] == ["fact", "fact", "chunk"]
        assert pack[2].url == "u1"

    def test_an_empty_pack_is_empty(self):
        assert _pack([], []) == []

    def test_the_fallback_shows_the_sources_rather_than_nothing(self):
        pack = [Evidence(n=1, text="a fact", kind="fact")]
        assert "a fact" in _fallback_answer(pack, "ru")


class TestRelaxation:
    def test_the_style_word_comes_out_first(self):
        """It distorts a web search more than the era does: "спокойные хиты
        80х" returns listicles about calm music, not the decade."""
        from lab.agent.models import Filters, Plan

        plan = Plan(intent="playlist",
                    filters=Filters(era=(1980, 1989), style="calm"),
                    web_queries=[], ce_query="", allowed_tools=[])
        branch = _branch(PlaylistBranch)
        relaxed = branch._relax(plan, ["calm 1980s hits"], done=0)
        assert relaxed == ["1980s hits"]

    def test_the_era_comes_out_second(self):
        from lab.agent.models import Filters, Plan

        plan = Plan(intent="playlist",
                    filters=Filters(era=(1980, 1989), style="calm"),
                    web_queries=[], ce_query="", allowed_tools=[])
        relaxed = _branch(PlaylistBranch)._relax(plan, ["1980 1989 hits"], done=1)
        assert relaxed == ["hits"]

    def test_relaxing_stops_at_the_cap(self):
        from lab.agent.models import Filters, Plan

        plan = Plan(intent="playlist", filters=Filters(style="calm"),
                    web_queries=[], ce_query="", allowed_tools=[])
        branch = _branch(PlaylistBranch, max_relaxations=1)
        assert branch._relax(plan, ["calm hits"], done=1) is None

    def test_nothing_to_relax_returns_none(self):
        from lab.agent.models import Filters, Plan

        plan = Plan(intent="playlist", filters=Filters(),
                    web_queries=[], ce_query="", allowed_tools=[])
        assert _branch(PlaylistBranch)._relax(plan, ["hits"], done=0) is None

    def test_strip_is_case_insensitive_and_tidies_spacing(self):
        assert _strip("Calm 1980s Hits", "calm") == "1980s Hits"


class TestJsonExtraction:
    def test_a_bare_object_parses(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_a_fenced_object_parses(self):
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_an_object_buried_in_prose_parses(self):
        """What a local model actually returns about a third of the time."""
        text = 'Here is the plan:\n{"intent": "general"}\nHope that helps!'
        assert extract_json_object(text) == {"intent": "general"}

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        assert extract_json_object('x {"a": "} not the end"} y') == {
            "a": "} not the end"}

    def test_nested_objects_come_back_whole(self):
        got = extract_json_object('prefix {"a": {"b": 2}} suffix')
        assert got == {"a": {"b": 2}}

    def test_a_bare_list_is_not_an_object(self):
        assert extract_json_object("[1, 2, 3]") is None

    def test_nothing_parseable_returns_none(self):
        assert extract_json_object("no json here") is None
        assert extract_json_object("") is None


class TestReasonGate:
    def test_a_specific_reason_survives(self):
        assert clean_reason("Задаёт темп всей подборке в самом начале",
                            title="Power", artist="Kanye West")

    def test_filler_is_dropped(self):
        assert clean_reason("Отличный трек", title="Power",
                            artist="Kanye West") is None

    def test_a_restatement_of_the_row_is_dropped(self):
        """The card already prints "Kanye West — Power"."""
        assert clean_reason("Трек Power от Kanye West", title="Power",
                            artist="Kanye West") is None

    def test_an_overlong_reason_is_dropped(self):
        assert clean_reason("x" * 200, title="Power", artist="K") is None

    def test_a_non_string_is_dropped(self):
        assert clean_reason(None, title="a", artist="b") is None


class TestEraFilter:
    """Filtering is the point of extracting an era; a missing year is not
    evidence of the wrong decade."""

    def _tracks(self):
        from lab.agent.models import ResolvedTrack
        return [ResolvedTrack("a", "Old", "X", 1985, "exact"),
                ResolvedTrack("b", "New", "X", 2021, "exact"),
                ResolvedTrack("c", "Unknown", "X", None, "exact")]

    def test_out_of_range_tracks_go(self):
        from lab.agent.catalog import filter_by_era
        kept = filter_by_era(self._tracks(), (2020, 2029))
        assert {t.title for t in kept} == {"New", "Unknown"}

    def test_a_track_without_a_year_always_survives(self):
        from lab.agent.catalog import filter_by_era
        kept = filter_by_era(self._tracks(), (1900, 1901))
        assert [t.title for t in kept] == ["Unknown"]

    def test_no_era_means_no_filtering(self):
        from lab.agent.catalog import filter_by_era
        assert len(filter_by_era(self._tracks(), None)) == 3

    def test_it_works_on_claims_too(self):
        """Same rule, applied to page claims before anything reaches a model."""
        from lab.agent.catalog import filter_by_era
        refs = [TrackRef(title="A", year=1999), TrackRef(title="B", year=2005),
                TrackRef(title="C")]
        kept = filter_by_era(refs, (2000, 2009))
        assert {r.title for r in kept} == {"B", "C"}


class TestSelection:
    """The deterministic half, driven directly — this is what a notebook runs,
    so it has to behave identically to what the agent runs."""

    def _tracks(self):
        from lab.agent.models import ResolvedTrack
        return [ResolvedTrack("a", "Kids", "MGMT", 2007, "exact",
                              sources=["wikipedia"]),
                ResolvedTrack("a", "Kids", "MGMT", 2007, "exact",
                              sources=["web"]),
                ResolvedTrack("b", "Creep", "Radiohead", 1992, "fuzzy",
                              sources=["web"])]

    def test_the_same_track_from_two_sources_becomes_one_row(self):
        from lab.agent.selection import merge_claims
        merged = merge_claims(self._tracks(),
                              source_weights={"wikipedia": 2.0, "web": 1.0})
        assert len(merged) == 2

    def test_corroboration_adds_up(self):
        """Two independent pages naming a track is stronger than one page
        naming it twice — and that is what the weight expresses."""
        from lab.agent.selection import merge_claims
        merged = merge_claims(self._tracks(),
                              source_weights={"wikipedia": 2.0, "web": 1.0})
        kids = next(t for t in merged if t.title == "Kids")
        assert kids.weight == 3.0
        assert set(kids.sources) == {"wikipedia", "web"}

    def test_ranking_puts_corroborated_exact_matches_first(self):
        from lab.agent.selection import merge_claims, rank_tracks
        ranked = rank_tracks(merge_claims(
            self._tracks(), source_weights={"wikipedia": 2.0, "web": 1.0}))
        assert ranked[0].title == "Kids"

    def test_the_more_specific_provenance_wins_on_merge(self):
        from lab.agent.models import ResolvedTrack
        from lab.agent.selection import merge_claims
        bare = ResolvedTrack("a", "Kids", "MGMT", 2007, "exact", sources=["web"])
        rich = ResolvedTrack("a", "Kids", "MGMT", 2007, "exact",
                             sources=["wikipedia"], section="Soundtrack",
                             page_title="TDU2")
        merged = merge_claims([bare, rich], source_weights={})
        assert merged[0].section == "Soundtrack"
