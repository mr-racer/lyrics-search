"""Finding a song in the library from its words.

Three properties are worth pinning, and each replaces something the previous
engine did differently:

* the cross-encoder reads WINDOWS of a lyric, not the lyric — truncation at 512
  tokens loses the second half of a song, and the line being asked about is as
  likely to be there as anywhere;
* the artist filter runs in code, because Qdrant's payload filter is an exact
  match and would drop everything tagged "Sade feat. …";
* the model naming no song ships an EMPTY hit list, so the retrieval tail is
  never presented as an answer.
"""

from __future__ import annotations

import pytest

from app.domain.models import TrackHit, TrackMetadata
from app.services.assistant.branches.lyrics import (LyricsBranch,
                                                    _match_best_hit,
                                                    _pick_matched_line,
                                                    _quoted_fragments, _windows)
from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Filters, Plan
from app.services.assistant.events import AgentSink
from app.services.assistant.timing import Timings

GRAVITY = ("I'm not here\nThis isn't happening\n"
           "Rain down, rain down\nCome on rain down on me\n"
           "That there, that's not me\nGravity always wins")


def _hit(track_id, title, artist, lyrics, year=2003):
    return TrackHit(
        track=TrackMetadata(track_id=track_id, title=title, artist=artist,
                            duration_sec=200.0, file_path=f"/m/{track_id}.mp3",
                            year=year, lyrics=lyrics),
        score=0.5, matched_on="lyrics", lyrics=lyrics.replace("\n", " "))


class _Service:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def search(self, query, *, mode="text", limit=10, **kw):
        self.calls.append({"query": query, "mode": mode, "limit": limit, **kw})
        return list(self.hits)


class _Hub:
    """Scores a window by how many of the query's words it contains."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.batches = []

    def ce_probabilities(self, query, docs):
        if not self.enabled:
            return None
        self.batches.append(list(docs))
        terms = {t for t in query.lower().split() if len(t) > 2}
        out = []
        for doc in docs:
            words = set(doc.lower().split())
            out.append(min(0.99, 0.1 + 0.3 * len(terms & words)))
        return out


class _LLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    async def ask_json(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return self.answer


class _Agent:
    def __init__(self, service, llm, hub=None, cfg=None):
        self.cfg = cfg or AgentConfig(lang="ru")
        self.sink = AgentSink()
        self.timings = Timings()
        self.search_service = service
        self.llm = llm
        self.hub = hub or _Hub()
        self.collection_name = "acct_1"

    def library_artist(self, raw):
        return raw


def _plan(**filters):
    return Plan(intent="lyrics_search", filters=Filters(**filters),
                web_queries=[], ce_query="The lyrics say that gravity always wins",
                lyrics_query="gravity always wins")


class TestWindows:
    def test_a_lyric_is_cut_into_overlapping_windows(self):
        windows = list(_windows(GRAVITY, 6, 3))
        assert len(windows) > 1
        assert all(len(w.split()) <= 6 for w in windows)

    def test_an_empty_lyric_produces_nothing(self):
        assert list(_windows("", 24, 12)) == []

    def test_a_long_lyric_is_bounded(self):
        """A page of repeated chorus does not become more informative by being
        scored forty more times, and the reranker batch is the expensive part."""
        assert len(list(_windows("la " * 5000, 8, 1))) <= 40


class TestReranking:
    async def test_the_matching_window_becomes_the_highlight(self):
        service = _Service([_hit("t1", "Street Spirit", "Radiohead", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "«Gravity always wins»",
                                      "song": "Street Spirit",
                                      "artist": "Radiohead",
                                      "confidence": "high"}))
        result = await LyricsBranch(agent).run("про гравитацию", _plan())
        assert result.song == "Street Spirit"
        assert "gravity" in (result.best_hit.matched_line or "").lower()

    async def test_windows_are_scored_not_whole_lyrics(self):
        """The window is smaller than the test lyric on purpose — a real song is
        far longer than the 512 tokens the cross-encoder reads."""
        hub = _Hub()
        cfg = AgentConfig(lyrics_window_words=6, lyrics_window_stride=3)
        service = _Service([_hit("t1", "Street Spirit", "Radiohead", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "x", "song": None}), hub=hub,
                       cfg=cfg)
        await LyricsBranch(agent).run("q", _plan())
        assert len(hub.batches[0]) > 1
        assert all(len(doc) < len(GRAVITY) for doc in hub.batches[0])

    async def test_without_a_cross_encoder_the_retrieval_order_stands(self):
        """Refusing to answer over a missing model would be worse than a worse
        order."""
        service = _Service([_hit("t1", "A", "X", GRAVITY),
                            _hit("t2", "B", "Y", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "m", "song": "A", "artist": "X"}),
                       hub=_Hub(enabled=False))
        result = await LyricsBranch(agent).run("q", _plan())
        assert result.song == "A"


class TestFilters:
    async def test_the_artist_filter_keeps_a_feat_credit(self):
        """The exact payload filter in Qdrant would drop this row; the catalog's
        own comparison does not."""
        service = _Service([_hit("t1", "Song", "Sade feat. Nas", GRAVITY),
                            _hit("t2", "Other", "Radiohead", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "m", "song": "Song",
                                      "artist": "Sade feat. Nas"}))
        result = await LyricsBranch(agent).run("q", _plan(artist="Sade"))
        assert result.song == "Song"
        assert [h.track.track_id for h in result.hits] == ["t1"]

    async def test_the_era_filter_drops_out_of_range_tracks(self):
        service = _Service([_hit("t1", "Old", "X", GRAVITY, year=1985),
                            _hit("t2", "New", "X", GRAVITY, year=2021)])
        agent = _Agent(service, _LLM({"message": "m", "song": "New",
                                      "artist": "X"}))
        result = await LyricsBranch(agent).run("q", _plan(era=(2020, 2029)))
        assert [h.track.track_id for h in result.hits] == ["t2"]

    async def test_a_track_with_no_year_survives_the_era_filter(self):
        """A missing year is not evidence of the wrong decade."""
        service = _Service([_hit("t1", "Undated", "X", GRAVITY, year=None)])
        agent = _Agent(service, _LLM({"message": "m", "song": "Undated",
                                      "artist": "X"}))
        result = await LyricsBranch(agent).run("q", _plan(era=(2020, 2029)))
        assert result.song == "Undated"

    async def test_nothing_left_after_filtering_is_an_honest_miss(self):
        service = _Service([_hit("t1", "Song", "Radiohead", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "m", "song": "Song"}))
        result = await LyricsBranch(agent).run("q", _plan(artist="Sade"))
        assert result.song is None
        assert result.hits == []


class TestTheAnswerGate:
    async def test_no_song_named_means_no_hits(self):
        """The retrieval tail is not an answer — presenting it as one is how
        near-misses get read as findings."""
        service = _Service([_hit("t1", "Song", "X", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "Не нашёл", "song": None}))
        result = await LyricsBranch(agent).run("q", _plan())
        assert result.song is None
        assert result.hits == []

    async def test_a_song_the_model_invented_finds_no_hit(self):
        service = _Service([_hit("t1", "Song", "X", GRAVITY)])
        agent = _Agent(service, _LLM({"message": "m", "song": "Not In The List"}))
        result = await LyricsBranch(agent).run("q", _plan())
        assert result.song is None

    async def test_the_context_is_capped(self):
        cfg = AgentConfig(lyrics_ctx_hits=3)
        service = _Service([_hit(f"t{i}", f"S{i}", "X", GRAVITY)
                            for i in range(10)])
        llm = _LLM({"message": "m", "song": None})
        agent = _Agent(service, llm, cfg=cfg)
        await LyricsBranch(agent).run("q", _plan())
        assert llm.prompts[0].count("\n1. ") <= 1
        assert "4. " not in llm.prompts[0]


class TestPortedHelpers:
    @pytest.mark.parametrize("song,expected", [
        ("Street Spirit", "t1"),
        ("street spirit", "t1"),
        ("Street Spirit (Remastered)", "t1"),
    ])
    def test_the_models_spelling_resolves_back_to_the_hit(self, song, expected):
        hits = [_hit("t1", "Street Spirit", "Radiohead", GRAVITY)]
        assert _match_best_hit(hits, song, "Radiohead").track.track_id == expected

    def test_the_artist_only_narrows_never_blocks(self):
        """Pass two exists because the model reformats the artist — feat.
        credits, "&" vs "and", a different romanisation."""
        hits = [_hit("t1", "Street Spirit", "Radiohead & Friends", GRAVITY)]
        assert _match_best_hit(hits, "Street Spirit", "Radiohead") is not None

    def test_nothing_named_resolves_to_nothing(self):
        assert _match_best_hit([_hit("t1", "A", "X", GRAVITY)], None, None) is None

    def test_only_real_quotes_of_three_words_count(self):
        message = 'Это «gravity always wins» из песни «Street Spirit», «нет»'
        assert _quoted_fragments(message) == ["gravity always wins"]

    def test_the_highlight_lands_on_the_quoted_line(self):
        assert _pick_matched_line(GRAVITY, "gravity always wins") == \
            "Gravity always wins"

    def test_no_overlap_means_no_highlight(self):
        assert _pick_matched_line(GRAVITY, "completely unrelated words") is None
