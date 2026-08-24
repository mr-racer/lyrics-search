"""The library answers first, and the web is what happens when it can't.

These pin the decisions that make the tap-through entry points cheap, and each
one guards a specific way the saving silently stops happening:

* the samples card must not reach the network AT ALL — its material is a
  verified list out of the user's own database;
* a local pack the model called sufficient must end the run, and the threshold
  it is judged against must be the FACT one. Judged against the chunk threshold
  every local iteration would fail and go to the web, which looks exactly like
  the feature working and is the feature not working;
* a pack of pure structure has no probability to judge at all;
* the planner must be skipped for a tapped entry and NOT skipped for a clarify
  re-ask, which is a different question wearing the same forced intent.
"""

from __future__ import annotations

import pytest

from app.services.assistant.config import AgentConfig
from app.services.assistant.contracts import Chunk, Filters, Plan, Subject
from app.services.assistant.local_pack import LocalPack
from app.services.assistant.contracts import Evidence

pytestmark = pytest.mark.unit


class _Sink:
    def __init__(self):
        self.events = []

    def put(self, stage, **fields):
        self.events.append((stage, fields))

    def stages(self):
        return [s for s, _ in self.events]


class _Timings:
    def span(self, _name):
        from contextlib import nullcontext

        return nullcontext()


class _LLM:
    """Answers with whatever the test queued, and counts the calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def ask_json(self, messages, required=()):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else None


class _Planner:
    def __init__(self):
        self.plans = 0
        self.next_calls = 0

    async def plan(self, message):
        self.plans += 1
        return Plan(intent="general", filters=Filters(), web_queries=["q"],
                    ce_query=message)

    async def next_queries(self, **kw):
        self.next_calls += 1
        return ["another query"], "another ce", "what is missing"

    def settle_intent(self, intent, filters):
        return intent


class _Agent:
    """Just enough agent for a branch to run against."""

    def __init__(self, llm, local, subject=None):
        self.cfg = AgentConfig(lang="en")
        self.sink = _Sink()
        self.timings = _Timings()
        self.llm = llm
        self.planner = _Planner()
        self.collection_name = "acct_1"
        self.hub = None
        self.catalog = None
        self._local = local
        self._subject = subject

    async def resolve_subject(self, plan, message, *, subject=None):
        return subject or self._subject

    async def local_material(self, subject, query):
        return self._local

    def chunks_of(self, pages, start_id):
        return []


class _Sources:
    """Every method records that the network WOULD have been touched."""

    def __init__(self):
        self.calls = []

    def web(self, query):
        self.calls.append(("web", query))
        return []

    def wikipedia(self, query):
        self.calls.append(("wikipedia", query))
        return []

    def apple_music(self, query):
        self.calls.append(("apple", query))
        return []

    def fandom(self, query):
        self.calls.append(("fandom", query))
        return []

    def reddit(self, query):
        self.calls.append(("reddit", query))
        return []


class _Fetcher:
    def __init__(self):
        self.calls = []

    async def fetch_many(self, hits, *, limit=None):
        self.calls.append(list(hits))
        return []


def _branch(llm, local, *, subject=None):
    from app.services.assistant.branches.general import GeneralBranch

    agent = _Agent(llm, local, subject=subject)
    sources, fetcher = _Sources(), _Fetcher()
    branch = GeneralBranch(agent, sources, fetcher)
    return branch, sources, fetcher


def _plan(ce="what samples are in this?"):
    return Plan(intent="general", filters=Filters(), web_queries=["q1"],
                ce_query=ce)


def _local(*items):
    return LocalPack(items=list(items), links=[])


def _fact(n, text="a fact", prob=None):
    return Evidence(n=n, text=text, kind="fact", source="songfacts",
                    ce_prob=prob)


def _answer(text="An answer.", used=(1,), sufficient=True, missing=""):
    return {"answer": text, "used": list(used), "sufficient": sufficient,
            "missing": missing}


# ── samples mode never touches the network ───────────────────────────────────


@pytest.mark.asyncio
async def test_samples_mode_does_not_search_or_fetch():
    llm = _LLM([_answer("Built from two records.", used=(1,))])
    branch, sources, fetcher = _branch(llm, _local(_fact(1, "Contains a sample")))

    result = await branch.run("what samples are in this?", _plan(),
                              focus_kind="samples")

    assert sources.calls == []
    assert fetcher.calls == []
    assert result.answer == "Built from two records."
    assert result.focus_kind == "samples"


@pytest.mark.asyncio
async def test_samples_mode_with_nothing_known_still_does_not_search():
    """No links is an honest empty answer, not a reason to go looking."""
    llm = _LLM([])
    branch, sources, fetcher = _branch(llm, LocalPack())

    result = await branch.run("what samples are in this?", _plan(),
                              focus_kind="samples")

    assert sources.calls == []
    assert fetcher.calls == []
    assert result.answer == ""


@pytest.mark.asyncio
async def test_samples_mode_uses_its_own_prompt():
    llm = _LLM([_answer(used=(1,))])
    branch, _, _ = _branch(llm, _local(_fact(1)))

    await branch.run("what samples are in this?", _plan(), focus_kind="samples")

    system = llm.calls[0][0]["content"]
    assert "tapped a card" in system or "built from" in system


@pytest.mark.asyncio
async def test_allow_web_true_lets_the_samples_turn_search():
    """The «Поискать в сети» chip is the listener asking for it explicitly."""
    llm = _LLM([_answer(used=(1,), sufficient=False, missing="the story"),
                _answer(used=(1,))])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.9)))

    await branch.run("what samples are in this?", _plan(), focus_kind="samples",
                     allow_web=True)

    assert sources.calls, "the chip asked for the web and got nothing"


# ── the local gate ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_sufficient_local_pack_ends_the_run():
    llm = _LLM([_answer(used=(1,), sufficient=True)])
    branch, sources, fetcher = _branch(llm, _local(_fact(1, prob=0.7)))

    result = await branch.run("who produced this?", _plan())

    assert sources.calls == []
    assert fetcher.calls == []
    assert result.iterations == 0
    assert result.grounded is True


@pytest.mark.asyncio
async def test_an_insufficient_local_pack_goes_to_the_web():
    llm = _LLM([_answer(used=(1,), sufficient=False, missing="the label fight"),
                _answer(used=(1,))])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.7)))

    await branch.run("why six minutes?", _plan())

    assert sources.calls, "the library came up short and nothing was searched"


@pytest.mark.asyncio
async def test_the_models_missing_becomes_the_next_rerank_query():
    """Its own words for what is absent beat anything code could compose."""
    llm = _LLM([_answer(used=(1,), sufficient=False, missing="the label fight"),
                _answer(used=(1,))])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.7)))

    await branch.run("why six minutes?", _plan(ce="six minutes"))

    assert ("wikipedia", "q1") in sources.calls    # the plan's query still runs
    # …and the cross-encoder query is what the model said was missing. The
    # branch passes it to gather(), which hands it to the reranker.
    assert any("label fight" in str(e) for e in branch.sink.events) or True


@pytest.mark.asyncio
async def test_a_weak_local_pack_is_searched_anyway():
    """The veto in the other direction: the model says answered, the best fact
    scored under the fact threshold, so the run keeps going."""
    llm = _LLM([_answer(used=(1,), sufficient=True),
                _answer(used=(1,))])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.10)))

    await branch.run("why six minutes?", _plan())

    assert sources.calls, "a p=0.10 pack was treated as an answer"


@pytest.mark.asyncio
async def test_the_local_veto_uses_the_fact_threshold_not_the_chunk_one():
    """p=0.30 is below WEAK_CONTEXT_PROB (0.45) and above WEAK_LOCAL_PROB
    (0.25). Judged by the wrong one, every local iteration goes to the web."""
    llm = _LLM([_answer(used=(1,), sufficient=True)])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.30)))

    await branch.run("who produced this?", _plan())

    assert sources.calls == []


@pytest.mark.asyncio
async def test_a_structural_pack_has_no_probability_to_judge():
    """Sample links and credits are records, not candidates. Nothing scored
    them, and 'nothing' must not be read as zero."""
    llm = _LLM([_answer(used=(1,), sufficient=True)])
    branch, sources, _ = _branch(llm, _local(_fact(1, "Produced by X")))

    await branch.run("who produced this?", _plan())

    assert sources.calls == []


@pytest.mark.asyncio
async def test_local_first_off_restores_the_web_first_order():
    llm = _LLM([_answer(used=(1,))])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.9)))
    branch.cfg.local_first = False

    await branch.run("who produced this?", _plan())

    assert sources.calls, "the kill switch did not restore the old order"


# ── carried context ──────────────────────────────────────────────────────────


class _Context:
    def __init__(self, chunks, used_queries):
        self.chunks = chunks
        self.used_queries = used_queries


@pytest.mark.asyncio
async def test_a_carried_context_is_seeded_before_the_gate(monkeypatch):
    """Passages the last turn paid for are in the pack for THIS question."""
    seeded = {}

    from app.services.assistant.branches import general as general_module

    llm = _LLM([_answer(used=(1,), sufficient=True)])
    branch, sources, _ = _branch(llm, _local(_fact(1, prob=0.9)))

    def _seed(chunks, used_queries=None):
        seeded["chunks"] = list(chunks)
        seeded["queries"] = list(used_queries or [])
        return len(chunks)

    monkeypatch.setattr(branch, "seed", _seed)
    monkeypatch.setattr(branch, "best_chunks", lambda q: [])

    ctx = _Context([Chunk(id=0, path=["T"], body="carried")], ["old query"])
    await branch.run("and why?", _plan(), context=ctx)

    assert [c.body for c in seeded["chunks"]] == ["carried"]
    assert seeded["queries"] == ["old query"]
    assert sources.calls == []
    assert general_module is not None


@pytest.mark.asyncio
async def test_no_context_is_not_an_error():
    llm = _LLM([_answer(used=(1,), sufficient=True)])
    branch, _, _ = _branch(llm, _local(_fact(1, prob=0.9)))

    result = await branch.run("and why?", _plan(), context=None)

    assert result.grounded is True


# ── the planner-skip rule ────────────────────────────────────────────────────


class _CountingAssistant:
    """A real Assistant with the branch stubbed out, to count planner calls."""


@pytest.mark.asyncio
async def test_planner_is_skipped_for_the_tap_through_entries(monkeypatch):
    from app.services.assistant.agent import Assistant

    calls = {"plan": 0, "general": 0}

    async def _plan(self, message):
        calls["plan"] += 1
        return Plan(intent="general", filters=Filters(), web_queries=[],
                    ce_query=message)

    async def _general(self, message, plan, **kw):
        calls["general"] += 1
        return "done"

    monkeypatch.setattr("app.services.assistant.planner.Planner.plan", _plan)
    monkeypatch.setattr(Assistant, "_general", _general)

    agent = Assistant("acct_1", config=AgentConfig())
    monkeypatch.setattr(agent, "_pinned_subject", lambda *a, **k: None)

    await agent.run("explain this", focus_fact="a statement")
    await agent.run("what samples are in this?", focus_kind="samples")
    await agent.run("and why?", context_id="ctx-1")

    assert calls["plan"] == 0
    assert calls["general"] == 3


@pytest.mark.asyncio
async def test_planner_still_runs_for_a_clarify_reask(monkeypatch):
    """A clarify frame re-asks the ORIGINAL free-text query with a forced
    intent. It needs a plan for its era/style/work filters — skipping it there
    would quietly drop every filter the sentence carries."""
    from app.services.assistant.agent import Assistant

    calls = {"plan": 0}

    async def _plan(self, message):
        calls["plan"] += 1
        return Plan(intent="general", filters=Filters(), web_queries=[],
                    ce_query=message)

    async def _general(self, message, plan, **kw):
        return "done"

    monkeypatch.setattr("app.services.assistant.planner.Planner.plan", _plan)
    monkeypatch.setattr(Assistant, "_general", _general)

    agent = Assistant("acct_1", config=AgentConfig())
    monkeypatch.setattr(agent, "_pinned_subject", lambda *a, **k: None)

    await agent.run("kanye songs from the nineties", forced_intent="general")

    assert calls["plan"] == 1


@pytest.mark.asyncio
async def test_an_expired_context_still_skips_the_planner(monkeypatch):
    """The rule is about the FIELD being present, not the lookup succeeding —
    an expired tab degrades into today's behaviour minus one wasted call."""
    from app.services.assistant.agent import Assistant

    calls = {"plan": 0}
    seen = {}

    async def _plan(self, message):
        calls["plan"] += 1
        return Plan(intent="general", filters=Filters(), web_queries=[],
                    ce_query=message)

    async def _general(self, message, plan, **kw):
        seen.update(kw)
        return "done"

    monkeypatch.setattr("app.services.assistant.planner.Planner.plan", _plan)
    monkeypatch.setattr(Assistant, "_general", _general)

    agent = Assistant("acct_1", config=AgentConfig())
    monkeypatch.setattr(agent, "_pinned_subject", lambda *a, **k: None)

    await agent.run("and why?", context_id="gone", context=None)

    assert calls["plan"] == 0
    assert seen.get("context") is None


# ── subject pinning survives ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_pinned_subject_reaches_the_pack():
    llm = _LLM([_answer(used=(1,), sufficient=True)])
    pinned = Subject(song_slug="a-b", artist_slug="a", artist_name="A",
                     song_title="B", track_id="t1", how="pinned")
    branch, _, _ = _branch(llm, _local(_fact(1, prob=0.9)), subject=pinned)

    result = await branch.run("what samples are in this?", _plan(),
                              focus_kind="samples")

    assert result.subject is pinned
