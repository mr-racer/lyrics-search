"""Playlist agent tools: cross-script artist scoring, the web-first gate on
get_songs, and the structured progress events the tools emit."""
import pytest

from app.services.playlist_agent.agent import _score_artists


# ─── _score_artists ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_score_artists_cross_script_cyrillic_query():
    # «канье» must find "Kanye West" even though the scripts differ.
    scored = _score_artists("канье", ["Drake", "Kanye West", "Queen"])
    assert scored, "cross-script query should produce candidates"
    assert scored[0]["artist"] == "Kanye West"
    assert scored[0]["score"] >= 0.5


@pytest.mark.unit
def test_score_artists_exact_latin():
    scored = _score_artists("queen", ["Queen", "Kanye West"])
    assert scored[0]["artist"] == "Queen"
    assert scored[0]["score"] == 1.0


@pytest.mark.unit
def test_score_artists_drops_garbage():
    assert _score_artists("абракадабра", ["Kanye West", "Drake"]) == []


@pytest.mark.unit
def test_score_artists_empty_values():
    assert _score_artists("канье", []) == []


# ─── agent tool flow (web-first gate + progress events) ───────────────────────


class FakeDeps:
    async def resolve_filter_values(self, key, raw):
        return ["Kanye West", "Drake"]


class FakeCatalog:
    def iter_songs(self):
        return [{"track_id": "1", "title": "Stronger", "artist": "Kanye West"}]

    def search_tracks_fuzzy(self, q, limit=3):
        return []


@pytest.mark.unit
async def test_get_songs_refuses_before_web_search_and_emits_progress(monkeypatch):
    """Scripted model: tries get_songs first (must be refused), then
    web_search, then get_songs again (must resolve). Verifies the refusal text
    reaches the model and that human-readable progress events are emitted."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from app.services.playlist_agent.agent import create_playlist_agent

    monkeypatch.setattr(
        "app.services.playlist_agent.agent.smart_web_search",
        lambda q, fetch, n: "Kanye West hits: Stronger",
    )

    calls = {"n": 0}

    def scripted(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:  # jump straight to get_songs — the gate must refuse
            return ModelResponse(parts=[ToolCallPart(
                "get_songs", {"items": [{"title": "Stronger", "artist": "Kanye West"}]})])
        if calls["n"] == 2:
            return ModelResponse(parts=[ToolCallPart(
                "web_search", {"query": "Kanye West greatest hits"})])
        if calls["n"] == 3:
            return ModelResponse(parts=[ToolCallPart(
                "get_songs", {"items": [{"title": "Stronger", "artist": "Kanye West"}]})])
        return ModelResponse(parts=[ToolCallPart("final_result", {
            "title": "Хиты Kanye West", "track_ids": ["1"], "comment": "", "missing": [],
        })])

    events = []
    state = {"web": 0, "resolved": {}, "missing": [],
             "on_status": events.append}
    agent = create_playlist_agent(FunctionModel(scripted), FakeDeps(), FakeCatalog(), state)
    result = await agent.run("[lang=ru] хиты канье")

    assert result.output.track_ids == ["1"]
    assert "1" in state["resolved"]

    # The premature get_songs call was refused with guidance, not executed.
    returns = [p for m in result.all_messages() for p in getattr(m, "parts", [])
               if isinstance(p, ToolReturnPart) and p.tool_name == "get_songs"]
    assert "refused" in str(returns[0].content)
    assert state["web"] == 1

    # Progress chain: the web search (with its query) and the matching pass.
    stages = [e["stage"] for e in events]
    assert stages == ["web_search", "matching", "matching_done"]
    assert events[0]["query"] == "Kanye West greatest hits"
    assert events[2]["found"] == 1 and events[2]["total"] == 1


@pytest.mark.unit
async def test_web_search_capped_at_six(monkeypatch):
    """The agent may keep digging when the library yield is poor, but only up
    to 6 web searches per run — the 7th is refused with the budget message."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from app.services.playlist_agent.agent import create_playlist_agent

    monkeypatch.setattr(
        "app.services.playlist_agent.agent.smart_web_search",
        lambda q, fetch, n: "some titles",
    )

    calls = {"n": 0}

    def scripted(messages, info):
        calls["n"] += 1
        if calls["n"] <= 7:
            return ModelResponse(parts=[ToolCallPart(
                "web_search", {"query": f"query {calls['n']}"})])
        return ModelResponse(parts=[ToolCallPart("final_result", {
            "title": "t", "track_ids": [], "comment": "", "missing": [],
        })])

    state = {"web": 0, "resolved": {}, "missing": [], "on_status": None}
    agent = create_playlist_agent(FunctionModel(scripted), FakeDeps(), FakeCatalog(), state)
    result = await agent.run("[lang=ru] хиты канье")

    assert state["web"] == 6
    returns = [p for m in result.all_messages() for p in getattr(m, "parts", [])
               if isinstance(p, ToolReturnPart) and p.tool_name == "web_search"]
    assert "limit reached" in str(returns[6].content)


@pytest.mark.unit
async def test_web_search_fetch_content_passthrough(monkeypatch):
    """fetch_content=true from the model reaches smart_web_search and trims
    max_results to 3 (full pages are ~4k chars each)."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from app.services.playlist_agent.agent import create_playlist_agent

    seen = []
    monkeypatch.setattr(
        "app.services.playlist_agent.agent.smart_web_search",
        lambda q, fetch, n: seen.append((q, fetch, n)) or "tracklist text",
    )

    calls = {"n": 0}

    def scripted(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(
                "web_search",
                {"query": "watch dogs soundtrack tracklist", "fetch_content": True})])
        return ModelResponse(parts=[ToolCallPart("final_result", {
            "title": "t", "track_ids": [], "comment": "", "missing": [],
        })])

    state = {"web": 0, "resolved": {}, "missing": [], "on_status": None}
    agent = create_playlist_agent(FunctionModel(scripted), FakeDeps(), FakeCatalog(), state)
    await agent.run("[lang=ru] саундтрек watch dogs")

    assert seen == [("watch dogs soundtrack tracklist", True, 3)]


@pytest.mark.unit
async def test_usage_limit_returns_partial_draft(monkeypatch):
    """A model that loops past the request budget must yield a partial draft
    built from state["resolved"] — not crash the whole stream route."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from app.services.playlist_agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "smart_web_search",
                        lambda q, fetch, n: "Kanye West hits: Stronger")

    calls = {"n": 0}

    def scripted(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart(
                "web_search", {"query": "kanye hits"})])
        if calls["n"] == 2:
            return ModelResponse(parts=[ToolCallPart(
                "get_songs", {"items": [{"title": "Stronger", "artist": "Kanye West"}]})])
        # Never emits final_result — keeps calling tools until the request
        # budget (_MAX_LLM_REQUESTS) is exhausted and UsageLimitExceeded fires.
        return ModelResponse(parts=[ToolCallPart(
            "web_search", {"query": f"junk {calls['n']}"})])

    monkeypatch.setattr(agent_mod, "_create_pydantic_model",
                        lambda base, name: FunctionModel(scripted))

    state = {}
    draft = await agent_mod.run_playlist_agent(
        "саундтрек watch dogs", "ru", FakeDeps(), FakeCatalog(), state)

    assert draft.track_ids == ["1"]
    assert "1" in state["resolved"]
    assert "оборвался" in draft.comment
