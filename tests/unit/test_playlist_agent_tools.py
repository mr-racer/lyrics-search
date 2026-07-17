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
