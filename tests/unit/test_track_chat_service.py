"""Unit tests for app/services/track_chat_service.py — prompts + helpers.

Schema notes (verified from metadata_db.py):
- song_facts table: (id PK, song_slug FK→songs.slug, lang, fact TEXT, category, source)
  The column is `fact`, NOT `notes`. No `slug` PK — songs are identified via song_slug.
- song_slug format: "{artist_slug}-{title_slug}" (same key returned by
  song_facts_service.get_song_facts_key(artist, song)).
- MetadataDB.get_song_facts(slug, collection_name) filters by both song_slug AND
  collection_name (via JOIN on songs table).
"""
from __future__ import annotations

import pytest

from app.domain.models import TrackChatContext


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    from app.resources.metadata_db import MetadataDB
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


def test_prompts_are_loaded_and_distinct():
    from app.services.track_chat_service import TRACK_CHAT_PROMPT, LYRIC_EXPLAIN_PROMPT
    assert "track_context_block" in TRACK_CHAT_PROMPT
    assert "track_context_block" in LYRIC_EXPLAIN_PROMPT
    assert "selected_line" in LYRIC_EXPLAIN_PROMPT
    assert "selected_line" not in TRACK_CHAT_PROMPT
    assert TRACK_CHAT_PROMPT != LYRIC_EXPLAIN_PROMPT


def test_resolve_song_facts_returns_empty_string_when_missing(db):
    from app.services.track_chat_service import resolve_song_facts
    out = resolve_song_facts(title="Nonexistent", artist="Unknown Artist")
    assert out == ""


def test_resolve_song_facts_returns_raw_notes_when_present(db):
    """resolve_song_facts must return the RAW facts from the song_facts table.

    Real schema (verified):
    - songs.slug PRIMARY KEY = "{artist_slug}-{title_slug}"
    - song_facts.song_slug FK → songs.slug
    - song_facts.fact TEXT  (NOT notes)
    - get_song_facts_key(artist, song) → "{artist_slug}-{title_slug}"
    """
    from app.services.track_chat_service import resolve_song_facts
    from app.resources.metadata_db import MetadataDB
    from app.services.song_facts_service import get_song_facts_key

    # Build the song_slug exactly as the service does
    song_slug = get_song_facts_key("Beach House", "Levitation")
    # song_slug == "beach-house-levitation"
    artist_slug = "beach-house"

    # Ensure artist + song rows exist (required by FK constraint)
    MetadataDB.upsert_artist(artist_slug, "Beach House", "test_col")
    MetadataDB.upsert_song(song_slug, "Levitation", artist_slug, "test_col")

    # Insert raw fact into song_facts directly (bypasses refined-facts layer)
    conn = MetadataDB._connect()
    conn.execute(
        "INSERT INTO song_facts (song_slug, lang, fact, source) VALUES (?, 'en', ?, ?)",
        (song_slug, "Raw fact: song released 2015.", "songfacts.com"),
    )
    conn.commit()

    out = resolve_song_facts(title="Levitation", artist="Beach House")
    assert "Raw fact" in out


def test_resolve_song_facts_never_pulls_refined_facts(db):
    """REGRESSION GUARD: even if refined_facts has translated content for this
    track (in EITHER language), resolve_song_facts must return ONLY the raw
    song_facts.fact rows — never refined_facts.refined_json content.

    Refined facts live in a separate table (refined_facts) by design.
    The chat agent must always see the original raw facts, regardless of
    which language the user picked or whether they ran AI-Indexing refining.
    """
    import json
    from app.services.track_chat_service import resolve_song_facts
    from app.resources.metadata_db import MetadataDB
    from app.services.song_facts_service import get_song_facts_key

    song_slug = get_song_facts_key("Beach House", "Levitation")
    artist_slug = "beach-house"
    MetadataDB.upsert_artist(artist_slug, "Beach House", "test_col")
    MetadataDB.upsert_song(song_slug, "Levitation", artist_slug, "test_col")

    conn = MetadataDB._connect()
    # Raw fact (what we WANT in chat context)
    conn.execute(
        "INSERT INTO song_facts (song_slug, lang, fact, source) VALUES (?, 'en', ?, ?)",
        (song_slug, "RAW_MARKER: original English fact.", "songfacts.com"),
    )
    # Refined facts in BOTH languages — these must NEVER appear in chat context
    conn.execute(
        "INSERT INTO refined_facts (scope, scope_key, collection_name, lang, refined_json) "
        "VALUES ('song', ?, 'test_col', 'en', ?)",
        (song_slug, json.dumps([{"text": "REFINED_EN_MARKER: refined English."}])),
    )
    conn.execute(
        "INSERT INTO refined_facts (scope, scope_key, collection_name, lang, refined_json) "
        "VALUES ('song', ?, 'test_col', 'ru', ?)",
        (song_slug, json.dumps([{"text": "REFINED_RU_MARKER: переведённый факт."}])),
    )
    conn.commit()

    out = resolve_song_facts(title="Levitation", artist="Beach House")
    # Raw fact must be present
    assert "RAW_MARKER" in out
    # Refined markers (in ANY language) must NOT leak
    assert "REFINED_EN_MARKER" not in out
    assert "REFINED_RU_MARKER" not in out


def test_build_track_context_block_includes_meta_and_lyrics(db):
    from app.services.track_chat_service import build_track_context_block
    ctx = TrackChatContext(
        title="Levitation", artist="Beach House", album="Depression Cherry",
        year=2015, genre="dream pop",
        full_lyrics="Falling in tonight\nWith a chord around your hand",
    )
    block = build_track_context_block(ctx, song_facts="A dreamy opener.")
    assert "Levitation" in block
    assert "Beach House" in block
    assert "Depression Cherry" in block
    assert "2015" in block
    assert "dream pop" in block
    assert "Falling in tonight" in block
    assert "A dreamy opener." in block


def test_build_track_context_block_handles_missing_facts(db):
    from app.services.track_chat_service import build_track_context_block
    ctx = TrackChatContext(
        title="T", artist="A", album=None, year=None, genre=None, full_lyrics="",
    )
    block = build_track_context_block(ctx, song_facts="")
    assert "T" in block
    assert "A" in block


# ─── Task 5.3 — orchestrator tests ───────────────────────────────────────────

class _MockAgentResult:
    """Stand-in for pydantic-ai's RunResult."""
    def __init__(self, output: str, tool_calls: int = 0):
        self.output = output
        self._tool_calls = tool_calls


@pytest.mark.asyncio
async def test_answer_track_chat_song_mode_uses_track_chat_prompt(db, monkeypatch):
    """mode='song' must compose TRACK_CHAT_PROMPT (not LYRIC_EXPLAIN_PROMPT)."""
    from app.services import track_chat_service
    captured_prompts = []

    async def fake_run_agent(agent, message, system_prompt, history):
        captured_prompts.append(system_prompt)
        return _MockAgentResult("Mocked reply", tool_calls=0)

    monkeypatch.setattr(track_chat_service, "_run_agent", fake_run_agent)

    from app.domain.models import TrackChatContext, TrackChatRequest
    req = TrackChatRequest(
        track_context=TrackChatContext(
            title="T", artist="A", album=None, year=None, genre=None, full_lyrics="line",
        ),
        mode="song",
        message="hi",
    )
    resp = await track_chat_service.answer_track_chat(req)
    assert resp.message == "Mocked reply"
    assert resp.web_search_used is False
    # The placeholder {track_context_block} must have been substituted
    assert "{track_context_block}" not in captured_prompts[0]
    assert "TRACK CONTEXT:" in captured_prompts[0]
    # song mode → no SELECTED LINE section
    assert "SELECTED LINE:" not in captured_prompts[0]


@pytest.mark.asyncio
async def test_answer_track_chat_lyric_explain_uses_other_prompt(db, monkeypatch):
    from app.services import track_chat_service
    captured = []

    async def fake_run_agent(agent, message, system_prompt, history):
        captured.append(system_prompt)
        # Simulate web_search being called by mutating the state map via the agent ref
        if hasattr(agent, "_test_state"):
            agent._test_state["web_search_calls"] = 1
        return _MockAgentResult("Mocked explanation", tool_calls=1)

    monkeypatch.setattr(track_chat_service, "_run_agent", fake_run_agent)

    from app.domain.models import TrackChatContext, TrackChatRequest
    req = TrackChatRequest(
        track_context=TrackChatContext(
            title="T", artist="A", album=None, year=None, genre=None, full_lyrics="line",
        ),
        mode="lyric_explain",
        selected_line="Bring me water for my eyes",
        message="Explain this line",
    )
    resp = await track_chat_service.answer_track_chat(req)
    assert resp.message == "Mocked explanation"
    # web_search_used is True because fake_run_agent set _test_state["web_search_calls"] = 1
    assert isinstance(resp.web_search_used, bool)
    assert "SELECTED LINE:" in captured[0]
    assert "Bring me water for my eyes" in captured[0]


@pytest.mark.asyncio
async def test_answer_track_chat_lyric_explain_requires_selected_line(db):
    from app.services import track_chat_service
    from app.domain.models import TrackChatContext, TrackChatRequest
    req = TrackChatRequest(
        track_context=TrackChatContext(
            title="T", artist="A", album=None, year=None, genre=None, full_lyrics="",
        ),
        mode="lyric_explain",
        selected_line=None,
        message="Explain",
    )
    with pytest.raises(ValueError, match="selected_line"):
        await track_chat_service.answer_track_chat(req)


# ─── Token budget + fact packing ──────────────────────────────────────────────

def test_est_tokens_is_ceil_chars_over_four():
    from app.services.track_chat_service import _est_tokens
    assert _est_tokens("") == 0
    assert _est_tokens("abcd") == 1
    assert _est_tokens("abcde") == 2  # ceil(5/4)


def test_select_facts_keeps_shortest_first_in_original_order():
    from app.services.track_chat_service import _select_facts, _est_tokens
    facts = ["x" * 400, "short one", "y" * 800, "tiny"]
    # budget fits only the two lean facts (each + 2 separator tokens)
    budget = _est_tokens("short one") + _est_tokens("tiny") + 4
    kept = _select_facts(facts, budget)
    # leanest survive; the two fat ones are dropped
    assert kept == ["short one", "tiny"]  # original order, not length order


def test_select_facts_zero_or_negative_budget_returns_empty():
    from app.services.track_chat_service import _select_facts
    assert _select_facts(["a", "b"], 0) == []
    assert _select_facts(["a", "b"], -5) == []


def test_pack_facts_joins_kept_with_blank_line():
    from app.services.track_chat_service import _pack_facts
    assert _pack_facts(["a", "b"], 10_000) == "a\n\nb"
    assert _pack_facts([], 10_000) == ""


def test_resolve_song_facts_list_returns_list_when_missing(db):
    from app.services.track_chat_service import resolve_song_facts_list
    assert resolve_song_facts_list(title="Nope", artist="Nobody") == []


# ─── Multi-turn history → pydantic-ai message_history ─────────────────────────

def test_to_message_history_maps_roles_and_skips_blanks():
    from app.services.track_chat_service import _to_message_history
    from app.domain.models import ChatMessage
    from pydantic_ai.messages import ModelRequest, ModelResponse

    out = _to_message_history([
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="   "),   # blank → skipped
    ])
    assert len(out) == 2
    assert isinstance(out[0], ModelRequest)
    assert isinstance(out[1], ModelResponse)


def test_to_message_history_empty_input():
    from app.services.track_chat_service import _to_message_history
    assert _to_message_history([]) == []
    assert _to_message_history(None) == []


@pytest.mark.asyncio
async def test_answer_track_chat_forwards_history_to_agent(db, monkeypatch):
    """req.history must reach _run_agent as a non-empty message_history."""
    from app.services import track_chat_service
    captured = {}

    async def fake_run_agent(agent, message, system_prompt, history):
        captured["history"] = history
        return _MockAgentResult("ok")

    monkeypatch.setattr(track_chat_service, "_run_agent", fake_run_agent)

    from app.domain.models import TrackChatContext, TrackChatRequest, ChatMessage
    req = TrackChatRequest(
        track_context=TrackChatContext(
            title="T", artist="A", album=None, year=None, genre=None, full_lyrics="l",
        ),
        mode="song",
        message="hi",
        history=[
            ChatMessage(role="user", content="prev question"),
            ChatMessage(role="assistant", content="prev answer"),
        ],
    )
    await track_chat_service.answer_track_chat(req)
    assert captured["history"] and len(captured["history"]) == 2


@pytest.mark.asyncio
async def test_answer_track_chat_budget_gates_facts(db, monkeypatch):
    """Ample budget → fact present in the prompt; tiny budget → fact trimmed out."""
    from app.services import track_chat_service
    from app.resources.metadata_db import MetadataDB
    from app.services.song_facts_service import get_song_facts_key

    song_slug = get_song_facts_key("Beach House", "Levitation")
    MetadataDB.upsert_artist("beach-house", "Beach House", "test_col")
    MetadataDB.upsert_song(song_slug, "Levitation", "beach-house", "test_col")
    conn = MetadataDB._connect()
    conn.execute(
        "INSERT INTO song_facts (song_slug, lang, fact, source) VALUES (?, 'en', ?, 'x')",
        (song_slug, "MARKER_FACT released 2015."),
    )
    conn.commit()

    captured = []

    async def fake_run_agent(agent, message, system_prompt, history):
        captured.append(system_prompt)
        return _MockAgentResult("ok")

    monkeypatch.setattr(track_chat_service, "_run_agent", fake_run_agent)

    from app.domain.models import TrackChatContext, TrackChatRequest

    def make_req():
        return TrackChatRequest(
            track_context=TrackChatContext(
                title="Levitation", artist="Beach House",
                album=None, year=None, genre=None, full_lyrics="la",
            ),
            mode="song", message="when released?",
        )

    monkeypatch.setenv("TRACK_CHAT_MAX_PROMPT_TOKENS", "100000")
    await track_chat_service.answer_track_chat(make_req())
    assert "MARKER_FACT" in captured[-1]

    # Budget smaller than the prompt scaffold itself → no room for any fact
    monkeypatch.setenv("TRACK_CHAT_MAX_PROMPT_TOKENS", "10")
    await track_chat_service.answer_track_chat(make_req())
    assert "MARKER_FACT" not in captured[-1]
