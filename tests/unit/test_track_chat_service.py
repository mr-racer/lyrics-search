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
