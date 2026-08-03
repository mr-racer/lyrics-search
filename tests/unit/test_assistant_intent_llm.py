"""The LLM half of the intent router.

What is pinned here is the CODE around the model: the closed label set, the
tolerance for the wrappings a small model puts around its JSON, the context line
it is given, and the rule that every failure mode — unreachable, slow,
unparseable, off-list — comes back as ``None`` so the router can fall back to
GLiNER instead of guessing.

The prompt's actual accuracy is a live-model question and lives in
``tests/docker/test_assistant_routing.py``.
"""
from __future__ import annotations

import asyncio

import pytest

from app.services.assistant import intent_llm as L


# ── the closed label set ─────────────────────────────────────────────────────


@pytest.mark.parametrize("label", ["search", "playlist", "facts", "followup", "unclear"])
def test_every_allowed_label_parses(label):
    assert L.parse_label('{"intent":"%s"}' % label) == label


@pytest.mark.parametrize("raw", [
    '{"intent":"recommendation"}',      # plausible, but not one of ours
    '{"intent":"SEARCH me a song"}',    # a sentence, not a label
    '{"label":"search"}',               # wrong key
    '{"intent":42}',
    '{"intent":null}',
    "search",                           # bare text, no object
    "",
    None,
    {"nope": 1},
])
def test_anything_off_list_is_a_failed_call(raw):
    """Not interpreted, not repaired — the router treats it exactly like an
    unreachable endpoint and asks GLiNER."""
    assert L.parse_label(raw) is None


def test_case_and_quoting_are_tolerated():
    assert L.parse_label('{"intent": "  Facts  "}') == "facts"


def test_the_object_is_dug_out_of_fences_and_prose():
    """Small models wrap the object inconsistently — same question, different
    wrapping between runs."""
    assert L.parse_label('```json\n{"intent":"playlist"}\n```') == "playlist"
    assert L.parse_label('Sure! {"intent":"playlist"} Hope that helps.') == "playlist"


def test_a_dict_passes_straight_through():
    assert L.parse_label({"intent": "search"}) == "search"


# ── the context line ─────────────────────────────────────────────────────────


def test_prompt_says_none_when_there_is_no_previous_turn():
    prompt = L._user_prompt("хиты Kanye", last_intent=None, last_message=None)
    assert "PREVIOUS: none" in prompt
    assert "MESSAGE: хиты Kanye" in prompt


def test_prompt_carries_the_previous_intent_and_message():
    prompt = L._user_prompt("а побыстрее?", last_intent="playlist",
                            last_message="хиты Kanye West")
    assert "PREVIOUS: playlist — «хиты Kanye West»" in prompt


def test_the_previous_turn_cannot_outgrow_the_message():
    """A long previous turn truncated so it stays context, not the subject."""
    prompt = L._user_prompt("ещё", last_intent="facts", last_message="о " * 500)
    assert len(prompt) < 2 * L.MAX_PREV_CHARS + 100


def test_a_long_message_is_capped():
    prompt = L._user_prompt("a" * 5000, last_intent=None, last_message=None)
    assert len(prompt) < L.MAX_MESSAGE_CHARS + 100


# ── the prompt itself ────────────────────────────────────────────────────────


def test_the_prompt_defines_every_label_it_allows():
    """A label the router accepts but the prompt never explains is a label the
    model will only ever emit by accident."""
    for label in L.LABELS:
        assert f'"{label}"' in L.SYSTEM_PROMPT


def test_the_prompt_shows_both_languages():
    """The app serves ru and en; a prompt exemplified only in English routes the
    other half of the traffic on hope."""
    assert "собери" in L.SYSTEM_PROMPT
    assert "playlist for" in L.SYSTEM_PROMPT or "tracks for" in L.SYSTEM_PROMPT


# ── every failure is the same failure ────────────────────────────────────────


async def test_no_llm_configured_means_no_call(monkeypatch):
    called = []
    monkeypatch.setattr(L, "available", lambda base_url=None: False)
    monkeypatch.setattr(L, "parse_label", lambda raw: called.append(raw))
    assert await L.classify("что-нибудь") is None
    assert called == []


async def test_empty_message_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(L, "available", lambda base_url=None: True)
    assert await L.classify("   ") is None


async def test_a_raising_endpoint_returns_none(monkeypatch):
    monkeypatch.setattr(L, "available", lambda base_url=None: True)

    async def _boom(*a, **kw):
        raise ConnectionError("LM Studio went away")

    monkeypatch.setattr("app.services.llm_client.ask_llm", _boom)
    assert await L.classify("собери хиты") is None


async def test_a_slow_endpoint_gives_up_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(L, "available", lambda base_url=None: True)
    monkeypatch.setattr(L, "TIMEOUT_SEC", 0.05)

    async def _slow(*a, **kw):
        await asyncio.sleep(5)
        return '{"intent":"search"}'

    monkeypatch.setattr("app.services.llm_client.ask_llm", _slow)
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await L.classify("собери хиты") is None
    assert loop.time() - started < 1.0


async def test_a_usable_label_comes_back(monkeypatch):
    monkeypatch.setattr(L, "available", lambda base_url=None: True)

    async def _ok(*a, **kw):
        return '{"intent":"playlist"}'

    monkeypatch.setattr("app.services.llm_client.ask_llm", _ok)
    assert await L.classify("собери хиты Канье") == "playlist"


def test_the_kill_switch_is_honoured(monkeypatch):
    monkeypatch.setattr(L, "ENABLED", False)
    assert L.available("http://localhost:1234/v1") is False
