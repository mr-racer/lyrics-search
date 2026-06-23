"""Unit tests for the search-chat history preamble (app/api/routes/chat.py).

The agentic search chat threads the prior conversation into the intent-
interpreting prompts (planner / scorer / answer) via `_history_preamble`, so
follow-ups ("а побыстрее?", "у того же артиста") resolve against the last turn.
"""
from __future__ import annotations

import pytest

from app.domain.models import ChatMessage


def test_history_preamble_empty_returns_blank():
    from app.api.routes.chat import _history_preamble
    assert _history_preamble([]) == ""
    assert _history_preamble(None) == ""


def test_history_preamble_all_blank_returns_blank():
    from app.api.routes.chat import _history_preamble
    assert _history_preamble([ChatMessage(role="user", content="   ")]) == ""


def test_history_preamble_formats_turns():
    from app.api.routes.chat import _history_preamble
    pre = _history_preamble([
        ChatMessage(role="user", content="find sad songs"),
        ChatMessage(role="assistant", content="here you go"),
    ])
    assert "PRIOR CONVERSATION" in pre
    assert "User: find sad songs" in pre
    assert "Assistant: here you go" in pre
    assert pre.endswith("\n\n")
