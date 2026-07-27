"""Unit tests for the playlist reason gate (app/services/assistant/reason_gate.py)."""

from __future__ import annotations

import pytest

from app.services.assistant import reason_gate
from app.services.assistant.reason_gate import clean_reason, gate_playlist


def test_keeps_a_concrete_reason():
    text = "медленный бит и хор на припеве — как в запросе"
    assert clean_reason(text, title="Runaway", artist="Kanye West") == text


def test_keeps_english_concrete_reason():
    text = "sparse piano loop, the slow build the wish asked for"
    assert clean_reason(text, title="Runaway", artist="Kanye West") == text


@pytest.mark.parametrize("text", [
    "отличный трек",
    "Отличный трек для вечера",
    "подходит под настроение",
    "great track, fits the mood",
    "a great song for this",
])
def test_drops_filler(text):
    assert clean_reason(text, title="Runaway", artist="Kanye West") is None


def test_drops_too_short_and_too_long():
    assert clean_reason("бодрая", title="X", artist="Y") is None
    assert clean_reason("а" * 91, title="X", artist="Y") is None


def test_drops_restatement_of_title_or_artist():
    assert clean_reason("Kanye West — Runaway", title="Runaway",
                        artist="Kanye West") is None
    assert clean_reason("трек «Bohemian Rhapsody»", title="Bohemian Rhapsody",
                        artist="Queen") is None


def test_non_string_and_missing_reason():
    assert clean_reason(None, title="X", artist="Y") is None
    assert clean_reason(42, title="X", artist="Y") is None


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(reason_gate, "SHOW_REASONS", False)
    assert clean_reason("медленный бит и хор на припеве", title="Runaway",
                        artist="Kanye West") is None


def test_gate_playlist_rewrites_tracks_in_place():
    payload = {
        "title": "Вечер",
        "tracks": [
            {"title": "Runaway", "artist": "Kanye West", "reason": "отличный трек"},
            {"title": "Blue", "artist": "Mono", "reason": "тягучая гитара, растёт к финалу"},
            {"title": "Red", "artist": "Solo", "reason": None},
        ],
    }
    out = gate_playlist(payload)
    assert out["tracks"][0]["reason"] is None
    assert out["tracks"][1]["reason"] == "тягучая гитара, растёт к финалу"
    assert out["tracks"][2]["reason"] is None


def test_gate_playlist_tolerates_missing_tracks():
    assert gate_playlist({"title": "x"}) == {"title": "x"}
