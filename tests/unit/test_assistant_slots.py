"""Slot carry-forward — the mechanism that makes multi-turn work without the
LLM ever deciding "is this a follow-up?".

The rule under test is deliberately unconditional: slots always survive, fresh
values overwrite, and None never wipes a good slot.
"""
from __future__ import annotations

from app.domain.models import AssistantRoute, AssistantSlots
from app.services.assistant.router import merge_slots


def test_fresh_entities_overwrite():
    slots = AssistantSlots(last_artist="Queen", last_song="Bohemian Rhapsody")
    route = AssistantRoute(intent="facts", artist="Kanye West", song="Runaway")
    merged = merge_slots(slots, route)
    assert merged.last_artist == "Kanye West"
    assert merged.last_song == "Runaway"
    assert merged.last_intent == "facts"


def test_missing_entities_keep_the_previous_ones():
    """«ещё у этого артиста» extracts no artist — the old one must survive,
    that is the whole follow-up mechanism."""
    slots = AssistantSlots(last_artist="Queen", last_intent="search")
    route = AssistantRoute(intent="playlist", artist=None, song=None)
    merged = merge_slots(slots, route)
    assert merged.last_artist == "Queen"
    assert merged.last_intent == "playlist"


def test_none_updates_never_wipe_a_slot():
    slots = AssistantSlots(last_track_id="t-1")
    merged = merge_slots(slots, None, last_track_id=None)
    assert merged.last_track_id == "t-1"


def test_explicit_updates_win():
    slots = AssistantSlots(last_track_id="t-1")
    merged = merge_slots(slots, None, last_track_id="t-2")
    assert merged.last_track_id == "t-2"


def test_playlist_ids_are_recorded():
    merged = merge_slots(AssistantSlots(), None, last_playlist_ids=["a", "b"])
    assert merged.last_playlist_ids == ["a", "b"]


def test_none_slots_start_from_a_blank_set():
    merged = merge_slots(None, AssistantRoute(intent="search", artist="Björk"))
    assert merged.last_intent == "search"
    assert merged.last_artist == "Björk"
    assert merged.last_playlist_ids == []


def test_route_without_intent_does_not_clear_the_last_one():
    """An unclear turn must not erase what the conversation was about — the
    sticky rule reads last_intent on the NEXT turn."""
    slots = AssistantSlots(last_intent="playlist")
    merged = merge_slots(slots, AssistantRoute(intent=None))
    assert merged.last_intent == "playlist"
