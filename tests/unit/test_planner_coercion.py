"""``_coerce_plan`` — what replaced PydanticAI's ``output_type`` on the planner.

The structured-output Agent hung forever on the deployment's 12b model, so the
planner now asks for plain JSON and this function does the validation in code.
It therefore has to survive everything a small model does to a schema: missing
keys, nulls where a Literal is required, bare strings instead of objects,
invented filter fields. It must never raise — the caller treats an empty plan as
"no planner queries" and falls through to its own loop.
"""
from __future__ import annotations

import pytest

from app.services.agents import _coerce_plan


def test_a_well_formed_plan_survives_intact():
    plan = _coerce_plan({
        "action": "search",
        "query_type": "text",
        "filters": {"artist": "Kanye West", "album": None, "genre": None},
        "filter_lookup": None,
        "queries": [{"query": "broken heart in the rain"}, {"query": "city lights"}],
        "search_mode": "AGGRESSIVE",
    })
    assert plan.action == "search"
    assert plan.query_type == "text"
    assert plan.filters.artist == "Kanye West"
    assert [q.query for q in plan.queries] == ["broken heart in the rain", "city lights"]
    assert plan.search_mode == "AGGRESSIVE"


@pytest.mark.parametrize("raw", [None, "not json", 42, [], {}])
def test_garbage_degrades_to_an_empty_search_plan(raw):
    plan = _coerce_plan(raw)
    assert plan.action == "search"
    assert plan.query_type == "hybrid"
    assert plan.queries == []


@pytest.mark.parametrize("field,value", [
    ("action", None), ("action", "sing"), ("action", False),
    ("query_type", None), ("query_type", "vibes"),
    ("search_mode", None), ("search_mode", "YOLO"),
])
def test_invalid_literals_fall_back_to_defaults(field, value):
    """Local models return null or an invented word where a Literal is required —
    a ValidationError here would kill the whole search request."""
    plan = _coerce_plan({field: value, "queries": []})
    assert plan.action in ("search", "request_filter")
    assert plan.query_type in ("text", "audio", "hybrid")
    assert plan.search_mode in ("CONSERVATIVE", "AGGRESSIVE")


def test_queries_given_as_bare_strings_are_accepted():
    plan = _coerce_plan({"queries": ["rain and loneliness", "  ", ""]})
    assert [q.query for q in plan.queries] == ["rain and loneliness"]


def test_query_whitespace_is_trimmed():
    plan = _coerce_plan({"queries": [{"query": "  neon city  "}]})
    assert plan.queries[0].query == "neon city"


def test_all_null_filters_become_none():
    """«filters: {artist: null, …}» must not produce an empty filter object that
    later narrows the search to nothing."""
    plan = _coerce_plan({"filters": {"artist": None, "album": None, "genre": None}})
    assert plan.filters is None


def test_invented_filter_keys_are_dropped_not_fatal():
    plan = _coerce_plan({"filters": {"artist": "Björk", "mood": "sad", "bpm": 120}})
    assert plan.filters.artist == "Björk"
    assert not hasattr(plan.filters, "mood")


def test_legacy_singular_year_range_is_ignored():
    """SearchFilters is extra="ignore" and the prompt still shows the old
    singular key — it must not blow up the plan."""
    plan = _coerce_plan({"filters": {"artist": "Queen", "year_range": "1970-1979"}})
    assert plan.filters.artist == "Queen"


def test_filter_lookup_keeps_only_non_empty_strings():
    plan = _coerce_plan({"filter_lookup": {"artist": "канье", "album": None, "genre": "  "}})
    assert plan.filter_lookup == {"artist": "канье"}


def test_empty_filter_lookup_becomes_none():
    """None is the signal the caller checks before spending a resolve round-trip."""
    plan = _coerce_plan({"filter_lookup": {"artist": None, "album": ""}})
    assert plan.filter_lookup is None


def test_request_filter_action_is_preserved():
    plan = _coerce_plan({"action": "request_filter",
                         "filter_lookup": {"artist": "канье уест"}})
    assert plan.action == "request_filter"
    assert plan.filter_lookup == {"artist": "канье уест"}


def test_malformed_query_entries_are_skipped_not_fatal():
    plan = _coerce_plan({"queries": [{"nope": "x"}, None, 7, {"query": "good one"}]})
    assert [q.query for q in plan.queries] == ["good one"]
