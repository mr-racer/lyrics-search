"""Tests for sonic_tags (AND) filter wiring.

We don't need a live Qdrant — Qdrant's filter-matching is its contract,
not ours. We assert the structure of the Filter object SearchService produces."""
from __future__ import annotations

from app.domain.models import SearchFilters
from app.services.search_service import SearchService


def _build(filters: SearchFilters):
    svc = SearchService.__new__(SearchService)  # bypass __init__
    return svc._build_qdrant_filter_models(filters)


def test_no_sonic_filter_returns_filter_only_with_legacy():
    f = _build(SearchFilters(artist="Beach House"))
    assert f is not None
    keys = {c.key for c in f.must}
    assert "artist" in keys
    assert "sonic_tags" not in keys


def test_sonic_tags_emit_one_must_condition_per_tag_for_and_semantics():
    f = _build(SearchFilters(sonic_tags=["melancholic", "lo-fi"]))
    assert f is not None
    tag_conds = [c for c in f.must if c.key == "sonic_tags"]
    assert len(tag_conds) == 2
    matched_values = {c.match.value for c in tag_conds}
    assert matched_values == {"melancholic", "lo-fi"}


def test_sonic_tags_combine_with_legacy_artist():
    f = _build(SearchFilters(
        artist="Beach House",
        sonic_tags=["melancholic"],
    ))
    assert f is not None
    keys = [c.key for c in f.must]
    assert keys.count("artist") == 1
    assert keys.count("sonic_tags") == 1


def test_empty_sonic_tags_list_does_not_emit_conditions():
    """Empty sonic_tags=[] must be treated as 'no filter'."""
    f = _build(SearchFilters(artist="A", sonic_tags=[]))
    assert f is not None
    keys = {c.key for c in f.must}
    assert keys == {"artist"}
