"""Tests for year_ranges (OR) filter wiring."""
from __future__ import annotations

from qdrant_client import models

from app.domain.models import SearchFilters
from app.services.search_service import SearchService


def _build(filters: SearchFilters):
    svc = SearchService.__new__(SearchService)
    return svc._build_qdrant_filter_models(filters)


def test_no_decade_filter_does_not_emit_year_range_condition():
    f = _build(SearchFilters(artist="A"))
    assert f is not None
    keys = {c.key for c in f.must}
    assert "year_range" not in keys


def test_single_decade_uses_match_any_with_one_value():
    f = _build(SearchFilters(year_ranges=["1990-1999"]))
    assert f is not None
    year_conds = [c for c in f.must if c.key == "year_range"]
    assert len(year_conds) == 1
    match = year_conds[0].match
    assert isinstance(match, models.MatchAny)
    assert list(match.any) == ["1990-1999"]


def test_multiple_decades_use_single_match_any_for_or_semantics():
    f = _build(SearchFilters(year_ranges=["1990-1999", "2000-2009", "2010-2019"]))
    assert f is not None
    year_conds = [c for c in f.must if c.key == "year_range"]
    assert len(year_conds) == 1
    match = year_conds[0].match
    assert isinstance(match, models.MatchAny)
    assert set(match.any) == {"1990-1999", "2000-2009", "2010-2019"}


def test_decade_combines_with_artist_and_sonic_tags():
    f = _build(SearchFilters(
        artist="A",
        year_ranges=["1990-1999"],
        sonic_tags=["melancholic"],
    ))
    assert f is not None
    keys = [c.key for c in f.must]
    assert keys.count("artist") == 1
    assert keys.count("year_range") == 1
    assert keys.count("sonic_tags") == 1


def test_empty_year_ranges_list_does_not_emit_condition():
    f = _build(SearchFilters(artist="A", year_ranges=[]))
    assert f is not None
    keys = {c.key for c in f.must}
    assert keys == {"artist"}
