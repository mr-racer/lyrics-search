"""Tests for SearchService filter builders."""

from app.domain.models import SearchFilters
from app.services.search_service import SearchService


class TestExtractFilterKwargs:
    def test_no_filters(self):
        svc = SearchService.__new__(SearchService)  # no __init__
        result = svc._extract_filter_kwargs(None)
        assert result == {}

    def test_empty_filters(self):
        svc = SearchService.__new__(SearchService)
        result = svc._extract_filter_kwargs(SearchFilters())
        assert result == {
            "artist": None,
            "album": None,
            "genre": None,
            "year_range": None,
            "sonic_tags": None,
        }

    def test_populated_filters(self):
        svc = SearchService.__new__(SearchService)
        f = SearchFilters(artist="A", album="B", genre="C")
        result = svc._extract_filter_kwargs(f)
        assert result == {
            "artist": "A",
            "album": "B",
            "genre": "C",
            "year_range": None,
            "sonic_tags": None,
        }


class TestBuildQdrantFilterModels:
    """Tests for _build_qdrant_filter_models (qdrant_client.models.Filter)."""

    def test_no_filters_returns_none(self):
        svc = SearchService.__new__(SearchService)
        assert svc._build_qdrant_filter_models(None) is None
        assert svc._build_qdrant_filter_models(SearchFilters()) is None

    def test_artist_filter(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter_models(SearchFilters(artist="A"))
        from qdrant_client import models

        assert isinstance(f, models.Filter)
        assert len(f.must) == 1

    def test_multiple_conditions(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter_models(
            SearchFilters(artist="A", album="B", genre="C")
        )
        from qdrant_client import models

        assert isinstance(f, models.Filter)
        assert len(f.must) == 3


def test_search_filters_accepts_sonic_tags_list():
    f = SearchFilters(sonic_tags=["melancholic", "lo-fi"])
    assert f.sonic_tags == ["melancholic", "lo-fi"]


def test_search_filters_sonic_tags_default_empty_not_none():
    """Empty list (not None) is the right default so iteration is safe."""
    f = SearchFilters()
    assert f.sonic_tags == []


def test_search_filters_accepts_year_ranges_list():
    from app.domain.models import SearchFilters
    f = SearchFilters(year_ranges=["1990-1999", "2000-2009"])
    assert f.year_ranges == ["1990-1999", "2000-2009"]


def test_search_filters_year_ranges_default_empty_not_none():
    from app.domain.models import SearchFilters
    f = SearchFilters()
    assert f.year_ranges == []


def test_search_filters_dead_fields_removed():
    """Plan B2.2 drops year_from/year_to/duration_*. They should not exist on the model."""
    from app.domain.models import SearchFilters
    fields = SearchFilters.model_fields
    assert "year_from" not in fields
    assert "year_to" not in fields
    assert "duration_min_sec" not in fields
    assert "duration_max_sec" not in fields
    assert "year_range" not in fields  # singular form replaced by year_ranges
