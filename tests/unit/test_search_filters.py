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
        assert result == {"artist": None, "album": None, "genre": None}

    def test_populated_filters(self):
        svc = SearchService.__new__(SearchService)
        f = SearchFilters(artist="A", album="B", genre="C")
        result = svc._extract_filter_kwargs(f)
        assert result == {"artist": "A", "album": "B", "genre": "C"}

# TODO UPDATE WITH CURRENTLY USED FUCNTION LOGIC - _build_qdrant_filter_models
class TestBuildQdrantFilter:
    def test_no_filters_returns_none(self):
        svc = SearchService.__new__(SearchService)
        assert svc._build_qdrant_filter(None) is None
        assert svc._build_qdrant_filter(SearchFilters()) is None

    def test_artist_filter(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter(SearchFilters(artist="The Weeknd"))
        assert f is not None
        assert len(f["must"]) == 1
        assert f["must"][0] == {"key": "artist", "match": {"value": "The Weeknd"}}

    def test_multiple_conditions(self):
        svc = SearchService.__new__(SearchService)
        f = svc._build_qdrant_filter(
            SearchFilters(artist="A", album="B", genre="C")
        )
        assert len(f["must"]) == 3


class TestBuildQdrantFilterModels:
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
