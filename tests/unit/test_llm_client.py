"""Tests for llm_client._normalize_base_url()."""

from app.services.llm_client import _normalize_base_url


class TestNormalizeBaseUrl:
    def test_adds_v1_if_missing(self):
        assert _normalize_base_url("http://localhost:8000") == "http://localhost:8000/v1"

    def test_keeps_existing_v1(self):
        assert _normalize_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"

    def test_strips_trailing_slash(self):
        assert _normalize_base_url("http://localhost:8000/") == "http://localhost:8000/v1"

    def test_strips_trailing_slash_with_v1(self):
        assert _normalize_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"

    def test_empty_string(self):
        assert _normalize_base_url("") == "/v1"

    def test_with_path(self):
        url = "http://example.com/custom/path"
        result = _normalize_base_url(url)
        assert result == "http://example.com/custom/path/v1"

    def test_ollama_style_url(self):
        assert _normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"
