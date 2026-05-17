"""Unit tests for the Refined Facts AI task — accessors, parser, batching."""

import json

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.ai_tasks import refined_facts


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_parse_llm_response_accepts_valid_dict():
    raw = json.dumps({
        "selected_facts": [
            {"reasoning": "interesting", "short_fact": "Hello"},
            {"reasoning": "weird", "short_fact": "World"},
        ]
    })
    parsed = refined_facts._parse_llm_response(raw)
    assert parsed == [
        {"refined_text": "Hello"},
        {"refined_text": "World"},
    ]


def test_parse_llm_response_accepts_empty_selected_facts():
    raw = json.dumps({"selected_facts": []})
    parsed = refined_facts._parse_llm_response(raw)
    assert parsed == []


def test_parse_llm_response_rejects_malformed_json():
    with pytest.raises(ValueError):
        refined_facts._parse_llm_response("not json")


def test_parse_llm_response_rejects_non_dict():
    with pytest.raises(ValueError):
        refined_facts._parse_llm_response(json.dumps([{"id": 1}]))


def test_parse_llm_response_rejects_missing_selected_facts():
    with pytest.raises(ValueError):
        refined_facts._parse_llm_response(json.dumps({"facts": []}))


def test_get_refined_facts_returns_none_when_absent():
    assert MetadataDB.get_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
    ) is None


def test_set_and_get_refined_facts_roundtrip():
    MetadataDB.set_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
        refined=["short fact a", "short fact b"],
    )
    out = MetadataDB.get_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
    )
    assert out == ["short fact a", "short fact b"]


def test_set_empty_refined_still_creates_row():
    """Empty list is explicit: AI judged nothing interesting. Row exists,
    /facts should return [] instead of falling back to originals."""
    MetadataDB.set_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
        refined=[],
    )
    out = MetadataDB.get_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
    )
    assert out == []  # explicit empty, not None


def test_delete_refined_facts_returns_count():
    MetadataDB.set_refined_facts(
        scope="song", scope_key="t1", collection_name="music", lang="en",
        refined=["a"],
    )
    MetadataDB.set_refined_facts(
        scope="artist", scope_key="bar", collection_name="music", lang="en",
        refined=["b"],
    )
    n = MetadataDB.delete_refined_facts("music")
    assert n == 2
