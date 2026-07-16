import json

import pytest

from app.services.fact_relations.llm_re import (
    build_llm_messages,
    mark_fact,
    merge_results,
    parse_llm_re,
)


@pytest.mark.unit
def test_mark_fact_wraps_first_occurrence():
    m = mark_fact("Produced by Rick Rubin.", [("Rick Rubin", "Person")])
    assert "[Person: Rick Rubin]" in m


@pytest.mark.unit
def test_parse_rejects_garbage():
    assert parse_llm_re("not json") is None
    assert parse_llm_re('{"producers": "oops"}') is None


@pytest.mark.unit
def test_merge_dedupes_case_insensitive():
    out = merge_results({"producers": ["Rick Rubin"], "samples": [], "sampled_by": []},
                        {"producers": ["rick rubin", "Dr. Dre"], "samples": [], "sampled_by": []})
    assert sorted(out["producers"]) == ["Dr. Dre", "Rick Rubin"]


# --- additional coverage (gap-fill beyond the brief's verbatim fixtures) ---

@pytest.mark.unit
def test_mark_fact_longest_first_and_no_nesting():
    # "Rick Rubin" is a substring of "Def Jam's Rick Rubin era" style text isn't
    # the point here; the point is that once a marker is inserted, a shorter
    # candidate whose text falls *inside* the already-marked span is skipped.
    fact = "Produced by Rick Rubin and Rick."
    marked = mark_fact(fact, [("Rick", "Person"), ("Rick Rubin", "Person")])
    # Longest ("Rick Rubin") is tried first and wins the first occurrence.
    assert "[Person: Rick Rubin]" in marked
    # The bare "Rick" inside "[Person: Rick Rubin]" must not get its own marker
    # (that would nest inside the existing bracket).
    assert "[Person: Rick]]" not in marked
    assert marked.count("[Person: Rick Rubin]") == 1


@pytest.mark.unit
def test_mark_fact_only_first_occurrence():
    fact = "Rick Rubin produced it. Rick Rubin also mixed it."
    marked = mark_fact(fact, [("Rick Rubin", "Person")])
    assert marked.count("[Person: Rick Rubin]") == 1


@pytest.mark.unit
def test_mark_fact_ignores_missing_and_duplicate_candidates():
    fact = "Produced by Rick Rubin."
    marked = mark_fact(fact, [("Rick Rubin", "Person"), ("Rick Rubin", "Person"), ("Nowhere", "Song")])
    assert marked == "Produced by [Person: Rick Rubin]."


@pytest.mark.unit
def test_build_llm_messages_shape_and_content():
    msgs = build_llm_messages("Song X", "some-artist", "[Person: Rick Rubin] produced [Song: Song X].")
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "producers" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "Song X" in msgs[1]["content"]
    assert "some artist" in msgs[1]["content"]  # dashes normalized to spaces
    assert "[Person: Rick Rubin]" in msgs[1]["content"]
    assert msgs[1]["content"].rstrip().endswith("JSON:")


@pytest.mark.unit
def test_parse_llm_re_accepts_dict_passthrough():
    parsed = parse_llm_re({"producers": ["Rick Rubin"], "samples": [], "sampled_by": []})
    assert parsed == {"producers": ["Rick Rubin"], "samples": [], "sampled_by": []}


@pytest.mark.unit
def test_parse_llm_re_strips_code_fence_and_prose():
    raw = "Sure, here you go:\n```json\n" \
          '{"producers": ["Dr. Dre"], "samples": [{"song": "Amen Brother", "artist": "The Winstons"}], "sampled_by": []}' \
          "\n```"
    parsed = parse_llm_re(raw)
    assert parsed == {
        "producers": ["Dr. Dre"],
        "samples": [{"song": "Amen Brother", "artist": "The Winstons"}],
        "sampled_by": [],
    }


@pytest.mark.unit
def test_parse_llm_re_normalizes_null_fields_and_drops_empty_relations():
    raw = json.dumps({
        "producers": [],
        "samples": [
            {"song": None, "artist": "James Brown"},   # voice-sample trap: song stays null
            {"song": None, "artist": None},             # empty relation, must be dropped
            {"song": "  ", "artist": "  "},              # blank strings normalize to None -> dropped
        ],
        "sampled_by": [],
    })
    parsed = parse_llm_re(raw)
    assert parsed == {
        "producers": [],
        "samples": [{"song": None, "artist": "James Brown"}],
        "sampled_by": [],
    }


@pytest.mark.unit
def test_parse_llm_re_rejects_non_dict_relation_items():
    assert parse_llm_re('{"producers": [], "samples": ["oops"], "sampled_by": []}') is None
    assert parse_llm_re('{"producers": [], "samples": [], "sampled_by": "oops"}') is None


@pytest.mark.unit
def test_parse_llm_re_rejects_top_level_non_dict():
    assert parse_llm_re("[1, 2, 3]") is None
    assert parse_llm_re(json.dumps([1, 2, 3])) is None


@pytest.mark.unit
def test_merge_results_dedupes_relations_and_keeps_as_is_first():
    as_is = {
        "producers": [],
        "samples": [{"song": "Amen Brother", "artist": "The Winstons"}],
        "sampled_by": [],
    }
    llm = {
        "producers": [],
        "samples": [
            {"song": "amen brother", "artist": "the winstons"},  # dup, dropped
            {"song": "Funky Drummer", "artist": "James Brown"},   # new, kept
        ],
        "sampled_by": [{"song": None, "artist": "N.W.A"}],
    }
    out = merge_results(as_is, llm)
    assert out["samples"] == [
        {"song": "Amen Brother", "artist": "The Winstons"},
        {"song": "Funky Drummer", "artist": "James Brown"},
    ]
    assert out["sampled_by"] == [{"song": None, "artist": "N.W.A"}]


@pytest.mark.unit
def test_merge_results_handles_none_llm():
    as_is = {"producers": ["Rick Rubin"], "samples": [], "sampled_by": []}
    out = merge_results(as_is, None)
    assert out == {"producers": ["Rick Rubin"], "samples": [], "sampled_by": []}
