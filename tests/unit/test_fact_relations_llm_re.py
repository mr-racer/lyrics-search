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


def _link(song, artist, relation="sample", direction="source"):
    return {"song": song, "artist": artist, "relation": relation, "direction": direction}


@pytest.mark.unit
def test_parse_llm_re_accepts_dict_passthrough():
    parsed = parse_llm_re({"producers": ["Rick Rubin"], "links": []})
    assert parsed == {"producers": ["Rick Rubin"], "links": []}


@pytest.mark.unit
def test_parse_llm_re_strips_code_fence_and_prose():
    raw = "Sure, here you go:\n```json\n" \
          '{"producers": ["Dr. Dre"], "links": [{"song": "Amen Brother", ' \
          '"artist": "The Winstons", "relation": "sample", "direction": "source"}]}' \
          "\n```"
    parsed = parse_llm_re(raw)
    assert parsed == {
        "producers": ["Dr. Dre"],
        "links": [_link("Amen Brother", "The Winstons")],
    }


@pytest.mark.unit
def test_parse_llm_re_normalizes_null_fields_and_drops_empty_links():
    raw = json.dumps({
        "producers": [],
        "links": [
            # voice-sample trap: song stays null
            {"song": None, "artist": "James Brown", "relation": "sample", "direction": "source"},
            {"song": None, "artist": None, "relation": "sample"},      # empty -> dropped
            {"song": "  ", "artist": "  ", "relation": "sample"},       # blank -> dropped
        ],
    })
    parsed = parse_llm_re(raw)
    assert parsed == {"producers": [], "links": [_link(None, "James Brown")]}


@pytest.mark.unit
def test_parse_llm_re_keeps_every_relation_label():
    """The classifier's job is to label, not to filter — gates do the dropping."""
    raw = json.dumps({"producers": [], "links": [
        {"song": "Hello", "artist": "Lionel Richie",
         "relation": "inspiration", "direction": "source"},
        {"song": "Milkshake", "artist": "Kelis",
         "relation": "lyrical_reference", "direction": "source"},
    ]})
    parsed = parse_llm_re(raw)
    assert [ln["relation"] for ln in parsed["links"]] == ["inspiration", "lyrical_reference"]


@pytest.mark.unit
def test_parse_llm_re_falls_back_on_invented_labels():
    """A 12B model inventing a label still tells us it is not a sample."""
    raw = json.dumps({"producers": [], "links": [
        {"song": "X", "artist": "Y", "relation": "homage", "direction": "sideways"},
    ]})
    parsed = parse_llm_re(raw)
    assert parsed["links"] == [_link("X", "Y", relation="other", direction="source")]


@pytest.mark.unit
def test_parse_llm_re_rejects_non_dict_link_items():
    assert parse_llm_re('{"producers": [], "links": ["oops"]}') is None
    assert parse_llm_re('{"producers": [], "links": "oops"}') is None


@pytest.mark.unit
def test_parse_llm_re_rejects_top_level_non_dict():
    assert parse_llm_re("[1, 2, 3]") is None
    assert parse_llm_re(json.dumps([1, 2, 3])) is None


@pytest.mark.unit
def test_merge_results_dedupes_links_and_keeps_as_is_first():
    as_is = {"producers": [], "links": [_link("Amen Brother", "The Winstons")]}
    llm = {"producers": [], "links": [
        _link("amen brother", "the winstons"),                     # dup, dropped
        _link("Funky Drummer", "James Brown"),                     # new, kept
        _link(None, "N.W.A", direction="usage"),                   # other direction, kept
    ]}
    out = merge_results(as_is, llm)
    assert out["links"] == [
        _link("Amen Brother", "The Winstons"),
        _link("Funky Drummer", "James Brown"),
        _link(None, "N.W.A", direction="usage"),
    ]


@pytest.mark.unit
def test_sound_relation_beats_a_weaker_one_on_the_same_pair():
    """Bound 2's real source was lost this way: one fact called the pair a
    lyrical reference and, being first, buried the fact that says
    "built around a sample of 'Bound'"."""
    weak = {"producers": [], "links": [
        _link("Bound", "Ponderosa Twins Plus One", relation="lyrical_reference")]}
    strong = {"producers": [], "links": [
        _link("Bound", "Ponderosa Twins Plus One", relation="sample")]}
    assert merge_results(weak, strong)["links"] == [
        _link("Bound", "Ponderosa Twins Plus One", relation="sample")]
    # …and the reverse order gives the same answer.
    assert merge_results(strong, weak)["links"] == [
        _link("Bound", "Ponderosa Twins Plus One", relation="sample")]


@pytest.mark.unit
def test_equal_strength_keeps_the_earlier_entry():
    a = {"producers": [], "links": [_link("X", "Y", relation="sample")]}
    b = {"producers": [], "links": [_link("X", "Y", relation="interpolation")]}
    assert merge_results(a, b)["links"] == [_link("X", "Y", relation="sample")]


@pytest.mark.unit
def test_merge_results_same_pair_both_directions_is_not_a_dup():
    a = {"producers": [], "links": [_link("X", "Y", direction="source")]}
    b = {"producers": [], "links": [_link("X", "Y", direction="usage")]}
    assert len(merge_results(a, b)["links"]) == 2


@pytest.mark.unit
def test_merge_results_handles_none_llm():
    as_is = {"producers": ["Rick Rubin"], "links": []}
    out = merge_results(as_is, None)
    assert out == {"producers": ["Rick Rubin"], "links": []}
