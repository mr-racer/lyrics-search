"""JSON extraction and field coercion for the assistant's LLM replies.

Both of these exist because local models are formatted loosely, not because
they answer badly. Every case below is a reply whose CONTENT was correct and
which the old code discarded over its shape — and each discard cost a whole
feature run, since the callers here have deterministic fallbacks rather than
retries.
"""

import pytest

from app.services.assistant.llm import (
    as_str,
    as_str_list,
    extract_json_array,
    extract_json_object,
)

ARRAY = ('["This song is a slow piano piece", '
         '"This song is a warm acoustic track"]')
OBJECT_ITEMS = ('[{"prompt": "This song is a slow piano piece"}, '
                '{"prompt": "This song is a warm acoustic track"}]')


class TestExtractJsonArray:
    @pytest.mark.parametrize("text", [
        ARRAY,
        "```json\n" + ARRAY + "\n```",
        "Here are the prompts:\n" + ARRAY,
        ARRAY + "\nHope this helps!",
        '{"prompts": ' + ARRAY + '}',
    ])
    def test_the_shapes_that_always_worked(self, text):
        assert len(extract_json_array(text)) == 2

    @pytest.mark.parametrize("text", [
        "<think>Variant 1: [tempo], variant 2: [timbre]</think>\n" + ARRAY,
        "Step [1]: acoustic mapping.\n" + ARRAY,
        ARRAY + "\nNote: see rule [7].",
    ])
    def test_a_stray_bracket_no_longer_eats_the_answer(self, text):
        """The old scan took everything between the first '[' and the last ']',
        so any bracket in a preamble or a footnote made the slice unparseable
        and the reply was lost whole."""
        assert extract_json_array(text) == [
            "This song is a slow piano piece",
            "This song is a warm acoustic track",
        ]

    def test_a_bracket_inside_a_string_is_not_a_delimiter(self):
        assert extract_json_array('["This song is a [slow] piano piece"]') == [
            "This song is a [slow] piano piece",
        ]

    def test_the_richest_array_wins_over_the_first_one(self):
        """"Step [1]" parses perfectly well as ``[1]``. Preferring the
        candidate that actually carries content keeps it from winning."""
        assert extract_json_array("Step [1]:\n" + ARRAY)[0].startswith("This song")

    @pytest.mark.parametrize("text", ["", "1. one\n2. two", "no json here"])
    def test_nothing_parseable_is_still_none(self, text):
        assert extract_json_array(text) is None

    def test_an_object_reply_is_left_to_the_object_extractor(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}


class TestAsStr:
    def test_a_dict_is_unwrapped_by_any_string_it_carries(self):
        """The production failure: the CLAP rephrasing asks for prompts and a
        9b model answers ``[{"prompt": ...}]``. Keying the unwrap to
        text/value returned "" for every item, so four good prompts became an
        empty list and the branch fell back to a one-line query."""
        assert as_str({"prompt": "X"}) == "X"
        assert as_str({"query": "X"}) == "X"
        assert as_str({"caption": "X"}) == "X"

    def test_the_blessed_keys_still_win_when_present(self):
        assert as_str({"prompt": "second", "text": "first"}) == "first"
        assert as_str({"prompt": "second", "value": "first"}) == "first"

    def test_a_blank_blessed_key_falls_through(self):
        assert as_str({"text": "   ", "prompt": "real"}) == "real"

    def test_a_dict_with_no_strings_is_empty(self):
        assert as_str({"n": 3, "ok": True}) == ""

    def test_the_limit_still_applies(self):
        assert as_str({"prompt": "x" * 500}, 10) == "x" * 10

    def test_a_list_of_object_items_becomes_a_list_of_strings(self):
        assert as_str_list(extract_json_array(OBJECT_ITEMS)) == [
            "This song is a slow piano piece",
            "This song is a warm acoustic track",
        ]
