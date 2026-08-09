"""The planner's validators — the layer that decides what the model is allowed
to have said.

Every test here corresponds to something a 12b model actually does: inventing a
mood the user never mentioned, answering "the nineties" where a range was
asked for, dropping the quotes around a title, returning the same query twice.
"""

from lab.agent.models import Abbreviation
from lab.agent.planner import (TOOLS, Planner, parse_abbreviation, parse_era,
                               quote_work, validate_style)


class _NoLLM:
    """Planner.validate needs no client; construction should not need one either."""


def _planner():
    return Planner(_NoLLM())


class TestEra:
    def test_a_range_is_parsed(self):
        assert parse_era("1990-1999") == (1990, 1999)

    def test_a_decade_word_becomes_a_range(self):
        assert parse_era("2000s") == (2000, 2009)

    def test_a_russian_decade_becomes_a_range(self):
        assert parse_era("1980е") == (1980, 1989)

    def test_after_a_year_is_open_ended(self):
        """The boundary word lives in the USER's sentence, not the model's
        answer — that is where it is looked for."""
        era = parse_era("2020", user_text="хиты Канье после 2020 года")
        assert era is not None and era[0] == 2020 and era[1] >= 2026

    def test_a_bare_year_without_a_boundary_word_is_a_point(self):
        assert parse_era("2006", user_text="песни на радио в 2006") == (2006, 2006)

    def test_nonsense_is_dropped_rather_than_used(self):
        assert parse_era("0-9999") is None
        assert parse_era("the nineties") is None
        assert parse_era(None) is None

    def test_a_reversed_range_is_swapped(self):
        assert parse_era("1999-1990") == (1990, 1999)

    def test_a_dict_shape_is_accepted(self):
        assert parse_era({"from": 1970, "to": 1979}) == (1970, 1979)


class TestStyle:
    def test_a_style_the_user_wrote_survives(self):
        assert validate_style("спокойные", "спокойные хиты 80х") == "спокойные"

    def test_an_invented_style_is_dropped(self):
        """The single most common planner hallucination: a mood nobody asked
        for, which would then filter out most of the library."""
        assert validate_style("energetic", "хиты Канье Уэста") is None

    def test_a_partly_invented_style_keeps_only_the_real_words(self):
        got = validate_style("спокойные энергичные", "спокойные хиты 80х")
        assert got == "спокойные"

    def test_case_and_punctuation_do_not_matter(self):
        assert validate_style("Клубные,", "популярные клубные хиты 00х") == "Клубные,"

    def test_empty_stays_empty(self):
        assert validate_style(None, "что угодно") is None


class TestQuoting:
    def test_a_missing_title_is_added_in_quotes(self):
        got = quote_work("soundtrack full track list", "Grand Theft Auto V")
        assert got.startswith('"Grand Theft Auto V"')

    def test_an_unquoted_title_gets_quoted_in_place(self):
        got = quote_work("Grand Theft Auto V soundtrack", "Grand Theft Auto V")
        assert got == '"Grand Theft Auto V" soundtrack'

    def test_an_already_quoted_title_is_left_alone(self):
        query = '"Grand Theft Auto V" soundtrack'
        assert quote_work(query, "Grand Theft Auto V") == query

    def test_no_work_means_no_change(self):
        assert quote_work("kanye west hits", None) == "kanye west hits"


class TestAbbreviation:
    def test_a_well_formed_expansion_is_kept(self):
        abbr = parse_abbreviation({"raw": "GTA 5", "expansion": "Grand Theft Auto V",
                                   "confidence": 0.9})
        assert isinstance(abbr, Abbreviation)
        assert abbr.confidence == 0.9

    def test_an_expansion_equal_to_the_input_is_not_one(self):
        assert parse_abbreviation({"raw": "Queen", "expansion": "queen",
                                   "confidence": 1.0}) is None

    def test_confidence_is_clamped(self):
        abbr = parse_abbreviation({"raw": "TDU", "expansion": "Test Drive Unlimited",
                                   "confidence": 7})
        assert abbr.confidence == 1.0

    def test_garbage_is_ignored(self):
        assert parse_abbreviation("GTA 5") is None
        assert parse_abbreviation({"raw": "", "expansion": "x"}) is None


class TestValidate:
    def test_an_unknown_intent_stops_the_run(self):
        """Better to ask than to run the wrong pipeline for thirty seconds."""
        assert _planner().validate({"intent": "vibes"}, "что-нибудь") is None

    def test_tools_are_assigned_by_code(self):
        plan = _planner().validate({"intent": "general"}, "почему Eminem так зовут")
        assert plan.allowed_tools == TOOLS["general"]
        assert "library_catalog" not in plan.allowed_tools

    def test_queries_fall_back_to_the_users_own_sentence(self):
        """A formatting slip must not end the run before it searched once."""
        plan = _planner().validate({"intent": "general", "web_queries": []},
                                   "почему Eminem так зовут")
        assert plan.web_queries == ["почему Eminem так зовут"]

    def test_duplicate_queries_collapse(self):
        plan = _planner().validate(
            {"intent": "playlist",
             "web_queries": ["kanye hits", "Kanye  Hits", "kanye best songs"]},
            "хиты канье")
        assert len(plan.web_queries) == 2

    def test_the_work_is_forced_into_every_query(self):
        plan = _planner().validate(
            {"intent": "playlist", "work": "Grand Theft Auto V",
             "web_queries": ["soundtrack list", "radio station songs"]},
            "музыка из гта 5")
        assert all('"Grand Theft Auto V"' in q for q in plan.web_queries)

    def test_an_absurd_count_is_ignored(self):
        plan = _planner().validate({"intent": "playlist", "count": 5000},
                                   "собери плейлист")
        assert plan.filters.count is None

    def test_ce_query_falls_back_to_the_message(self):
        plan = _planner().validate({"intent": "general"}, "почему так вышло")
        assert plan.ce_query == "почему так вышло"
