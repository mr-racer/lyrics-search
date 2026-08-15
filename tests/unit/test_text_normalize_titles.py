"""The key two titles must share to be the same song.

Written after a GTA station resolved 1 track out of 41. Most of that turned out
to be library coverage — the artists genuinely were not there — but the
measurement underneath found a real gap: 781 of 5630 library titles end in a
qualifying bracket, and a plain claim could not reach any of them.

The two halves of the rule pull against each other, so both are pinned here:
strip enough that "Work" finds "Work (Freemasons Remix)", strip little enough
that "(I Can't Get No) Satisfaction" never becomes "Satisfaction".
"""

import pytest

from app.services.text_normalize import strip_qualifiers, title_key


class TestQualifiersGo:
    @pytest.mark.parametrize("title,expected", [
        ("Work (Freemasons Remix)", "work"),
        ("Dusk Till Dawn (Radio Edit)", "dusk till dawn"),
        ("Take Ü There (Missy Elliott remix)", "take u there"),
        ("Give Me Novacaine (Live at Irving Plaza, New York, NY, 9/21/04)",
         "give me novacaine"),
        ("Forever Young (Special dance version) [2019 remaster]",
         "forever young"),
        ("Crazy Little Thing Called Love (acoustic)",
         "crazy little thing called love"),
        ("My Song [Explicit]", "my song"),
    ])
    def test_a_recording_note_is_not_part_of_the_name(self, title, expected):
        assert title_key(title) == expected

    def test_stacked_brackets_all_go(self):
        """They stack in real tags: one pass would leave "Faint [Live]"."""
        assert title_key("Faint [Live] [bonus track]") == "faint"

    @pytest.mark.parametrize("title", [
        "Power (feat. Dwele)", "30 Hours (ft. André 3000)",
        "Beautiful [feat. Enrique Iglesias]",
        "Lighters (Bad Meets Evil Feat. Bruno Mars)",
    ])
    def test_featuring_still_goes_bracketed_or_not(self, title):
        assert "feat" not in title_key(title)
        assert "ft" not in title_key(title).split()

    def test_the_claim_and_the_library_meet_in_the_middle(self):
        """The point of the whole exercise: both sides normalise the same, so
        it does not matter which of them carries the bracket."""
        assert title_key("Power (feat. Dwele)") == title_key("POWER")
        assert title_key("Work") == title_key("Work (Freemasons Remix)")


class TestTitlesStay:
    @pytest.mark.parametrize("title,expected", [
        ("(I Can’t Get No) Satisfaction", "i can t get no satisfaction"),
        ("(They Long to Be) Close to You", "they long to be close to you"),
        ("(Sittin' On) the Dock of the Bay", "sittin on the dock of the bay"),
    ])
    def test_a_leading_bracket_opens_the_sentence(self, title, expected):
        """Cutting these would turn them into "Satisfaction" and "Close to
        You" — different songs that other artists really do have."""
        assert title_key(title) == expected

    @pytest.mark.parametrize("title,expected", [
        ("See You on Monday (You’re Lost)", "see you on monday you re lost"),
        ("I Just Wanna Love U (Give It 2 Me)", "i just wanna love u give it 2 me"),
        ("Eh, Eh (Nothing Else I Can Say)", "eh eh nothing else i can say"),
        ("Crazy (Nobody Else)", "crazy nobody else"),
        ("The Betrayal (Act III)", "the betrayal act iii"),
    ])
    def test_a_trailing_bracket_can_still_be_the_title(self, title, expected):
        assert title_key(title) == expected

    def test_a_title_that_is_only_a_bracket_survives(self):
        """"[Premade Sandwiches]" is a real track name; stripping it would
        leave nothing to match on at all."""
        assert title_key("[Premade Sandwiches]") == "premade sandwiches"

    def test_a_bracket_without_a_qualifier_word_is_left_alone(self):
        assert strip_qualifiers("Canon (primo)") == "Canon (primo)"


class TestEdges:
    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_nothing_in_nothing_out(self, value):
        assert title_key(value) == ""

    def test_it_is_idempotent(self):
        once = title_key("Work (Freemasons Remix)")
        assert title_key(once) == once
