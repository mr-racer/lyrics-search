"""Tests for library route _score_query() — browse relevance scoring.

_score_query logic:
  For each word w in q_words (already lowered):
    - if w == value.lower()          → +10  (word equals the ENTIRE value)
    - elif value.lower().startswith(w) → +5  (value starts with the word)
    - elif w in value.lower()         → +2  (word appears inside value)
  If q_full in value.lower()          → +3  (full query bonus)

Note: q_full is empty string passes ``in`` check, so empty query adds +3.
"""

from app.api.routes.library import _score_query


class TestScoreQuery:
    def test_no_match(self):
        """Words and full query don't appear in value."""
        assert _score_query(["hello", "world"], "hello world", "foo bar baz") == 0.0

    def test_exact_word_equals_value(self):
        """Word matches the entire lowered value → +10, full query bonus +3 = 13."""
        assert _score_query(["weeknd"], "weeknd", "Weeknd") == 13.0

    def test_prefix_match(self):
        """Value starts with the word → +5, full query bonus +3 = 8."""
        assert _score_query(["bil"], "bil", "Billie") == 8.0

    def test_substring_match(self):
        """Word appears inside value → +2, full query bonus +3 = 5."""
        assert _score_query(["lie"], "lie", "Billie") == 5.0

    def test_full_query_bonus(self):
        """Full query appears in value → +3 bonus on top of word scores."""
        score = _score_query(["the", "weeknd"], "the weeknd", "The Weeknd")
        # "the" startswith "the weeknd" → +5, "weeknd" in "the weeknd" → +2, full +3
        assert score == 10.0

    def test_multiple_word_matches(self):
        score = _score_query(["blinding", "lights"], "blinding lights", "Blinding Lights")
        # "blinding" startswith → +5, "lights" in value → +2, full +3
        assert score == 10.0

    def test_none_value(self):
        assert _score_query(["a"], "a", "") == 0.0

    def test_case_insensitive(self):
        """Exact match: word == lowered value → +10, bonus +3 = 13."""
        assert _score_query(["pop"], "pop", "POP") == 13.0

    def test_word_inside_value_not_prefix(self):
        """'lie' in 'Billie' is substring → +2, bonus +3 = 5."""
        assert _score_query(["lie"], "lie", "Billie") == 5.0

    def test_empty_query_words(self):
        """Empty q_words but q_full="" is always ``in`` any string → +3."""
        assert _score_query([], "", "anything") == 3.0

    def test_word_matches_whole_value_exact(self):
        """Single word equals the whole value → +10, bonus +3 = 13."""
        assert _score_query(["hello"], "hello", "Hello") == 13.0

    def test_multiple_words_each_scored(self):
        """Each word scored independently, plus full query bonus."""
        score = _score_query(["a", "b"], "a b", "a and b here")
        # "a" in value → +2, "b" in value → +2, full "a b" in → +3
        assert score == 7.0
