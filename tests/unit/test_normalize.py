"""Consolidated normalization unit tests.

Merged from:
  - test_text_normalize.py   -> TestTextNormalize
  - test_normalize_genre.py  -> TestNormalizeGenre
  - test_unit_norm.py        -> TestUnitNorm
  - test_validate_year.py    -> TestValidateYear / TestValidateYearInvalid
"""
import numpy as np
import pytest

from app.services.text_normalize import fold, tokenize, translit_variants, analyze
from app.indexing.metadata_readers import normalize_genre, validate_year
from app.resources.clap_features import unit_norm


class TestTextNormalize:
    """Unit tests for the catalog-search text analyzer (no Qdrant, no I/O)."""

    pytestmark = pytest.mark.unit

    def test_fold_strips_diacritics_and_lowercases(self):
        assert fold("Beyoncé") == "beyonce"
        assert fold("Motörhead") == "motorhead"
        assert fold("Sigur Rós") == "sigur ros"

    def test_fold_maps_punctuation_to_space_and_collapses(self):
        assert fold("  The   Wall ") == "the wall"
        assert fold("Rock & Roll!") == "rock roll"
        assert fold("AC/DC") == "ac dc"

    def test_fold_keeps_cyrillic_lowercased(self):
        assert fold("КиНо") == "кино"
        # NFKD folds ё→е and й→и — intentional, forgiving match.
        assert fold("Ёлка") == "елка"
        assert fold("Майк") == "маик"

    def test_tokenize_drops_noise_tokens(self):
        assert tokenize("The Beatles feat. Someone") == ["the", "beatles", "someone"]
        assert tokenize("Eminem ft Dido") == ["eminem", "dido"]

    def test_tokenize_splits_on_punctuation(self):
        assert tokenize("AC/DC") == ["ac", "dc"]
        assert tokenize("") == []

    def test_translit_cyrillic_to_latin_variant(self):
        v = translit_variants("кино")
        assert "кино" in v and "kino" in v
        # fold collapses й→и first, so цой→цои→tsoi (translit is char-mapping, approximate).
        assert "tsoi" in translit_variants("цой")
        assert "shar" in translit_variants("шар")

    def test_translit_latin_to_cyrillic_variant(self):
        assert "кино" in translit_variants("kino")
        assert "дом" in translit_variants("dom")

    def test_translit_leaves_pure_token_when_no_cross_script(self):
        # A token with no letters stays as-is; a single script returns at least itself.
        assert "123" in translit_variants("123")

    def test_analyze_flattens_tokens_and_translit_variants(self):
        bag = set(analyze("Кино"))
        assert {"кино", "kino"} <= bag
        assert analyze("") == []
        assert {"the", "wall"} <= set(analyze("The Wall"))


class TestNormalizeGenre:
    """Tests for file_processor.utils.normalize_genre()."""

    def test_none_returns_other(self):
        assert normalize_genre(None) == "Other"

    def test_exact_match_pop(self):
        assert normalize_genre("pop") == "Pop"

    def test_case_insensitive(self):
        assert normalize_genre("POP") == "Pop"

    def test_rock_mapping(self):
        assert normalize_genre("rock") == "Rock"

    def test_hip_hop_mapping(self):
        assert normalize_genre("hip hop") == "Hip-Hop"

    def test_hip_hon_with_hyphen(self):
        assert normalize_genre("hip-hop") == "Hip-Hop"

    def test_r_and_b_mapping(self):
        assert normalize_genre("r&b") == "R&B/Soul"

    def test_electronic_mapping(self):
        assert normalize_genre("electronic") == "Electronic"

    def test_russian_genre(self):
        assert normalize_genre("поп") == "Pop"

    def test_noise_returns_other(self):
        assert normalize_genre("none") == "Other"

    def test_unknown_returns_other(self):
        assert normalize_genre("some-random-genre") == "Other"

    def test_multi_genre_comma_split(self):
        """Multi-genre strings split on comma, take first match."""
        assert normalize_genre("pop, rock") == "Pop"

    def test_multi_genre_slash_split(self):
        """'pop/rock' — 'pop/rock' is in genre_map → Rock (exact match before split)."""
        result = normalize_genre("pop/rock")
        assert result == "Rock"

    def test_keyword_fallback_hip_hop(self):
        assert normalize_genre("hip hop fusion") == "Hip-Hop"

    def test_keyword_fallback_metal(self):
        assert normalize_genre("death metal") == "Rock"

    def test_nu_metal(self):
        assert normalize_genre("nu metal") == "Nu-Metal"

    def test_soul_mapping(self):
        assert normalize_genre("soul") == "R&B/Soul"

    def test_indie_mapping(self):
        assert normalize_genre("indie") == "Alternative"

    def test_dance_mapping(self):
        assert normalize_genre("dance") == "Electronic"

    def test_blues_mapping(self):
        assert normalize_genre("blues") == "R&B/Soul"

    def test_empty_string(self):
        assert normalize_genre("") == "Other"

    def test_whitespace_handling(self):
        assert normalize_genre("  pop  ") == "Pop"


class TestUnitNorm:
    """Tests for search_engine.utils.unit_norm()."""

    def test_unit_vector_returns_itself(self):
        v = np.array([1.0, 0.0, 0.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_zero_vector_returns_zero(self):
        v = np.array([0.0, 0.0, 0.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_random_vector_normalizes_to_unit_length(self):
        v = np.array([3.0, 4.0, 0.0])
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_1d_vector_works(self):
        v = np.array([5.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_negative_values_handled(self):
        v = np.array([-3.0, 4.0, 0.0])
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        assert result[0] < 0

    def test_returns_numpy_array(self):
        v = np.array([1.0, 2.0, 3.0])
        result = unit_norm(v)
        assert isinstance(result, np.ndarray)

    def test_high_dim_vector(self):
        v = np.random.randn(512)
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


class TestValidateYear:
    """Valid years (1900–current) return int."""

    def test_valid_year_2020(self):
        assert validate_year("2020") == 2020

    def test_valid_year_1900(self):
        assert validate_year("1900") == 1900

    def test_valid_year_current(self):
        assert validate_year("2026") == 2026

    def test_valid_year_1999(self):
        assert validate_year("1999") == 1999

    def test_embedded_year_extracted(self):
        assert validate_year("released in 2015") == 2015

    def test_year_in_album_string(self):
        assert validate_year("After Hours (2020)") == 2020


class TestValidateYearInvalid:
    """Out-of-range and malformed inputs return None."""

    def test_none_input(self):
        assert validate_year(None) is None

    def test_empty_string(self):
        assert validate_year("") is None

    def test_year_before_1900(self):
        assert validate_year("1899") is None

    def test_year_after_current(self):
        assert validate_year("2030") is None

    def test_non_numeric(self):
        assert validate_year("two thousand twenty") is None

    def test_single_digit(self):
        assert validate_year("5") is None

    def test_negative_year(self):
        assert validate_year("-200") is None

    def test_float_string(self):
        # Regex extracts "2020" from "2020.5" — this is the actual (documented) behavior
        assert validate_year("2020.5") == 2020
