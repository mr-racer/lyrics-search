"""Tests for file_processor.utils.validate_year()."""

from app.indexing.metadata_readers import validate_year


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
