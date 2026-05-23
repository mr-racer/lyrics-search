"""Tests for file_processor.utils.normalize_genre()."""

from app.indexing.metadata_readers import normalize_genre


class TestNormalizeGenre:
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
