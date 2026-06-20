"""Unit tests for the catalog-search text analyzer (no Qdrant, no I/O)."""
import pytest

pytestmark = pytest.mark.unit

from app.services.text_normalize import fold, tokenize, translit_variants, analyze


def test_fold_strips_diacritics_and_lowercases():
    assert fold("Beyoncé") == "beyonce"
    assert fold("Motörhead") == "motorhead"
    assert fold("Sigur Rós") == "sigur ros"


def test_fold_maps_punctuation_to_space_and_collapses():
    assert fold("  The   Wall ") == "the wall"
    assert fold("Rock & Roll!") == "rock roll"
    assert fold("AC/DC") == "ac dc"


def test_fold_keeps_cyrillic_lowercased():
    assert fold("КиНо") == "кино"
    # NFKD folds ё→е and й→и — intentional, forgiving match.
    assert fold("Ёлка") == "елка"
    assert fold("Майк") == "маик"


def test_tokenize_drops_noise_tokens():
    assert tokenize("The Beatles feat. Someone") == ["the", "beatles", "someone"]
    assert tokenize("Eminem ft Dido") == ["eminem", "dido"]


def test_tokenize_splits_on_punctuation():
    assert tokenize("AC/DC") == ["ac", "dc"]
    assert tokenize("") == []


def test_translit_cyrillic_to_latin_variant():
    v = translit_variants("кино")
    assert "кино" in v and "kino" in v
    # fold collapses й→и first, so цой→цои→tsoi (translit is char-mapping, approximate).
    assert "tsoi" in translit_variants("цой")
    assert "shar" in translit_variants("шар")


def test_translit_latin_to_cyrillic_variant():
    assert "кино" in translit_variants("kino")
    assert "дом" in translit_variants("dom")


def test_translit_leaves_pure_token_when_no_cross_script():
    # A token with no letters stays as-is; a single script returns at least itself.
    assert "123" in translit_variants("123")


def test_analyze_flattens_tokens_and_translit_variants():
    bag = set(analyze("Кино"))
    assert {"кино", "kino"} <= bag
    assert analyze("") == []
    assert {"the", "wall"} <= set(analyze("The Wall"))
