"""Genius slug building — in particular '&' in the TITLE.

Genius spells the ampersand out everywhere in a slug, not just in the artist
part: ``genius.com/Coldplay-death-and-all-his-friends-lyrics``. The '&' was
being replaced for the artist only, so every song whose title carries one
(``Death & All His Friends``, ``Sex & Candy``) built a URL Genius answers 404
to — no description, no line annotations, no producer/label credits.
"""
from app.services.genius_service import build_genius_url


def test_ampersand_in_the_title_becomes_and():
    assert build_genius_url("Coldplay", "Death & All His Friends") == (
        "https://genius.com/Coldplay-death-and-all-his-friends-lyrics")


def test_ampersand_in_the_artist_still_becomes_and():
    assert build_genius_url("Simon & Garfunkel", "The Boxer") == (
        "https://genius.com/Simon-and-garfunkel-the-boxer-lyrics")


def test_ampersand_on_both_sides():
    assert build_genius_url("Florence & The Machine", "Sex & Candy") == (
        "https://genius.com/Florence-and-the-machine-sex-and-candy-lyrics")


def test_a_bare_ampersand_does_not_leave_a_double_dash():
    """'Me & You' — the '&' is its own word, so it must fold into one dash."""
    assert build_genius_url("Kanye West", "Me & You") == (
        "https://genius.com/Kanye-west-me-and-you-lyrics")


def test_titles_without_an_ampersand_are_untouched():
    assert build_genius_url("Dr. Dre", "Still D.R.E.") == (
        "https://genius.com/Dr-dre-still-dre-lyrics")
