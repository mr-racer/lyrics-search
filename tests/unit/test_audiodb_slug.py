"""Tests for canonical-artist-name and AudioDB URL slug helpers."""
import pytest

from app.services.audiodb_service import _canonical_artist_name, _audiodb_slug


@pytest.mark.parametrize("input_name,expected", [
    ("Kanye West", "Kanye West"),
    ("Dua Lipa feat. Angele", "Dua Lipa"),
    ("Dua Lipa feat Angele", "Dua Lipa"),                    # no period
    ("Kanye West ft. Jay-Z & Rihanna", "Kanye West"),
    ("Kanye West ft Jay-Z", "Kanye West"),
    ("Sufjan Stevens featuring Bryce Dessner", "Sufjan Stevens"),
    ("Tyler, the Creator", "Tyler, the Creator"),            # no feat — unchanged
    ("21 Savage", "21 Savage"),
    ("Florence + the Machine", "Florence + the Machine"),
    ("AC/DC", "AC/DC"),
])
def test_canonical_artist_name(input_name, expected):
    assert _canonical_artist_name(input_name) == expected


@pytest.mark.parametrize("canonical,expected_slug", [
    ("Kanye West", "kanye+west"),
    ("Dua Lipa", "dua+lipa"),
    ("AC/DC", "acdc"),
    ("Tyler, the Creator", "tyler+the+creator"),
    ("Guns N' Roses", "guns+n+roses"),
    ("21 Savage", "21+savage"),
    ("Florence + the Machine", "florence+the+machine"),
    ("The Beatles", "the+beatles"),
])
def test_audiodb_slug(canonical, expected_slug):
    assert _audiodb_slug(canonical) == expected_slug
