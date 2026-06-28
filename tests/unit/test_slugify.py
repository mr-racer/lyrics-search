"""Tests for _slugify() used in artist_facts_service and song_facts_service."""

import pytest

from app.services.artist_facts_service import _slugify as artist_slugify
from app.services.song_facts_service import _slugify as song_slugify


class TestSlugify:
    """Both services share the same slugify logic."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("The Weeknd", "the-weeknd"),
            ("Kendrick", "kendrick"),
            ("A B C", "a-b-c"),
            ("already-lowercase", "already-lowercase"),
            ("Billie Eilish", "billie-eilish"),
            ("Daft Punk", "daft-punk"),
            ("SingleWord", "singleword"),
        ],
    )
    def test_artist_slugify(self, input_name, expected):
        assert artist_slugify(input_name) == expected

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("Blinding Lights", "blinding-lights"),
            ("Hello", "hello"),
            ("1-800-273-8255", "1-800-273-8255"),
            ("Cruel Summer", "cruel-summer"),
        ],
    )
    def test_song_slugify(self, input_name, expected):
        assert song_slugify(input_name) == expected

    def test_both_slugify_identical_behavior(self):
        """Both implementations should produce the same output."""
        samples = ["The Weeknd", "Adele", "Billie Eilish", "Tame Impala"]
        for s in samples:
            assert artist_slugify(s) == song_slugify(s)


class TestSongFactsPrimaryArtist:
    """Song facts must query/cache under the PRIMARY artist for collab tags —
    songfacts.com lists a song once, under its primary performer."""

    def test_cache_key_uses_primary_artist(self):
        from app.services.song_facts_service import get_song_facts_key
        # "DNCE, Nicki Minaj" → primary is DNCE
        assert get_song_facts_key("DNCE, Nicki Minaj", "Kissing Strangers") == \
            "dnce-kissing-strangers"

    def test_single_artist_key_unchanged(self):
        from app.services.song_facts_service import get_song_facts_key
        assert get_song_facts_key("Drake", "Hotline Bling") == "drake-hotline-bling"

    def test_url_uses_primary_artist(self, mocker):
        from app.services import song_facts_service as sf

        captured = {}

        def fake_get(url, timeout=None, proxies=None):
            captured["url"] = url
            raise sf.requests.RequestException("stop before network")

        mocker.patch.object(sf.requests, "get", side_effect=fake_get)
        sf._fetch_song_facts_html("DNCE, Nicki Minaj", "Kissing Strangers")
        assert "/facts/dnce/" in captured["url"]
        assert "dnce-nicki-minaj" not in captured["url"]
