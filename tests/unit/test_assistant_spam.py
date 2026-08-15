"""Adult and gambling hosts, out at the door.

The hosts here are the ones a real run returned for "Kanye West hit songs" —
a broken engine's output, fused in by rank so it interleaved one-for-one with
the genuine results.

The second half of the file is the part that matters more: a filter that eats
real music results is worse than the spam it removes, because the loss is
invisible. Hence hosts only, and hence the split between substrings that are
safe anywhere and labels that are not.
"""

import pytest

from app.services.assistant.web_sources import is_junk
from app.services.assistant.spam import is_spam_host, spam_report


class TestSpamHosts:
    @pytest.mark.parametrize("url", [
        "https://stripchat.app/",
        "https://stripchat.global/girls",
        "https://de.stripchat.global/girls",
        "https://stripchat.me/en/",
        "https://strip.chat/girls/german",
        "https://lemoncams.com/stripchat-cams",
        "https://www.playtech.com/",
        "https://www.investors.playtech.com/",
        "https://www.playtech.com/locations/estonia/",
    ])
    def test_the_hosts_from_the_real_run(self, url):
        assert is_spam_host(url)

    @pytest.mark.parametrize("url", [
        "https://pornhub.com/x", "https://chaturbate.com/",
        "https://1xbet.com/en", "https://casino-online.de/",
        "https://bet.com/", "https://poker.org/",
    ])
    def test_the_usual_suspects(self, url):
        assert is_spam_host(url)


class TestNoFalsePositives:
    """Every one of these is a page a music question could legitimately want."""

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/Kanye_West_singles_discography",
        "https://chart2000.com/tt/kanye_west.htm",
        "https://top40weekly.com/kanye-wests-top-songs/",
        "https://www.timeout.com/music/best-kanye-west-songs",
        "https://kworb.net/spotify/artist/5K4W6rqBFWDnAN6FQUkS6x_songs.html",
        "https://www.billboard.com/charts/hot-100/",
        "https://pitchfork.com/reviews/albums/",
    ])
    def test_the_good_results_from_the_same_run(self, url):
        assert not is_spam_host(url)

    @pytest.mark.parametrize("url", [
        # "bet" and "sex" as substrings live inside ordinary domains, so they
        # are matched as whole labels only.
        "https://www.betterhelp.com/",
        "https://arbeit.de/",
        "https://www.middlesex.gov.uk/",
        "https://essex-music.co.uk/",
        # "cams" inside "scams" is the one collision worth excluding by hand.
        "https://www.scamsafe.org/",
    ])
    def test_substrings_that_would_have_been_false_positives(self, url):
        assert not is_spam_host(url)

    def test_a_separator_does_not_hide_a_name(self):
        """"strip.chat" is "stripchat" with a dot in it."""
        assert is_spam_host("https://strip.chat/girls")
        assert is_spam_host("https://chatur-bate.example/")

    def test_a_song_called_casino_is_not_a_host(self):
        """The filter reads hosts, never titles — "Casino" is an album."""
        assert not is_spam_host("https://genius.com/albums/Casino")


class TestIngestionGate:
    def test_spam_is_dropped_by_the_same_gate_as_dead_ends(self):
        """One gate at the edge, before dedup and before the cross-encoder."""
        assert is_junk("https://stripchat.global/girls")
        assert is_junk("https://open.spotify.com/track/1")
        assert not is_junk("https://en.wikipedia.org/wiki/Kanye_West")

    def test_the_report_groups_by_host(self):
        counts = spam_report([
            "https://stripchat.global/girls",
            "https://stripchat.global/subscriptions",
            "https://www.playtech.com/",
            "https://en.wikipedia.org/wiki/Kanye_West",
        ])
        assert counts == {"stripchat.global": 2, "www.playtech.com": 1}

    def test_junk_input_is_safe(self):
        assert not is_spam_host("")
        assert not is_spam_host("not a url")
