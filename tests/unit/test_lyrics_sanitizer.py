"""Lyrics sanitizer — provider artifacts out, lyrics untouched.

Every artifact below is verbatim from the 2026-07-26 production Qdrant
payloads (main library, 5474 tracks with lyrics).
"""
import pytest

from app.indexing.lyrics_sanitizer import sanitize_lyrics

pytestmark = pytest.mark.unit


class TestNewlines:
    """1115 of 5474 tracks (20%) used bare CR — the player splits on \\n only,
    so those rendered as one unbroken paragraph."""

    def test_bare_cr_becomes_newline(self):
        out = sanitize_lyrics("Wake up\rGonna wake up to nothing\rBreak up")
        assert out.split("\n") == ["Wake up", "Gonna wake up to nothing", "Break up"]

    def test_crlf_becomes_newline(self):
        assert sanitize_lyrics("one\r\ntwo") == "one\ntwo"

    def test_blank_runs_collapse(self):
        assert sanitize_lyrics("one\n\n\n\n\ntwo") == "one\n\ntwo"


class TestCjkCredits:
    @pytest.mark.parametrize("line", [
        "作词 : Billie Joe Armstrong/Mike Dirnt/Tre Cool",
        "作曲 : Shakira/Tim Mitchell",
        "编曲 : 泪鑫",
        "制作人 : Green Day/Rob Cavallo",
        "混音师 : Chris Lord-Alge",
        "母带工程师 : Ted Jensen",
        "吉他 : Josh Klinghoffer",
        "作词：Elton John",          # full-width colon
    ])
    def test_credit_header_dropped(self, line):
        assert sanitize_lyrics(f"{line}\nReal first line") == "Real first line"

    def test_whole_credit_block_dropped(self):
        raw = (
            "作词 : Rory Graham/Dan Priddy\n"
            "作曲 : Rory Graham/Dan Priddy\n"
            "制作人 : Mark Crew\n"
            "\n"
            "I was born in the summer"
        )
        assert sanitize_lyrics(raw) == "I was born in the summer"

    def test_credit_block_hidden_behind_cr_is_still_dropped(self):
        """The reason newline normalization has to come first."""
        raw = "作词 : Foy Vance/Rory Graham\r作曲 : Foy Vance/Rory Graham\rOdetta, oh"
        assert sanitize_lyrics(raw) == "Odetta, oh"

    def test_ordinary_cjk_lyrics_survive(self):
        raw = "月亮代表我的心\n你问我爱你有多深"
        assert sanitize_lyrics(raw) == raw


class TestTimestamps:
    @pytest.mark.parametrize("raw,expected", [
        ("[00:12.34]I'm at a payphone", "I'm at a payphone"),
        ("[1:05]Short form", "Short form"),
        ("[00:01:02.500]Hour form", "Hour form"),
        ("[01:28.29][02:47.78]Romeo and Juliet", "Romeo and Juliet"),
    ])
    def test_timestamps_stripped(self, raw, expected):
        assert sanitize_lyrics(raw) == expected

    def test_section_headers_are_not_timestamps(self):
        """[Verse 1: Eminem] is structure; consumers that care strip it."""
        raw = "[Verse 1: Eminem]\nReal line"
        assert sanitize_lyrics(raw) == raw

    def test_bracketed_words_survive(self):
        raw = "She said [laughs] nothing at all"
        assert sanitize_lyrics(raw) == raw


class TestPassthrough:
    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_passes_through(self, value):
        assert sanitize_lyrics(value) == value

    def test_nothing_left_becomes_none(self):
        """Callers keep their existing "no lyrics" fallback."""
        assert sanitize_lyrics("作词 : X\n作曲 : Y\n\n") is None

    def test_clean_lyrics_are_untouched(self):
        raw = "Got you a beeper to feel important\nAin't no nigga like the one I got"
        assert sanitize_lyrics(raw) == raw
