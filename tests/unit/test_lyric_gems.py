"""Unit tests for the lyric-gems pipeline (preprocess / gazetteers / matching /
resolver JSON parsing). Pure Python — no GLiNER, no LLM, no network."""

import pytest

from app.services.lyric_gems import matching, preprocess
from app.services.lyric_gems.gazetteers import capsule_hits
from app.services.lyric_gems.resolver import parse_strict_json

pytestmark = pytest.mark.unit


# ── preprocess ───────────────────────────────────────────────────────────────

class TestCleanLyrics:
    def test_strips_section_headers(self):
        raw = "[Verse 1: Eminem]\nReal line one\n[Chorus: Rihanna]\nReal line two"
        cleaned = preprocess.clean_lyrics(raw)
        assert "Eminem" not in cleaned
        assert "Rihanna" not in cleaned
        assert "Real line one" in cleaned and "Real line two" in cleaned

    def test_strips_hook_brace_headers(self):
        cleaned = preprocess.clean_lyrics("(Hook){50 Cent}\nActual lyric")
        assert "50 Cent" not in cleaned
        assert "Actual lyric" in cleaned

    def test_strips_tracklist_lines(self):
        raw = "Real lyric\n14. Apologize (Timbaland Remix) [0:03:04.82]"
        cleaned = preprocess.clean_lyrics(raw)
        assert "Timbaland" not in cleaned
        assert "Real lyric" in cleaned

    def test_keeps_normal_lines_verbatim(self):
        raw = "Timbaland, let me spit my prose on"
        assert preprocess.clean_lyrics(raw) == raw


class TestCyrRatio:
    def test_english(self):
        assert preprocess.cyr_ratio("hello world") == 0.0

    def test_russian(self):
        assert preprocess.cyr_ratio("привет мир") == 1.0

    def test_empty(self):
        assert preprocess.cyr_ratio("12345 !!!") == 0.0


class TestChunking:
    def test_respects_max_chunks(self):
        text = "\n".join(["line " * 30] * 400)
        chunks = preprocess.chunk_text(text, chunk_chars=500, max_chunks=3)
        assert len(chunks) == 3


class TestFindQuote:
    def test_finds_line(self):
        lyr = "First line\nGot you a beeper to feel important\nLast line"
        assert preprocess.find_quote("beeper", lyr) == "Got you a beeper to feel important"

    def test_never_quotes_section_header(self):
        lyr = "[Verse: Eminem]\nno mention here"
        assert preprocess.find_quote("Eminem", lyr) is None

    def test_case_insensitive(self):
        assert preprocess.find_quote("MYSPACE", "rip sign on your myspace page") is not None


class TestAppearsCapitalized:
    def test_capitalized(self):
        assert preprocess.appears_capitalized("sidekick", "shorty is a Sidekick")

    def test_lowercase_only(self):
        assert not preprocess.appears_capitalized("bizarre", "the feeling is bizarre")


# ── capsule gazetteer ────────────────────────────────────────────────────────

class TestCapsuleHits:
    def test_pager_hit_with_quote(self):
        lyr = "Got you a beeper to feel important"
        hits = capsule_hits(lyr, lyr, "Ain't No Nigga")
        assert any(h["canonical"] == "pager" for h in hits)
        hit = next(h for h in hits if h["canonical"] == "pager")
        assert hit["quote"] == lyr
        assert hit["detail"]["ru"] == "пейджер"

    def test_two_way_needs_pager_context(self):
        highway = "Behind these two-way highway lines"
        assert capsule_hits(highway, highway, "Jet Pack Blues") == []
        pager = "My two-way pager is lookin' for some danger"
        hits = capsule_hits(pager, pager, "Creamer")
        assert any(h["detail"]["item"] == "two-way" for h in hits)

    def test_title_mention_is_not_a_surprise(self):
        lyr = "I'm at a payphone trying to call home"
        assert capsule_hits(lyr, lyr, "Payphone") == []

    def test_brandlike_needs_capitalization(self):
        fruit = "picking blackberry jam in the garden"
        assert capsule_hits(fruit, fruit, "Jam") == []
        phone = "If Hov's a Blackberry Bold, then shorty is a Sidekick"
        hits = capsule_hits(phone, phone, "Venus Vs. Mars")
        assert {h["canonical"] for h in hits} >= {"BlackBerry", "Sidekick"}


# ── matching ─────────────────────────────────────────────────────────────────

def _index():
    artists = {
        "dr-dre": "Dr. Dre",
        "jay-z": "JAY‐Z",   # non-ASCII hyphen on purpose (real payload data)
        "the-game": "The Game",
        "timbaland": "Timbaland",
    }
    aliases = {"timbo": "timbaland", "hov": "jay-z"}
    return matching.build_artist_index(artists, aliases)


class TestNamedropMatching:
    def test_plain_hit(self):
        idx = _index()
        own = matching.own_name_keys("Snoop Dogg", ["Snoop Dogg"], "For All My Niggaz")
        gem, needs_llm = matching.match_namedrop("Dr. Dre", 0.9, idx, own, "For All My Niggaz")
        assert gem is not None and gem["detail"]["artist_slug"] == "dr-dre"
        assert needs_llm is False

    def test_unicode_dash_normalization(self):
        idx = _index()
        own = matching.own_name_keys("R. Kelly", ["R. Kelly"], "Honey")
        gem, _ = matching.match_namedrop("Jay-Z", 0.9, idx, own, "Honey")
        assert gem is not None and gem["detail"]["artist_slug"] == "jay-z"

    def test_alias_resolves(self):
        idx = _index()
        own = matching.own_name_keys("X", ["X"], "Y")
        gem, _ = matching.match_namedrop("Timbo", 0.8, idx, own, "Y")
        assert gem is not None and gem["canonical"] == "Timbaland"

    def test_self_and_feat_excluded(self):
        idx = _index()
        own = matching.own_name_keys("JAY‐Z", ["JAY‐Z", "Timbaland"], "Big Pimpin'")
        assert matching.match_namedrop("Jay-Z", 0.99, idx, own, "Big Pimpin'")[0] is None
        assert matching.match_namedrop("Timbaland", 0.99, idx, own, "Big Pimpin'")[0] is None

    def test_title_mention_excluded(self):
        idx = _index()
        own = matching.own_name_keys("Jay-Z", ["Jay-Z"], "Renegade (feat. Dr. Dre)")
        assert matching.match_namedrop("Dr. Dre", 0.99, idx, own, "Renegade (feat. Dr. Dre)")[0] is None

    def test_generic_name_needs_llm(self):
        idx = _index()
        own = matching.own_name_keys("Kanye West", ["Kanye West"], "Eazy")
        gem, needs_llm = matching.match_namedrop("The Game", 0.9, idx, own, "Eazy")
        assert gem is not None and needs_llm is True

    def test_low_score_dropped(self):
        idx = _index()
        own = matching.own_name_keys("X", ["X"], "Y")
        assert matching.match_namedrop("Dr. Dre", 0.4, idx, own, "Y")[0] is None

    def test_stopword_dropped(self):
        idx = _index()
        own = matching.own_name_keys("X", ["X"], "Y")
        assert matching.match_namedrop("baby", 0.99, idx, own, "Y")[0] is None


class TestSongrefMatching:
    def _catalog(self):
        return matching.build_song_catalog([
            {"album": "The Blueprint", "title": "Renegade", "artist": "Jay-Z"},
            {"album": "Home", "title": "Wow.", "artist": "Post Malone"},  # both too generic
        ])

    def test_album_hit(self):
        cat = self._catalog()
        gem = matching.match_songref("The Blueprint", 0.8, cat, "Off That", "The Blueprint 3")
        assert gem is not None and gem["detail"]["ref_kind"] == "album"

    def test_generic_titles_never_enter_catalog(self):
        cat = self._catalog()
        assert matching.match_songref("Home", 0.99, cat, "X", "Y") is None
        assert matching.match_songref("Wow.", 0.99, cat, "X", "Y") is None

    def test_own_album_excluded(self):
        cat = self._catalog()
        assert matching.match_songref("The Blueprint", 0.9, cat, "Song", "The Blueprint") is None


class TestPopcultureMatching:
    def test_known_character(self):
        gem, needs_llm = matching.match_popculture("Tony Montana", 0.9)
        assert gem is not None and gem["canonical"] == "Tony Montana"
        assert needs_llm is False

    def test_alias_maps_to_canonical(self):
        gem, _ = matching.match_popculture("Bruce Wayne", 0.9)
        assert gem is not None and gem["canonical"] == "Batman"
        assert gem["detail"]["ru"] == "Бэтмен"

    def test_unknown_confident_goes_to_llm(self):
        gem, needs_llm = matching.match_popculture("Beetlejuice", 0.9)
        assert gem is None and needs_llm is True

    def test_unknown_low_score_dropped(self):
        gem, needs_llm = matching.match_popculture("Beetlejuice", 0.65)
        assert gem is None and needs_llm is False


# ── resolver JSON parsing ────────────────────────────────────────────────────

class TestParseStrictJson:
    def test_plain(self):
        assert parse_strict_json('{"verdict":"yes","canonical":"X","why":"w"}') == {
            "verdict": "yes", "canonical": "X", "why": "w",
        }

    def test_fenced_and_prosy(self):
        raw = 'Sure! Here is the answer:\n```json\n{"verdict":"no","canonical":"","why":"nope"}\n```'
        assert parse_strict_json(raw)["verdict"] == "no"

    def test_garbage_is_none(self):
        assert parse_strict_json("I think it is probably fine") is None
        assert parse_strict_json("") is None
        assert parse_strict_json('{"broken": ') is None
