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


class TestMarkupStripping:
    """Structure that used to survive into gems. Cases are verbatim from the
    2026-07-26 production snapshot (40 of 585 gems were built on these)."""

    @pytest.mark.parametrize("line", [
        "[Verse 1: Dr. Dre, Eminem, & (Eddie)]",     # nested brackets
        "[Chorus: Jay-Z & (Kanye West)]",
        "[Verse 2: Memphis Bleek (Beanie Sigel)]",
        "'''Charlie Wilson and Brenda Lee:'''",      # wiki bold
        "'''Daft Punk:'''",
        "Verse Two: Foxy Brown",                     # speaker / role labels
        "Lead Vocals : Freddie Mercury",
        "Artist : Rihanna feat. DrakeTitle : Work",  # provider metadata header
        "*Instrumental for What's the Difference by Dr. Dre plays in the background*",
        "作词 : Kevin Parker",                        # CJK provider credit header
        "作曲 : Curtis Jackson & Khari Cain",
    ])
    def test_structural_lines_detected(self, line):
        assert preprocess.is_structural_line(line)

    @pytest.mark.parametrize("line,expected", [
        # A real Jay-Z line with a colon — lowercase words, so not a label.
        ("My portfolio reads: leads to Don Corleone, nigga please",
         "My portfolio reads: leads to Don Corleone, nigga please"),
        # Personal speaker label: peeled, the sung words stay.
        ("Raekwon: Yo, Meth, hold up, hold up", "Yo, Meth, hold up, hold up"),
        ("[01:28.29][02:47.78]Romeo and Juliet they never felt this way I bet",
         "Romeo and Juliet they never felt this way I bet"),
    ])
    def test_sung_lines_kept(self, line, expected):
        assert not preprocess.is_structural_line(line)
        assert preprocess.strip_line_markup(line).strip() == expected

    def test_speaker_name_cannot_be_quoted(self):
        assert preprocess.find_quote("Raekwon", "Raekwon: Yo, Meth, hold up") is None
        assert preprocess.find_quote("Dr. Dre", "[Verse 1: Dr. Dre, Eminem, & (Eddie)]") is None


class TestNewlineNormalization:
    def test_carriage_returns_split_into_lines(self):
        lyr = "Wake up\rGonna wake up to nothing\rBreak up\rThe break up is coming"
        assert preprocess.find_quote("Wake Up", lyr) == "Wake up"

    def test_blob_longer_than_a_line_is_not_a_quote(self):
        blob = "If you ain't got no money take your broke ass home " * 4 + "Ludacris"
        assert preprocess.find_quote("Ludacris", blob) is None

    def test_whole_word_only(self):
        assert preprocess.find_quote("Wee", "Caught between the moon") is None
        assert preprocess.find_quote("Wee", "Sampled from Wee tonight") is not None


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

    def test_single_word_title_rejected(self):
        cat = matching.build_song_catalog([
            {"album": "Relapse", "title": "Beautiful", "artist": "Eminem"},
        ])
        assert matching.match_songref("Relapse", 0.99, cat, "X", "Y") is None

    def test_score_bar_is_higher_than_other_kinds(self):
        cat = self._catalog()
        assert matching.match_songref("The Blueprint", 0.7, cat, "Off That", "Z") is None

    def test_ambiguous_title_rejected(self):
        cat = matching.build_song_catalog([
            {"album": "A", "title": "Hey Mama", "artist": "Kanye West"},
            {"album": "Elephunk", "title": "Hey Mama", "artist": "The Black Eyed Peas"},
        ])
        assert cat[matching.norm_name("Hey Mama")]["ambiguous"] is True
        assert matching.match_songref("Hey Mama", 0.99, cat, "X", "Y") is None


class TestCitationMarker:
    """The gate that separates a named work from a coincidental phrase.

    Cases are verbatim from the 2026-07-26 production snapshot.
    """

    @pytest.mark.parametrize("span,quote", [
        ("Jesus Walks", 'I made "Jesus Walks," I\'m never going to hell'),
        ("Guilty Conscience", 'Throw on "Guilty Conscience" at concerts'),
        ("Guilty Conscience", 'The song Guilty Conscience has gotten such rotten responses"'),
        ("Friend Or Foe", 'I know you heard "Friend or Foe," this ain\'t different from that'),
        ("State of the Art", "Since 'State of the Art' we've stayed in the yard"),
        ("Promiscuous", 'love my ass and my abs in the video of "Promiscuous"'),
        ("Jesus of Suburbia", 'This next song\'s called "Jesus of Suburbia"'),
    ])
    def test_accepts_cited_titles(self, span, quote):
        assert matching.has_citation_marker(span, quote)

    @pytest.mark.parametrize("span,quote", [
        ("Wake Up", "Wake up"),                                   # The Black Keys — Sister
        ("i love you", "'cos I love you"),                        # Phil Collins
        ("Pump It", "Juke it (mo') Pump it (mo')"),               # Kanye West
        ("Don't Forget Me", "Don't forget me"),                   # Lana Del Rey
        ("Relapse", "Encore, I was on drugs; Relapse, I was flushin' 'em out"),
    ])
    def test_rejects_bare_phrases(self, span, quote):
        assert not matching.has_citation_marker(span, quote)

    def test_absent_span_is_not_a_citation(self):
        assert not matching.has_citation_marker("Renegade", 'I made "Jesus Walks"')


class TestSongrefProductionRegressions:
    """Every one of these produced a bogus gem on prod before 2026-07-26."""

    def _cat(self, rows):
        return matching.build_song_catalog(rows)

    @pytest.mark.parametrize("own_title,ref,quote", [
        # Bracketed tail hid the self-reference: catalog key was stripped,
        # own-title key was not.
        ("Las Palabras de Amor (The Words of Love)", "Las Palabras de Amor", "Las palabras de amor"),
        ("The Lost Chord (ft. Leee John)", "The Lost Chord", "The lost chord"),
        ("Where'd You Go (feat. Holly Brook)", "Where'd You Go", "Where'd you go? I miss you so"),
        ("All Around The World (Feat. LaToiya Williams)", "All Around The World", "All around the world (Oh)"),
        # Contained in the track's own title.
        ("Give Me Novacaine (Live at Irving Plaza)", "Novacaine", 'This next song\'s called "Novacaine"'),
    ])
    def test_self_reference_rejected(self, own_title, ref, quote):
        cat = self._cat([{"album": "Album", "title": own_title, "artist": "A"}])
        assert matching.match_songref(ref, 0.99, cat, own_title, "Album", quote=quote) is None

    def test_wake_up_phrase_rejected(self):
        cat = self._cat([{"album": "The Dutchess", "title": "Wake Up", "artist": "Fergie"}])
        gem = matching.match_songref("Wake Up", 0.99, cat, "Sister", "El Camino", quote="Wake up")
        assert gem is None

    def test_real_reference_survives(self):
        cat = self._cat([{"album": "The College Dropout", "title": "Jesus Walks", "artist": "Kanye West"}])
        gem = matching.match_songref(
            "Jesus Walks", 0.88, cat, "Otis", "Watch the Throne",
            quote='I made "Jesus Walks," I\'m never going to hell',
        )
        assert gem is not None and gem["canonical"] == "Jesus Walks"


class TestNamedropSelfReference:
    """Aliases resolve to the performer — production cases that leaked."""

    def _idx(self):
        return matching.build_artist_index(
            {"eminem": "Eminem", "kanye-west": "Kanye West", "kobe": "Kobe",
             "bad-meets-evil": "Bad Meets Evil", "dr-dre": "Dr. Dre"},
            {"marshall mathers": "eminem", "pablo": "kanye-west"},
        )

    def test_alias_of_own_artist_rejected_by_slug(self):
        idx = self._idx()
        own = matching.own_name_keys("Eminem", ["Eminem"], "If I Had")
        gem, _ = matching.match_namedrop(
            "Marshall Mathers", 0.9, idx, own, "If I Had",
            own_slugs=frozenset({"eminem"}),
        )
        assert gem is None

    def test_group_member_rejected_via_kin_slugs(self):
        idx = self._idx()
        own = matching.own_name_keys("Bad Meets Evil", ["Bad Meets Evil"], "Fast Lane")
        # kin widening happens in the task; here it arrives as own_slugs
        gem, _ = matching.match_namedrop(
            "Eminem", 0.96, idx, own, "Fast Lane",
            own_slugs=frozenset({"bad-meets-evil", "eminem"}),
        )
        assert gem is None

    def test_rare_one_word_artist_goes_to_llm(self):
        idx = self._idx()
        own = matching.own_name_keys("Jay-Z", ["Jay-Z"], "Venus Vs. Mars")
        gem, needs_llm = matching.match_namedrop(
            "Kobe", 0.88, idx, own, "Venus Vs. Mars", rare_slugs=frozenset({"kobe"}),
        )
        assert gem is not None and needs_llm is True

    def test_well_stocked_artist_still_auto_accepts(self):
        idx = self._idx()
        own = matching.own_name_keys("Jay-Z", ["Jay-Z"], "Hola Hovito")
        gem, needs_llm = matching.match_namedrop(
            "Dr. Dre", 0.9, idx, own, "Hola Hovito", rare_slugs=frozenset({"kobe"}),
        )
        assert gem is not None and needs_llm is False


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
