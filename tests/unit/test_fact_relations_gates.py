"""Gates over classified sample links.

Every case here is a row that existed in production on 2026-07-26 — the
snapshot that motivated this pass. The expected outcome is what the user
should see instead.
"""
import pytest

from app.services.fact_relations.gates import (
    MAX_LINKS_PER_DIRECTION,
    SongContext,
    apply_gates,
    dst_key,
    fact_mentions_sampling,
    gate_claim,
    links_to_relations,
)
from app.services.fact_relations.library_index import LibraryIndex

pytestmark = pytest.mark.unit


def claim(song=None, artist=None, relation="sample", direction="source", **kw):
    return {"song": song, "artist": artist, "relation": relation,
            "direction": direction, **kw}


# ── the lexical prefilter ────────────────────────────────────────────────────

class TestSampleLexicon:
    @pytest.mark.parametrize("fact", [
        'The song is built around a sample of "Bound", a 1971 Soul tune by Ponderosa Twins Plus One.',
        'Needlz sampled "Shakespeare\'s Sonnet" (The Mirror of the Soul, 1977) by Alla Pugacheva',
        "both containing interpolation of Wee's \"Aeroplane (Reprise)\"",
        'This sample is from Brenda Lee\'s 1959 recording "Sweet Nothings."',
        "the soul instrumental and background vocals are sampled from the Ponderosa Twins Plus One's 1971 song",
    ])
    def test_accepts_real_sampling_facts(self, fact):
        assert fact_mentions_sampling(fact)

    @pytest.mark.parametrize("fact", [
        # Adele — "Hello": three bogus samples came out of this one fact.
        "The song's first line matches the melody of Lionel Richie's iconic 1983 line, "
        '"Hello, is it me you\'re looking for?" from his own "Hello".',
        # 50 Cent — "Piggy Bank": a diss, not a sample.
        'Referring to Kelis\'s chorus in her song "Milkshake": My milkshake brings all the boys to the yard',
        'Reference to DMX\'s song "Get At Me Dog" that was released in 1998.',
        # Kanye — "Bound 2": explaining an allusion in a line.
        'Kanye alludes to Biggie\'s classic lyric from "Juicy"',
        # Kanye — "Jesus Walks": a shared slogan, no audio taken.
        'George Clinton\'s "One Nation Under a Groove" also quotes the slogan',
    ])
    def test_rejects_reference_facts(self, fact):
        assert not fact_mentions_sampling(fact)


# ── single-claim gates ───────────────────────────────────────────────────────

class TestRelationGate:
    @pytest.mark.parametrize("relation", ["sample", "interpolation"])
    def test_sound_relations_pass(self, relation):
        ctx = SongContext(title="Bound 2", artist="kanye-west")
        link, reason = gate_claim(
            claim("Bound", "Ponderosa Twins Plus One", relation=relation), ctx)
        assert reason is None and link["relation"] == relation

    @pytest.mark.parametrize("relation,case", [
        ("lyrical_reference", "50 Cent/Piggy Bank -> Kelis/Milkshake"),
        ("inspiration", "Adele/Hello -> Lionel Richie/Hello"),
        ("remix", "Sabrina Carpenter/Espresso <- Working Late Remix"),
        ("cover", "Bee Gees/How Can You Mend... <- Madonna"),
        ("other", "an unproven lawsuit"),
    ])
    def test_non_sound_relations_rejected(self, relation, case):
        ctx = SongContext(title="T", artist="a")
        link, reason = gate_claim(claim("S", "A", relation=relation), ctx)
        assert link is None and reason == f"relation:{relation}", case


class TestSelfReference:
    def test_bound_2_sampled_by_kanyes_own_track(self):
        """kanye-west/Bound 2 (2013) <- Illest Motherfucker Alive (2011)."""
        ctx = SongContext(title="Bound 2", artist="kanye-west", year=2013)
        link, reason = gate_claim(
            claim("Illest Motherfucker Alive", "Kanye West", direction="usage"), ctx)
        assert link is None and reason == "self-artist"

    def test_alias_of_the_performer_loses(self):
        """kanye-west/New Slaves <- "Jesus Walks", Ye."""
        ctx = SongContext(
            title="New Slaves", artist="kanye-west", year=2013,
            artist_keys=frozenset({"kanye west", "ye"}),
        )
        link, reason = gate_claim(claim("Jesus Walks", "Ye", direction="usage"), ctx)
        assert link is None and reason == "self-artist"

    def test_eminem_mosh_samples_his_own_square_dance(self):
        ctx = SongContext(title="Mosh", artist="eminem", year=2004)
        link, reason = gate_claim(claim("Square Dance", "Eminem"), ctx)
        assert link is None and reason == "self-artist"

    def test_radiohead_videotape_and_its_own_album(self):
        ctx = SongContext(title="Videotape", artist="radiohead", year=2007)
        link, reason = gate_claim(
            claim("The King of Limbs", "Radiohead", direction="usage"), ctx)
        assert link is None and reason == "self-artist"

    def test_song_referring_to_itself(self):
        ctx = SongContext(title="Espresso", artist="sabrina-carpenter")
        link, reason = gate_claim(claim("Espresso", "Someone Else"), ctx)
        assert link is None and reason == "self-title"


class TestMissingSides:
    def test_bare_title_rejected(self):
        """metallica/Don't Tread on Me -> "America" with no artist matched
        Imagine Dragons (2013) purely on the title."""
        ctx = SongContext(title="Don't Tread on Me", artist="metallica", year=1991)
        link, reason = gate_claim(claim("America", None), ctx)
        assert link is None and reason == "no-artist"

    def test_usage_without_a_song_rejected(self):
        """bee-gees/How Can You Mend... <- {song: None, artist: Madonna}."""
        ctx = SongContext(title="How Can You Mend a Broken Heart", artist="bee-gees")
        link, reason = gate_claim(claim(None, "Madonna", direction="usage"), ctx)
        assert link is None and reason == "no-song"

    def test_source_without_a_song_is_a_sampled_voice(self):
        ctx = SongContext(title="T", artist="a")
        link, reason = gate_claim(claim(None, "James Brown"), ctx)
        assert reason is None and link["dst_artist"] == "James Brown"


class TestLibraryGates:
    def _lib(self):
        return LibraryIndex(
            tracks=[
                {"title": "Cool", "artist": "Dua Lipa", "album": "Future Nostalgia", "year": 2021},
                {"title": "After Hours", "artist": "The Weeknd", "album": "After Hours", "year": 2020},
                # The same song twice: an original and a compilation copy. The
                # index must report the EARLIER year or the remaster wins.
                {"title": "Stayin' Alive", "artist": "Bee Gees", "album": "Saturday Night Fever", "year": 1977},
                {"title": "Stayin' Alive", "artist": "Bee Gees", "album": "Greatest Hits", "year": 2017},
            ],
            songs=[{"slug": "dua-lipa-cool", "title": "Cool", "artist_slug": "dua-lipa"}],
        )

    def test_source_from_the_future_rejected(self):
        """michael-jackson/Bad (1987) -> Dua Lipa/Cool (2021)."""
        ctx = SongContext(title="Bad", artist="michael-jackson", year=1987, library=self._lib())
        link, reason = gate_claim(claim("Cool", "Dua Lipa"), ctx)
        assert link is None and reason == "year"

    def test_same_album_rejected(self):
        """the-weeknd/Hardest to Love <- After Hours, same record."""
        ctx = SongContext(
            title="Hardest to Love", artist="the-weeknd", year=2020,
            album="After Hours", library=self._lib(),
        )
        link, reason = gate_claim(
            claim("After Hours", "The Weeknd", direction="usage"), ctx)
        # self-artist fires first here, which is also correct — assert it dies.
        assert link is None and reason in ("self-artist", "same-album")

    def test_remaster_year_does_not_kill_a_real_sample(self):
        """A 1996 track sampling Bee Gees must survive the 2017 compilation copy."""
        ctx = SongContext(title="Virtual Insanity", artist="jamiroquai", year=1996,
                          library=self._lib())
        link, reason = gate_claim(claim("Stayin' Alive", "Bee Gees"), ctx)
        assert reason is None and link["dst_year"] == 1977

    def test_resolved_target_gets_its_slug(self):
        ctx = SongContext(title="T", artist="a", year=2024, library=self._lib())
        link, _ = gate_claim(claim("Cool", "Dua Lipa"), ctx)
        assert link["dst_slug"] == "dua-lipa-cool"

    def test_equal_years_are_allowed(self):
        ctx = SongContext(title="T", artist="a", year=2021, library=self._lib())
        link, reason = gate_claim(claim("Cool", "Dua Lipa"), ctx)
        assert reason is None and link is not None


# ── whole-song gates ─────────────────────────────────────────────────────────

class TestApplyGates:
    def test_punctuation_variants_collapse(self):
        """Bound 2 stored both "Sweet Nothin's" and "Sweet Nothings"."""
        assert dst_key("Brenda Lee", "Sweet Nothin's") == dst_key("Brenda Lee", "Sweet Nothings")
        ctx = SongContext(title="Bound 2", artist="kanye-west")
        links, _ = apply_gates([
            claim("Sweet Nothin's", "Brenda Lee"),
            claim("Sweet Nothings", "Brenda Lee"),
        ], ctx)
        assert len(links) == 1

    def test_bracketed_tail_collapses(self):
        assert dst_key("Wee", "Aeroplane (Reprise)") == dst_key("Wee", "Aeroplane")

    def test_both_directions_at_once_drops_both(self):
        ctx = SongContext(title="T", artist="a")
        links, rejected = apply_gates([
            claim("X", "Artist B", direction="source"),
            claim("X", "Artist B", direction="usage"),
        ], ctx)
        assert links == []
        assert {r for _c, r in rejected} == {"direction-echo"}

    def test_cap_per_direction(self):
        ctx = SongContext(title="T", artist="a")
        many = [claim(f"Song {i}", f"Artist {i}", confidence=i / 100)
                for i in range(MAX_LINKS_PER_DIRECTION + 4)]
        links, rejected = apply_gates(many, ctx)
        assert len(links) == MAX_LINKS_PER_DIRECTION
        assert sum(1 for _c, r in rejected if r == "cap") == 4
        # Highest confidence survives.
        assert links[0]["dst_title"] == f"Song {MAX_LINKS_PER_DIRECTION + 3}"

    def test_bound_2_production_case(self):
        """The real eleven claims stored for kanye-west/Bound 2 -> three links.

        Ponderosa Twins Plus One, Wee and Brenda Lee are the actual sources;
        everything else came from annotations about lines, slang and allusions.
        """
        ctx = SongContext(
            title="Bound 2", artist="kanye-west", year=2013, album="Yeezus",
            artist_keys=frozenset({"kanye west", "ye", "kanye"}),
        )
        claims = [
            claim("Bound", "Ponderosa Twins Plus One"),
            claim("Aeroplane (Reprise)", "Wee", relation="interpolation"),
            claim("Sweet Nothin's", "Brenda Lee"),
            claim("Sweet Nothings", "Brenda Lee"),                       # dup
            claim("Aeroplane", "Wee"),                                    # dup
            claim(None, "Suicide"),                                       # no song, source: kept?
            claim("MOR, R&B", None),                                      # no artist
            claim("Juicy", "Biggie", relation="lyrical_reference"),
            claim("Dutty Wine", "Tony Matterhorn", relation="lyrical_reference"),
            claim("Bound to fall in love", None),                         # a lyric, not a song
            claim("Illest Motherfucker Alive", "Kanye West", direction="usage"),
        ]
        links, rejected = apply_gates(claims, ctx)
        titles = {ln["dst_title"] for ln in links}
        assert "Bound" in titles
        assert "Aeroplane (Reprise)" in titles
        assert titles & {"Sweet Nothin's", "Sweet Nothings"}
        assert "Juicy" not in titles and "Dutty Wine" not in titles
        assert "Illest Motherfucker Alive" not in titles
        assert "MOR, R&B" not in titles and "Bound to fall in love" not in titles
        reasons = {r for _c, r in rejected}
        assert {"relation:lyrical_reference", "no-artist", "self-artist"} <= reasons

    def test_links_to_relations_shape(self):
        ctx = SongContext(title="T", artist="a")
        links, _ = apply_gates([
            claim("S1", "A1", direction="source"),
            claim("S2", "A2", direction="usage"),
        ], ctx)
        rel = links_to_relations(links)
        assert rel["samples"] == [{"song": "S1", "artist": "A1"}]
        assert rel["sampled_by"] == [{"song": "S2", "artist": "A2"}]
