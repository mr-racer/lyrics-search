"""The library catalog: SQLite in, matched tracks out.

The matching rules are inherited from the production resolver, so these tests
are as much a description of what must not regress as of new behaviour: exact
on a folded feat-stripped title, an artist containment check, fuzzy only behind
a strict gate, and every result honestly tagged.
"""

import sqlite3

import pytest

from lab.agent.catalog import LibraryCatalog
from lab.agent.models import TrackRef

COLLECTION = "acct_test"

TRACKS = [
    ("t1", "Runaway", "Kanye West", 2010),
    ("t2", "Power", "Kanye West", 2010),
    ("t3", "Bohemian Rhapsody", "Queen", 1975),
    ("t4", "Kids", "MGMT", 2007),
    ("t5", "Stronger (feat. Daft Punk)", "Kanye West", 2007),
    ("t6", "Creep", "Radiohead", None),
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "metadata.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE track_metadata (
        collection_name TEXT, track_id TEXT, title TEXT, artist TEXT,
        artists TEXT, artist_slugs TEXT, primary_artist_slug TEXT,
        album TEXT, year INTEGER)""")
    conn.execute("CREATE TABLE artists (slug TEXT, name TEXT, collection_name TEXT)")
    conn.execute("CREATE TABLE songs (slug TEXT, title TEXT, artist_slug TEXT, "
                 "collection_name TEXT)")
    for track_id, title, artist, year in TRACKS:
        slug = artist.lower().replace(" ", "-")
        conn.execute(
            "INSERT INTO track_metadata VALUES (?,?,?,?,?,?,?,?,?)",
            (COLLECTION, track_id, title, artist, None, None, slug, "", year))
        conn.execute("INSERT INTO songs VALUES (?,?,?,?)",
                     (f"{slug}-{title.lower().replace(' ', '-')}", title, slug,
                      COLLECTION))
    for slug, name in {("kanye-west", "Kanye West"), ("queen", "Queen"),
                       ("mgmt", "MGMT"), ("radiohead", "Radiohead")}:
        conn.execute("INSERT INTO artists VALUES (?,?,?)", (slug, name, COLLECTION))
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def catalog(db):
    return LibraryCatalog(db, COLLECTION)


class TestLoading:
    def test_it_reads_the_tracks(self, catalog):
        assert len(catalog) == len(TRACKS)
        assert not catalog.degraded

    def test_the_collection_is_picked_when_not_given(self, db):
        assert LibraryCatalog(db).collection_name == COLLECTION

    def test_an_unreadable_file_degrades_instead_of_raising(self, tmp_path):
        """A missing dump is a normal lab state — the general branch still
        works off the web, it just cannot resolve names."""
        empty = LibraryCatalog(str(tmp_path / "nope.db"))
        assert len(empty) == 0

    def test_stats_report_the_year_span(self, catalog):
        assert catalog.stats()["years"] == (1975, 2010)


class TestArtistResolution:
    def test_cyrillic_finds_the_latin_spelling(self, catalog):
        """The whole reason name matching is transliteration-aware."""
        best = catalog.resolve_artist("Канье Уэст")
        assert best and best[0]["artist"] == "Kanye West"

    def test_a_partial_name_resolves(self, catalog):
        assert catalog.resolve_artist("radiohed")[0]["artist"] == "Radiohead"

    def test_an_unknown_artist_returns_nothing(self, catalog):
        assert catalog.resolve_artist("Дельфин") == []

    def test_the_slug_comes_back_with_the_name(self, catalog):
        assert catalog.artist_slug_for("Kanye West") == "kanye-west"


class TestTrackResolution:
    def test_exact_title_and_artist(self, catalog):
        resolved, missing = catalog.resolve_tracks(
            [TrackRef(title="Runaway", artist="Kanye West")])
        assert not missing
        assert resolved[0].track_id == "t1"
        assert resolved[0].match == "exact"

    def test_the_feat_suffix_is_ignored_on_both_sides(self, catalog):
        resolved, _ = catalog.resolve_tracks(
            [TrackRef(title="Stronger", artist="Kanye West")])
        assert resolved and resolved[0].track_id == "t5"

    def test_a_swapped_line_still_matches(self, catalog):
        """Pages write "Artist — Title" and "Title — Artist" about equally
        often, and a table can have the columns the other way round."""
        resolved, _ = catalog.resolve_tracks(
            [TrackRef(title="Queen", artist="Bohemian Rhapsody")])
        assert resolved and resolved[0].track_id == "t3"

    def test_the_library_year_wins_over_the_page(self, catalog):
        """A listicle prints the compilation's year as often as the release's."""
        resolved, _ = catalog.resolve_tracks(
            [TrackRef(title="Power", artist="Kanye West", year=2015)])
        assert resolved[0].year == 2010

    def test_the_page_year_is_used_when_the_library_has_none(self, catalog):
        resolved, _ = catalog.resolve_tracks(
            [TrackRef(title="Creep", artist="Radiohead", year=1992)])
        assert resolved[0].year == 1992

    def test_a_track_the_library_lacks_comes_back_as_missing(self, catalog):
        resolved, missing = catalog.resolve_tracks(
            [TrackRef(title="Paranoid Android", artist="Radiohead")])
        assert not resolved
        assert missing[0].title == "Paranoid Android"

    def test_a_hallucinated_title_matches_nothing(self, catalog):
        """The guarantee that makes model-extracted titles safe to accept."""
        _, missing = catalog.resolve_tracks(
            [TrackRef(title="Interstellar Dogfight", artist="Kanye West")])
        assert len(missing) == 1

    def test_the_right_title_under_the_wrong_artist_is_refused(self, catalog):
        _, missing = catalog.resolve_tracks(
            [TrackRef(title="Runaway", artist="Bon Jovi")])
        assert len(missing) == 1

    def test_a_typo_lands_as_fuzzy_and_is_labelled(self, catalog):
        resolved, _ = catalog.resolve_tracks(
            [TrackRef(title="Bohemian Rapsody", artist="Queen")])
        assert resolved and resolved[0].match == "fuzzy"

    def test_the_same_track_twice_yields_one_row(self, catalog):
        resolved, _ = catalog.resolve_tracks([
            TrackRef(title="Kids", artist="MGMT"),
            TrackRef(title="Kids", artist="MGMT"),
        ])
        assert len(resolved) == 1

    def test_the_fuzzy_budget_is_respected(self, catalog):
        """A 200-row soundtrack against a big library is a lot of comparisons;
        the cap is what keeps that bounded."""
        refs = [TrackRef(title=f"Nothing Like This {i}") for i in range(10)]
        resolved, missing = catalog.resolve_tracks(refs, max_fuzzy=2)
        assert not resolved and len(missing) == 10


class TestDegradedMode:
    def test_songs_are_used_when_the_track_mirror_is_empty(self, tmp_path):
        path = tmp_path / "partial.db"
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE track_metadata (
            collection_name TEXT, track_id TEXT, title TEXT, artist TEXT,
            artists TEXT, artist_slugs TEXT, primary_artist_slug TEXT,
            album TEXT, year INTEGER)""")
        conn.execute("CREATE TABLE artists (slug TEXT, name TEXT, collection_name TEXT)")
        conn.execute("CREATE TABLE songs (slug TEXT, title TEXT, artist_slug TEXT, "
                     "collection_name TEXT)")
        conn.execute("INSERT INTO songs VALUES ('amerie-1-thing','1 Thing',"
                     "'amerie','acct_x')")
        conn.execute("INSERT INTO artists VALUES ('amerie','Amerie','acct_x')")
        conn.commit()
        conn.close()

        catalog = LibraryCatalog(str(path), "acct_x")
        assert catalog.degraded
        assert catalog.resolve_artist("Амери")[0]["artist"] == "Amerie"
        # Names resolve, but there is no track id to build a playlist from.
        resolved, _ = catalog.resolve_tracks([TrackRef(title="1 Thing")])
        assert resolved and resolved[0].track_id == ""


class TestSongSlug:
    def test_the_slug_is_looked_up_not_derived(self, catalog):
        """Deriving it would mean re-implementing two app-side slugifiers that
        CLAUDE.md warns are not interchangeable. The table already knows."""
        assert catalog.song_slug_for("Runaway", "Kanye West") == "kanye-west-runaway"

    def test_an_unknown_song_has_no_slug(self, catalog):
        assert catalog.song_slug_for("Not In Here", "Kanye West") is None


@pytest.fixture
def collab_db(tmp_path):
    """A library shaped like the one that produced the Amerie/Fergie bug: the
    artist is only ever tagged as a collaboration, and a similarly-spelled
    stranger is present."""
    path = tmp_path / "collab.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE track_metadata (
        collection_name TEXT, track_id TEXT, title TEXT, artist TEXT,
        artists TEXT, artist_slugs TEXT, primary_artist_slug TEXT,
        album TEXT, year INTEGER)""")
    conn.execute("CREATE TABLE artists (slug TEXT, name TEXT, collection_name TEXT)")
    conn.execute("CREATE TABLE songs (slug TEXT, title TEXT, artist_slug TEXT, "
                 "collection_name TEXT)")
    rows = [("a1", "1 Thing", "Amerie feat. Nas", "amerie"),
            ("f1", "Big Girls Don't Cry", "Fergie", "fergie"),
            ("e1", "Hurt", "Nine Inch Nails", "nine-inch-nails"),
            ("c1", "Hurt", "Johnny Cash", "johnny-cash")]
    for track_id, title, artist, slug in rows:
        conn.execute("INSERT INTO track_metadata VALUES (?,?,?,?,?,?,?,?,?)",
                     (COLLECTION, track_id, title, artist, None, None, slug, "", 2005))
        conn.execute("INSERT INTO songs VALUES (?,?,?,?)",
                     (f"{slug}-{title.lower().replace(' ', '-')}", title, slug,
                      COLLECTION))
    # Note: NO plain "Amerie" row here — only the collab tag and Fergie.
    conn.execute("INSERT INTO artists VALUES ('fergie','Fergie',?)", (COLLECTION,))
    conn.execute("INSERT INTO artists VALUES ('nine-inch-nails','Nine Inch Nails',?)",
                 (COLLECTION,))
    conn.execute("INSERT INTO artists VALUES ('johnny-cash','Johnny Cash',?)",
                 (COLLECTION,))
    conn.commit()
    conn.close()
    return str(path)


class TestSubjectResolution:
    """Identity is decided by structure. Where structure runs out, nobody
    guesses silently — the ambiguity is handed on."""

    def test_a_named_song_answers_the_artist_question_outright(self, collab_db):
        """The bug that started this: "Amerie" scores 0.667 against "Fergie"
        and only 0.571 against "Amerie feat. Nas". No similarity is consulted
        at all when the song is in the library — the row carries the slug."""
        cat = LibraryCatalog(collab_db, COLLECTION)
        subject = cat.resolve_subject(song="1 Thing", artist="Amerie")
        assert subject.how == "song-row"
        assert subject.artist_slug == "amerie"
        assert subject.song_slug == "amerie-1-thing"

    def test_a_bare_collab_tag_resolves_by_participant(self, collab_db):
        cat = LibraryCatalog(collab_db, COLLECTION)
        subject = cat.resolve_subject(artist="Amerie")
        assert subject.how == "participant"
        assert subject.artist_slug == "amerie"

    def test_a_stranger_is_never_picked_silently(self, collab_db):
        """Fergie outscores every real candidate for "Ameria". The old code
        took her; this one refuses and says so."""
        cat = LibraryCatalog(collab_db, COLLECTION)
        assert cat.artist_slug_for("Ameria") is None

    def test_transliteration_is_treated_as_exact(self, catalog):
        """Not a fuzzy tier — the strings are EQUAL once Cyrillic is mapped, so
        no model call is needed to be sure."""
        subject = catalog.resolve_subject(artist="МГМТ")
        assert subject.how == "transliteration"
        assert subject.artist_slug == "mgmt"

    def test_a_near_miss_across_alphabets_goes_to_the_shortlist(self, catalog):
        """«Радиохед» scores 0.941 against "Radiohead" — very likely right, but
        "Muse"/"Fuse" scores 0.750 while «канье»/"Kanye West" scores 0.571, so
        no threshold separates likely from wrong. Someone has to judge."""
        subject = catalog.resolve_subject(artist="Радиохед")
        assert subject.how == "shortlist"
        assert subject.candidates[0]["artist"] == "Radiohead"

    def test_an_unmatched_name_becomes_a_shortlist_not_a_pick(self, collab_db):
        cat = LibraryCatalog(collab_db, COLLECTION)
        subject = cat.resolve_subject(artist="Ferji")
        assert subject.how == "shortlist"
        assert not subject.resolved
        assert any(c["artist"] == "Fergie" for c in subject.candidates)

    def test_one_title_by_two_artists_is_an_ambiguity(self, collab_db):
        """"Hurt" is in this library twice. Guessing which one the listener
        meant would load the wrong song's facts half the time."""
        cat = LibraryCatalog(collab_db, COLLECTION)
        subject = cat.resolve_subject(song="Hurt")
        assert subject.how == "shortlist"
        assert {c["artist"] for c in subject.candidates} == {"Nine Inch Nails",
                                                             "Johnny Cash"}
        assert all(c["song_slug"] for c in subject.candidates)

    def test_the_artist_disambiguates_a_shared_title(self, collab_db):
        cat = LibraryCatalog(collab_db, COLLECTION)
        subject = cat.resolve_subject(song="Hurt", artist="Johnny Cash")
        assert subject.how == "song-row"
        assert subject.artist_slug == "johnny-cash"

    def test_nothing_named_resolves_to_nothing(self, collab_db):
        cat = LibraryCatalog(collab_db, COLLECTION)
        assert cat.resolve_subject().how == "none"

    def test_a_leading_prefix_that_is_not_a_participant_is_refused(self, collab_db):
        """"Fer" is a prefix of "Fergie" as a string, but not a participant of
        it as a tag — the separator has to be a real one."""
        cat = LibraryCatalog(collab_db, COLLECTION)
        assert cat.artist_slug_for("Fer") is None
