"""Sampling links: cleaning at write time and the MusicBrainz lane.

What broke in production and is pinned here:

* raw model output went straight into the table — an album stored as a song,
  the same link under two spellings, and ``dst_slug`` null on every row, which
  left the derived "sampled by" side empty for the whole library;
* the only verifier was a script that had never completed a run, so nothing
  was ever checked.
"""

import asyncio

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.facts_v2 import sample_links as sl
from app.services.facts_v2.verify_lane import (
    VerifyLane, clean_and_store, seed_collection, verify_song_links,
)


class _IsolatedDB:
    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        yield
        MetadataDB._reset_for_tests()


def _candidate(artist, title, *, direction="source", relation="sample",
               fact="", src_slug="src-song"):
    return {"artist": artist, "title": title, "direction": direction,
            "relation": relation, "src_slug": src_slug, "src_artist": "Kanye West",
            "src_title": "Bound 2", "fact": fact}


class _FakeMB:
    """Stands in for MusicBrainz. ``verdicts`` maps a title to verified/not."""

    def __init__(self, verdicts=None, unreachable=False):
        self.verdicts = verdicts or {}
        self.unreachable = unreachable
        self.asked = []
        self.calls = 0

    def verify(self, artist, title):
        self.asked.append((artist, title))
        self.calls += 1
        if self.unreachable:
            return {"checked": False, "error": "offline"}
        return {"checked": True, "verified": bool(self.verdicts.get(title))}


class TestCleanAtWriteTime(_IsolatedDB):
    @pytest.mark.unit
    def test_shape_junk_never_reaches_the_table(self):
        """"The Jamie Foxx Show — The Jamie Foxx Show" is a parse error."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        clean_and_store("acct_test", "src-song", [
            _candidate("The Jamie Foxx Show", "The Jamie Foxx Show"),
            _candidate("Ponderosa Twins Plus One", "Bound"),
        ])
        stored = MetadataDB.get_sample_links("acct_test", "src-song")
        assert [e["song"] for e in stored["samples"]] == ["Bound"]

    @pytest.mark.unit
    def test_two_spellings_of_one_link_collapse(self):
        """Production stored "Billy Squier — The Big Beat" beside
        "Billy Squire — Big Beat" and showed the user both."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="99 Problems",
            artist_slug="jay-z", source="test",
        )
        clean_and_store("acct_test", "src-song", [
            _candidate("Billy Squier", "The Big Beat"),
            _candidate("Billy Squire", "Big Beat"),
        ])
        stored = MetadataDB.get_sample_links("acct_test", "src-song")
        assert len(stored["samples"]) == 1

    @pytest.mark.unit
    def test_link_to_an_owned_track_gets_its_slug(self):
        """dst_slug was null on every stored row, which is what left the
        derived "sampled by" side empty everywhere: it is built from it."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        MetadataDB.add_song_facts_batch(
            "ponderosa-bound", "acct_test", ["f"], title="Bound",
            artist_slug="ponderosa-twins-plus-one", source="test",
        )
        resolve, n_index = sl.library_resolver_from_db("acct_test")
        assert n_index >= 2

        artist = MetadataDB._connect().execute(
            "SELECT name FROM artists WHERE slug = ?",
            ("ponderosa-twins-plus-one",),
        ).fetchone()[0]
        clean_and_store("acct_test", "src-song",
                        [_candidate(artist, "Bound")], resolve=resolve)

        rows = MetadataDB.get_all_sample_links("acct_test")
        assert [r["dst_slug"] for r in rows] == ["ponderosa-bound"]
        # …and that is what makes the other side appear, without a second row.
        back = MetadataDB.get_sample_links("acct_test", "ponderosa-bound")
        assert [e["song"] for e in back["sampled_by"]] == ["Bound 2"]


class TestVerifyLane(_IsolatedDB):
    def _seed_one(self, artist="Some Guy", title="Some Song"):
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        return clean_and_store("acct_test", "src-song",
                               [_candidate(artist, title, fact="")])

    @pytest.mark.unit
    def test_link_musicbrainz_disowns_is_dropped(self):
        keep = self._seed_one(title="Mecca and the Soul Brother")
        mb = _FakeMB(verdicts={"Mecca and the Soul Brother": False})

        got = verify_song_links("acct_test", "src-song", keep, mb)

        assert got["dropped"] == 1
        assert MetadataDB.get_all_sample_links("acct_test") == []

    @pytest.mark.unit
    def test_unreachable_musicbrainz_keeps_the_link(self):
        """Silence is not a verdict. Offline must not empty the pill."""
        keep = self._seed_one()
        mb = _FakeMB(unreachable=True)

        got = verify_song_links("acct_test", "src-song", keep, mb)

        assert got["dropped"] == 0
        assert len(MetadataDB.get_all_sample_links("acct_test")) == 1

    @pytest.mark.unit
    def test_owned_track_is_never_asked_about(self):
        """The user has the file — that is proof no external source improves."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        MetadataDB.add_song_facts_batch(
            "ponderosa-bound", "acct_test", ["f"], title="Bound",
            artist_slug="ponderosa-twins-plus-one", source="test",
        )
        resolve, _ = sl.library_resolver_from_db("acct_test")
        artist = MetadataDB._connect().execute(
            "SELECT name FROM artists WHERE slug = ?",
            ("ponderosa-twins-plus-one",),
        ).fetchone()[0]
        keep = clean_and_store("acct_test", "src-song",
                               [_candidate(artist, "Bound")], resolve=resolve)
        mb = _FakeMB()

        verify_song_links("acct_test", "src-song", keep, mb)

        assert mb.asked == []
        assert len(MetadataDB.get_all_sample_links("acct_test")) == 1

    @pytest.mark.unit
    def test_disabled_lane_swallows_submissions(self):
        """A build without musicbrainzngs must not queue work nobody drains."""
        lane = VerifyLane("acct_test", enabled=False)
        lane.start()
        lane.submit("src-song", [{"artist": "a", "title": "b"}])
        asyncio.get_event_loop_policy()          # no loop needed: nothing ran
        assert lane._worker is None


class TestSeeding(_IsolatedDB):
    @pytest.mark.unit
    def test_seeding_picks_up_links_written_before_verification_existed(self):
        """A resumed refined_facts run returns before the writer for facts it
        has already processed, so links stored by an older build would never
        be cleaned or checked without this."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        # Written the old way: raw, unresolved, and one of them is an album.
        MetadataDB.replace_sample_links("acct_test", "src-song", [
            {"direction": "source", "dst_key": "x|1", "dst_title": "Bound",
             "dst_artist": "Ponderosa Twins Plus One", "dst_slug": None,
             "relation": "sample"},
            {"direction": "source", "dst_key": "x|2",
             "dst_title": "The Jamie Foxx Show",
             "dst_artist": "The Jamie Foxx Show", "dst_slug": None,
             "relation": "sample"},
        ])

        queued = seed_collection("acct_test")

        assert [slug for slug, _ in queued] == ["src-song"]
        stored = MetadataDB.get_sample_links("acct_test", "src-song")
        assert [e["song"] for e in stored["samples"]] == ["Bound"]

    @pytest.mark.unit
    def test_seeding_a_clean_library_rewrites_nothing(self):
        """Re-running the task must not be thousands of pointless writes."""
        MetadataDB.add_song_facts_batch(
            "src-song", "acct_test", ["f"], title="Bound 2",
            artist_slug="kanye-west", source="test",
        )
        clean_and_store("acct_test", "src-song",
                        [_candidate("Ponderosa Twins Plus One", "Bound")])
        before = MetadataDB.get_all_sample_links("acct_test")

        calls = []
        original = MetadataDB.replace_sample_links
        try:
            MetadataDB.replace_sample_links = classmethod(
                lambda cls, *a, **k: calls.append(a))
            seed_collection("acct_test")
        finally:
            MetadataDB.replace_sample_links = original

        assert calls == []
        assert MetadataDB.get_all_sample_links("acct_test") == before
