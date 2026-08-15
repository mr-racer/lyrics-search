"""Per-account gating of the assistant's library catalog.

This is the regression test for a porting bug, and the bug is worth stating
because it is invisible from the outside.

``track_metadata`` has a composite primary key ``(collection_name, track_id)``,
so filtering it by ``collection_name`` is correct. ``songs`` and ``artists`` do
NOT: their primary key is a global slug and ``collection_name`` is one mutable
column on it — whichever account indexed a slug LAST owns that column. That is
exactly the failure ``fact_visibility`` was created to fix, and the lab version
of this catalog (which only ever saw a single-account dump) filtered those two
tables by the column.

Ported unchanged, it would hand every account but one an empty artist index: the
assistant would answer "I couldn't work out who this is about" for artists the
user demonstrably owns, and nothing in the log would say why.
"""

from __future__ import annotations

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.library_catalog import LibraryCatalog, invalidate

ALICE = "acct_alice"
BOB = "acct_bob"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    invalidate()
    yield
    MetadataDB._reset_for_tests()
    invalidate()


def _track(collection, track_id, title, artist, slug):
    MetadataDB.upsert_track_metadata(collection, track_id, {
        "title": title, "artist": artist, "primary_artist_slug": slug,
        "album": "A", "year": 2005, "duration": 200.0,
        "file_path": f"/music/{track_id}.mp3",
    })


def _shared_artist_and_song(collections):
    """One real-world artist and song, indexed by several accounts.

    ``upsert_artist`` / ``upsert_song`` write the shared row and mark the slug
    visible to the calling account — the second call overwrites the row's
    ``collection_name`` column, which is precisely what must not decide anything.
    """
    for collection in collections:
        MetadataDB.upsert_artist("kanye-west", "Kanye West", collection)
        MetadataDB.upsert_song("kanye-west-stronger", "Stronger", "kanye-west",
                               collection)


class TestSharedSlugs:
    def test_both_accounts_see_an_artist_they_both_indexed(self):
        _shared_artist_and_song([ALICE, BOB])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")
        _track(BOB, "t2", "Stronger", "Kanye West", "kanye-west")

        for collection in (ALICE, BOB):
            catalog = LibraryCatalog(collection)
            assert "Kanye West" in catalog.artists
            assert catalog.artist_names["kanye-west"] == "Kanye West"
            assert [r["slug"] for r in catalog.song_rows] == ["kanye-west-stronger"]

    def test_the_last_writer_does_not_steal_the_first_ones_visibility(self):
        """Bob indexing the slug after Alice must not empty Alice's index.

        With the gate on ``artists.collection_name`` this is exactly what
        happened, and the symptom was an assistant that had forgotten an artist
        the user still owned.
        """
        _shared_artist_and_song([ALICE])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")
        _shared_artist_and_song([BOB])          # Bob writes the row last

        alice = LibraryCatalog(ALICE)
        assert alice.artist_names.get("kanye-west") == "Kanye West"
        assert alice.song_rows

    def test_an_account_that_never_indexed_the_slug_does_not_see_it(self):
        """The other half of the invariant: visibility is per account, and a
        shared row is not a shared library."""
        _shared_artist_and_song([ALICE])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")

        bob = LibraryCatalog(BOB)
        assert bob.song_rows == []
        assert bob.artist_names == {}


class TestTrackScoping:
    def test_tracks_are_scoped_by_collection(self):
        _shared_artist_and_song([ALICE, BOB])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")
        _track(BOB, "t2", "Runaway", "Kanye West", "kanye-west")

        assert [s["title"] for s in LibraryCatalog(ALICE).songs] == ["Stronger"]
        assert [s["title"] for s in LibraryCatalog(BOB).songs] == ["Runaway"]

    def test_a_pinned_track_from_another_account_resolves_to_nothing(self):
        """The subject pinned by the UI is still checked against this account —
        a track id is not a capability."""
        _shared_artist_and_song([ALICE, BOB])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")

        assert LibraryCatalog(BOB).subject_for_track("t1") is None


class TestCaching:
    def test_the_catalog_is_reused_within_the_ttl(self):
        from app.services.library_catalog import get_catalog

        _shared_artist_and_song([ALICE])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")
        assert get_catalog(ALICE) is get_catalog(ALICE)

    def test_invalidating_rebuilds_it(self):
        """Indexing calls this; without it a freshly indexed track stays
        invisible to the assistant for the whole TTL."""
        from app.services.library_catalog import get_catalog

        _shared_artist_and_song([ALICE])
        _track(ALICE, "t1", "Stronger", "Kanye West", "kanye-west")
        first = get_catalog(ALICE)
        _track(ALICE, "t2", "Runaway", "Kanye West", "kanye-west")
        invalidate(ALICE)
        assert len(get_catalog(ALICE)) == 2
        assert len(first) == 1
