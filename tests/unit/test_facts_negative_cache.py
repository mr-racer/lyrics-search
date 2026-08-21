"""songfacts.com misses are remembered, so a rescan stops re-asking.

Both fact fetchers used to persist only successes. An artist or song with no
page upstream therefore hit the network on EVERY indexing run and collected the
same 404 forever — the bulk of the traffic on a settled library, since most
tracks have no songfacts entry at all.

The rule copied from AudioDB: remember a miss only when the server actually
answered. A timeout or a 5xx says nothing about whether the page exists, and
writing it down would bury the artist permanently after one bad night upstream.
"""
import asyncio

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import artist_facts_service as afs
from app.services import song_facts_service as sfs

FACTS_HTML = """
<ul class="artistfacts-results">
  <li><div class="inner">Coldplay formed at university in 1996.</div></li>
</ul>
"""
SONG_HTML = """
<ul class="songfacts-results">
  <li><div class="inner">Recorded in a single take.</div></li>
</ul>
"""


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


@pytest.fixture
def no_legacy_txt(tmp_path, monkeypatch):
    """Point the legacy .txt fallback at an empty dir so it never answers."""
    monkeypatch.setattr(afs, "_FACTS_CACHE_DIR", tmp_path / "afacts")
    monkeypatch.setattr(sfs, "_SONG_FACTS_CACHE_DIR", tmp_path / "sfacts")


def _artist_fetcher(monkeypatch, results):
    """Install a fake _fetch_facts_html; return the list of artists it saw."""
    seen: list = []

    def _fake(artist):
        seen.append(artist)
        return results.pop(0)

    monkeypatch.setattr(afs, "_fetch_facts_html", _fake)
    return seen


def _song_fetcher(monkeypatch, results):
    seen: list = []

    def _fake(artist, song):
        seen.append((artist, song))
        return results.pop(0)

    monkeypatch.setattr(sfs, "_fetch_song_facts_html", _fake)
    return seen


class TestArtistMisses:
    def test_a_definitive_miss_is_never_asked_again(self, isolated_db, no_legacy_txt, monkeypatch):
        # (html, definitive): the server answered, there is simply no page.
        seen = _artist_fetcher(monkeypatch, [(None, True)])

        first = asyncio.run(afs.fetch_artist_facts("Nobody At All", "colA"))
        second = asyncio.run(afs.fetch_artist_facts("Nobody At All", "colA"))

        assert first == (None, True)
        assert second == (None, False), "the second run must not touch the network"
        assert len(seen) == 1

    def test_a_transient_failure_is_retried(self, isolated_db, no_legacy_txt, monkeypatch):
        """A timeout is not evidence that the page is missing."""
        seen = _artist_fetcher(monkeypatch, [(None, False), (FACTS_HTML, True)])

        asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))
        text, hit = asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))

        assert len(seen) == 2
        assert hit is True
        assert "formed at university" in text

    def test_a_hit_is_not_recorded_as_a_miss(self, isolated_db, no_legacy_txt, monkeypatch):
        _artist_fetcher(monkeypatch, [(FACTS_HTML, True)])

        asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))

        assert MetadataDB.has_fact_miss("artist", afs._slugify("Coldplay")) is False

    def test_a_cached_hit_reports_no_network_use(self, isolated_db, no_legacy_txt, monkeypatch):
        _artist_fetcher(monkeypatch, [(FACTS_HTML, True)])
        asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))

        text, hit = asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))

        assert hit is False and "formed at university" in text


class TestSongMisses:
    def test_a_definitive_miss_is_never_asked_again(self, isolated_db, no_legacy_txt, monkeypatch):
        seen = _song_fetcher(monkeypatch, [(None, True)])

        first = asyncio.run(sfs.fetch_song_facts("Daft Punk", "Too Long", "colA"))
        second = asyncio.run(sfs.fetch_song_facts("Daft Punk", "Too Long", "colA"))

        assert first == (None, True)
        assert second == (None, False)
        assert len(seen) == 1

    def test_a_transient_failure_is_retried(self, isolated_db, no_legacy_txt, monkeypatch):
        seen = _song_fetcher(monkeypatch, [(None, False), (SONG_HTML, True)])

        asyncio.run(sfs.fetch_song_facts("Daft Punk", "Too Long", "colA"))
        text, hit = asyncio.run(sfs.fetch_song_facts("Daft Punk", "Too Long", "colA"))

        assert len(seen) == 2
        assert hit is True and "single take" in text


class TestPolitenessDelay:
    """The delay exists to be kind to songfacts.com. Sleeping after a SQLite
    read is 0.5 s of nothing per artist — minutes across a library."""

    def test_artists_sleep_only_after_a_real_request(self, isolated_db, no_legacy_txt, monkeypatch):
        _artist_fetcher(monkeypatch, [(FACTS_HTML, True), (None, True)])
        asyncio.run(afs.fetch_artist_facts("Coldplay", "colA"))          # cached hit
        asyncio.run(afs.fetch_artist_facts("Nobody At All", "colA"))     # remembered miss

        slept: list = []

        async def _no_sleep(secs):
            slept.append(secs)

        monkeypatch.setattr(afs.asyncio, "sleep", _no_sleep)
        asyncio.run(afs.fetch_facts_for_artists(
            ["Coldplay", "Nobody At All"], "colA", delay=0.5))

        assert slept == [], "neither artist needed the network"

    def test_songs_sleep_only_after_a_real_request(self, isolated_db, no_legacy_txt, monkeypatch):
        _song_fetcher(monkeypatch, [(SONG_HTML, True), (None, True)])
        asyncio.run(sfs.fetch_song_facts("Daft Punk", "One More Time", "colA"))
        asyncio.run(sfs.fetch_song_facts("Daft Punk", "Too Long", "colA"))

        slept: list = []

        async def _no_sleep(secs):
            slept.append(secs)

        monkeypatch.setattr(sfs.asyncio, "sleep", _no_sleep)
        asyncio.run(sfs.fetch_facts_for_songs(
            [("Daft Punk", "One More Time"), ("Daft Punk", "Too Long")],
            "colA", delay=0.5))

        assert slept == []


class TestMissStore:
    def test_the_miss_pool_is_shared_across_accounts(self, isolated_db):
        """songfacts.com does not have a page for this artist for ANYBODY —
        the same reasoning that makes the facts pool itself unscoped."""
        MetadataDB.mark_fact_miss("artist", "nobody-at-all")

        assert MetadataDB.has_fact_miss("artist", "nobody-at-all") is True
        assert MetadataDB.has_fact_miss("song", "nobody-at-all") is False

    def test_marking_twice_is_harmless(self, isolated_db):
        MetadataDB.mark_fact_miss("song", "daft-punk-too-long")
        MetadataDB.mark_fact_miss("song", "daft-punk-too-long")

        assert MetadataDB.has_fact_miss("song", "daft-punk-too-long") is True
