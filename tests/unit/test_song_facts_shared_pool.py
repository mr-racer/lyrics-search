"""Song facts are a shared pool — a second account must not re-fetch them.

``fetch_artist_facts`` has short-circuited on the shared pool for a while:
facts are keyed by slug and the same real-world artist has the same facts for
everyone, so once ANY account has them, the rest just get visibility. Songs
were the half that still went back to songfacts.com per account.
"""
import asyncio

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import song_facts_service as sfs

SONG_HTML = """
<ul class="songfacts-results">
  <li><div class="inner">Recorded in a single take.</div></li>
</ul>
"""


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    monkeypatch.setattr(sfs, "_SONG_FACTS_CACHE_DIR", tmp_path / "sfacts")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _fetcher(monkeypatch, results):
    seen: list = []

    def _fake(artist, song):
        seen.append((artist, song))
        return results.pop(0)

    monkeypatch.setattr(sfs, "_fetch_song_facts_html", _fake)
    return seen


def test_a_second_account_reads_the_shared_pool(isolated_db, monkeypatch):
    seen = _fetcher(monkeypatch, [(SONG_HTML, True)])

    asyncio.run(sfs.fetch_song_facts("Daft Punk", "One More Time", "colA"))
    text, hit_network = asyncio.run(
        sfs.fetch_song_facts("Daft Punk", "One More Time", "colB"))

    assert len(seen) == 1, "the second account must not hit songfacts.com"
    assert hit_network is False
    assert "single take" in text


def test_the_shared_pool_becomes_visible_to_the_second_account(isolated_db, monkeypatch):
    """Reading it once is not enough — the account needs its own visibility row,
    or every later UI read (which IS visibility-gated) comes back empty."""
    _fetcher(monkeypatch, [(SONG_HTML, True)])
    asyncio.run(sfs.fetch_song_facts("Daft Punk", "One More Time", "colA"))
    asyncio.run(sfs.fetch_song_facts("Daft Punk", "One More Time", "colB"))

    key = sfs.get_song_facts_key("Daft Punk", "One More Time")
    assert MetadataDB.get_song_facts(key, "colB")


def test_the_shared_pool_is_consulted_before_the_network(isolated_db, monkeypatch):
    """Written directly by another account's run — no local cache, no miss row."""
    key = sfs.get_song_facts_key("Daft Punk", "Aerodynamic")
    MetadataDB.add_song_facts_batch(
        key, "colA", ["Built around a guitar solo."], source="songfacts.com",
        artist_name="Daft Punk", title="Aerodynamic", artist_slug="daft-punk",
    )
    seen = _fetcher(monkeypatch, [])

    text, hit_network = asyncio.run(
        sfs.fetch_song_facts("Daft Punk", "Aerodynamic", "colB"))

    assert seen == []
    assert hit_network is False
    assert "guitar solo" in text
