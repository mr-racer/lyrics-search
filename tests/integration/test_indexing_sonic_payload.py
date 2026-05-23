"""Verify that the indexing payload includes sonic_tags
when the SQLite row already has them (e.g. re-indexing a curated library)."""
from __future__ import annotations

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


def test_payload_includes_sonic_tags_when_present(db):
    # slug = _slugify("Artist B") + "-" + _slugify("Title Y")
    #       = "artist-b" + "-" + "title-y" = "artist-b-title-y"
    song_slug = "artist-b-title-y"
    db.upsert_artist("artist-b", "Artist B", "test_col")
    db.upsert_song(song_slug, "Title Y", "artist-b", "test_col")
    db.upsert_sonic_descriptor(
        song_slug=song_slug,
        tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
    )

    from app.services.indexing_service import _build_payload_for_upsert
    song_info = {"title": "Title Y", "artist": "Artist B", "album": None,
                 "year": 2019, "year_range": "2015-2019", "genre": None,
                 "duration": 180, "file_path": "/p", "cover_art_path": None,
                 "producer": None, "label": None, "samples": None, "sampled_by": None,
                 "bitrate_kbps": None, "lyrics": ""}
    payload = _build_payload_for_upsert(song_info, slug=song_slug)
    assert sorted(payload["sonic_tags"]) == ["lo-fi", "melancholic"]


def test_payload_omits_sonic_when_sqlite_empty(db):
    """Track whose SQLite row exists but has no sonic descriptor: payload has empty list."""
    from app.services.indexing_service import _build_payload_for_upsert
    song_info = {"title": "Z", "artist": "C", "album": None, "year": None,
                 "year_range": None, "genre": None, "duration": None,
                 "file_path": "/p", "cover_art_path": None, "producer": None,
                 "label": None, "samples": None, "sampled_by": None,
                 "bitrate_kbps": None, "lyrics": ""}
    payload = _build_payload_for_upsert(song_info, slug="unknown-z")
    assert payload["sonic_tags"] == []


def test_payload_tolerates_no_slug(db):
    """When the caller can't derive a slug, payload stays clean (empty tags)."""
    from app.services.indexing_service import _build_payload_for_upsert
    song_info = {"title": "Z", "artist": "C", "album": None, "year": None,
                 "year_range": None, "genre": None, "duration": None,
                 "file_path": "/p", "cover_art_path": None, "producer": None,
                 "label": None, "samples": None, "sampled_by": None,
                 "bitrate_kbps": None, "lyrics": ""}
    payload = _build_payload_for_upsert(song_info, slug=None)
    assert payload["sonic_tags"] == []
