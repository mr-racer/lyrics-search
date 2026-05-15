"""Unit tests for autoplay-related accessors and service logic (Task 7 + Task 8)."""

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_get_reactions_for_tracks_returns_mapping():
    MetadataDB.set_reaction("t1", "music", "like")
    MetadataDB.set_reaction("t2", "music", "dislike")
    out = MetadataDB.get_reactions_for_tracks("music", ["t1", "t2", "t3"])
    assert out == {"t1": "like", "t2": "dislike"}


def test_get_reactions_for_tracks_empty_list():
    assert MetadataDB.get_reactions_for_tracks("music", []) == {}


def test_get_reactions_for_tracks_unknown_collection():
    MetadataDB.set_reaction("t1", "music", "like")
    assert MetadataDB.get_reactions_for_tracks("other", ["t1"]) == {}


def test_get_reactions_for_tracks_filters_out_reactionless():
    """Track ids with no reaction row are omitted from the result."""
    MetadataDB.set_reaction("t1", "music", "like")
    out = MetadataDB.get_reactions_for_tracks("music", ["t1", "t2"])
    assert "t1" in out and "t2" not in out
