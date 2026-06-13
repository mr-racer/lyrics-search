"""DB accessors feeding the stream profiles: get_playback_signals + reactions."""
from datetime import datetime

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def _insert(track, session="s1", interacted=None, played=100.0):
    MetadataDB.record_playback_event(
        session_id=session, collection_name="colA", track_id=track,
        played_sec=played, total_dur=200.0, interacted=interacted,
    )


class TestGetPlaybackSignals:
    def test_chronological_order(self):
        for t in ("t1", "t2", "t3"):
            _insert(t)
        out = MetadataDB.get_playback_signals("colA")
        assert [r["track_id"] for r in out] == ["t1", "t2", "t3"]

    def test_limit_keeps_latest(self):
        for i in range(5):
            _insert(f"t{i}")
        out = MetadataDB.get_playback_signals("colA", limit=2)
        assert [r["track_id"] for r in out] == ["t3", "t4"]

    def test_scoped_to_collection(self):
        _insert("mine")
        MetadataDB.record_playback_event(
            session_id="sX", collection_name="colB", track_id="other",
            played_sec=10.0, total_dur=None,
        )
        out = MetadataDB.get_playback_signals("colA")
        assert [r["track_id"] for r in out] == ["mine"]

    def test_types_normalised(self):
        _insert("t1", interacted=True)
        row = MetadataDB.get_playback_signals("colA")[0]
        assert isinstance(row["played_at"], datetime)
        assert row["interacted"] is True
        assert isinstance(row["played_sec"], float)
        assert row["session_id"] == "s1"

    def test_legacy_interacted_null_maps_to_none(self):
        _insert("t1", interacted=None)
        assert MetadataDB.get_playback_signals("colA")[0]["interacted"] is None


class TestGetReactionsWithUpdatedAt:
    def test_returns_all_reactions_with_iso_timestamps(self):
        MetadataDB.set_reaction("liked", "colA", "like")
        MetadataDB.set_reaction("hated", "colA", "dislike")
        out = MetadataDB.get_reactions_with_updated_at("colA")
        by_id = {tid: (reaction, ts) for tid, reaction, ts in out}
        assert by_id["liked"][0] == "like"
        assert by_id["hated"][0] == "dislike"
        # ISO-parseable timestamp
        datetime.fromisoformat(by_id["liked"][1])

    def test_scoped_to_collection(self):
        MetadataDB.set_reaction("a", "colA", "like")
        MetadataDB.set_reaction("b", "colB", "like")
        out = MetadataDB.get_reactions_with_updated_at("colA")
        assert [t[0] for t in out] == ["a"]


class TestPruneOrphanedTracks:
    """Re-indexing mints fresh uuid4 point ids; old reactions/events become
    orphans. prune_orphaned_tracks drops rows whose track_id is no longer a
    live Qdrant id, scoped to the re-indexed collection."""

    def test_removes_orphan_reactions_and_events(self):
        MetadataDB.set_reaction("kept", "colA", "like")
        MetadataDB.set_reaction("gone", "colA", "like")
        _insert("kept")
        _insert("gone")

        removed = MetadataDB.prune_orphaned_tracks("colA", {"kept"})

        assert removed == {"track_reactions": 1, "playback_events": 1}
        assert [t[0] for t in MetadataDB.get_reactions_with_updated_at("colA")] == ["kept"]
        assert [r["track_id"] for r in MetadataDB.get_playback_signals("colA")] == ["kept"]

    def test_scoped_to_collection(self):
        """A re-index of colA must not touch colB's history."""
        MetadataDB.set_reaction("b", "colB", "like")
        MetadataDB.set_reaction("orphan", "colA", "like")

        MetadataDB.prune_orphaned_tracks("colA", set())

        assert [t[0] for t in MetadataDB.get_reactions_with_updated_at("colB")] == ["b"]
        assert MetadataDB.get_reactions_with_updated_at("colA") == []

    def test_noop_when_all_live(self):
        MetadataDB.set_reaction("a", "colA", "like")
        _insert("a")
        removed = MetadataDB.prune_orphaned_tracks("colA", {"a", "extra"})
        assert removed == {"track_reactions": 0, "playback_events": 0}
