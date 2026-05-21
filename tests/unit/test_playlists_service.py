"""Unit tests for playlists_service (Plan 19)."""
import pytest
from app.resources.metadata_db import MetadataDB
from app.services import playlists_service
from app.services.playlists_service import (
    PlaylistNotFound, PlaylistNameCollision, TrackAlreadyInPlaylist,
)


class FakeQdrant:
    """Tiny stub that returns predefined points by id."""
    def __init__(self, points_by_id: dict):
        self.points_by_id = points_by_id

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        out = []
        for tid in ids:
            if tid in self.points_by_id:
                p = type("P", (), {})()
                p.id = tid
                p.payload = self.points_by_id[tid]
                out.append(p)
        return out


@pytest.fixture(autouse=True)
def _clean():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM playlist_tracks")
    conn.execute("DELETE FROM playlists")
    conn.commit()
    yield


def test_create_returns_summary():
    s = playlists_service.create("col_a", "Mix", "desc", qdrant=FakeQdrant({}))
    assert s.id > 0
    assert s.name == "Mix"
    assert s.track_count == 0
    assert s.cover_track_ids == []
    assert s.cover_art_paths == []


def test_create_duplicate_raises_collision():
    playlists_service.create("col_a", "Mix", None, qdrant=FakeQdrant({}))
    with pytest.raises(PlaylistNameCollision):
        playlists_service.create("col_a", "Mix", None, qdrant=FakeQdrant({}))


def test_get_detail_orphan_partitioning():
    qdrant = FakeQdrant({"t1": {"title": "A", "artist": "X", "cover_art_path": "/c1"}})
    s = playlists_service.create("col_a", "M", None, qdrant=qdrant)
    playlists_service.add_track(s.id, "t1", qdrant=qdrant)
    playlists_service.add_track(s.id, "t_orphan", qdrant=qdrant)
    detail = playlists_service.get(s.id, qdrant=qdrant)
    assert [t.track_id for t in detail.tracks] == ["t1"]
    assert detail.missing_track_ids == ["t_orphan"]


def test_add_track_position_increments():
    qdrant = FakeQdrant({"t1": {"title": "A", "artist": "X"}, "t2": {"title": "B", "artist": "Y"}})
    s = playlists_service.create("col_a", "M", None, qdrant=qdrant)
    playlists_service.add_track(s.id, "t1", qdrant=qdrant)
    playlists_service.add_track(s.id, "t2", qdrant=qdrant)
    detail = playlists_service.get(s.id, qdrant=qdrant)
    assert [t.position for t in detail.tracks] == [1, 2]


def test_add_duplicate_raises():
    qdrant = FakeQdrant({"t1": {"title": "A", "artist": "X"}})
    s = playlists_service.create("col_a", "M", None, qdrant=qdrant)
    playlists_service.add_track(s.id, "t1", qdrant=qdrant)
    with pytest.raises(TrackAlreadyInPlaylist):
        playlists_service.add_track(s.id, "t1", qdrant=qdrant)


def test_get_missing_raises():
    with pytest.raises(PlaylistNotFound):
        playlists_service.get(99999, qdrant=FakeQdrant({}))


def test_list_with_include_track_id_sets_contains_track():
    qdrant = FakeQdrant({"t1": {"title": "A", "artist": "X"}})
    a = playlists_service.create("col_a", "A", None, qdrant=qdrant)
    b = playlists_service.create("col_a", "B", None, qdrant=qdrant)
    playlists_service.add_track(a.id, "t1", qdrant=qdrant)
    rows = playlists_service.list_playlists("col_a", include_track_id="t1", qdrant=qdrant)
    by_name = {r.name: r for r in rows}
    assert by_name["A"].contains_track is True
    assert by_name["B"].contains_track is False


def test_list_without_include_track_id_leaves_contains_track_none():
    qdrant = FakeQdrant({})
    playlists_service.create("col_a", "X", None, qdrant=qdrant)
    rows = playlists_service.list_playlists("col_a", include_track_id=None, qdrant=qdrant)
    assert rows[0].contains_track is None


def test_summary_cover_ids_take_first_four_by_position():
    qdrant = FakeQdrant({
        f"t{i}": {"title": f"T{i}", "artist": "A", "cover_art_path": f"/c{i}"}
        for i in range(1, 7)
    })
    s = playlists_service.create("col_a", "M", None, qdrant=qdrant)
    for i in range(1, 7):
        playlists_service.add_track(s.id, f"t{i}", qdrant=qdrant)
    rows = playlists_service.list_playlists("col_a", include_track_id=None, qdrant=qdrant)
    summary = rows[0]
    assert summary.cover_track_ids == ["t1", "t2", "t3", "t4"]
    assert summary.cover_art_paths == ["/c1", "/c2", "/c3", "/c4"]


def test_reorder_validates_set_equality():
    qdrant = FakeQdrant({"t1": {"title": "A", "artist": "X"}, "t2": {"title": "B", "artist": "Y"}})
    s = playlists_service.create("col_a", "M", None, qdrant=qdrant)
    playlists_service.add_track(s.id, "t1", qdrant=qdrant)
    playlists_service.add_track(s.id, "t2", qdrant=qdrant)
    with pytest.raises(ValueError):
        playlists_service.reorder(s.id, ["t1", "t999"], qdrant=qdrant)
