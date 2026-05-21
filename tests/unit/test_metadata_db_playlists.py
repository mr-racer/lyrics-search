"""Unit tests for MetadataDB playlist classmethods (Plan 19)."""
import pytest
from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def reset_playlists():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM playlist_tracks")
    conn.execute("DELETE FROM playlists")
    conn.commit()
    yield
    conn.execute("DELETE FROM playlist_tracks")
    conn.execute("DELETE FROM playlists")
    conn.commit()


def test_create_playlist_returns_row_with_id_and_timestamps():
    pid = MetadataDB.create_playlist("col_a", "Late night", "ночное вождение")
    assert isinstance(pid, int) and pid > 0
    row = MetadataDB.get_playlist_row(pid)
    assert row is not None
    assert row["id"] == pid
    assert row["collection_name"] == "col_a"
    assert row["name"] == "Late night"
    assert row["description"] == "ночное вождение"
    assert row["created_at"] and row["updated_at"]


def test_create_playlist_duplicate_name_same_collection_raises():
    MetadataDB.create_playlist("col_a", "Mix", None)
    with pytest.raises(Exception):  # sqlite3.IntegrityError
        MetadataDB.create_playlist("col_a", "Mix", None)


def test_create_playlist_same_name_different_collection_ok():
    pid1 = MetadataDB.create_playlist("col_a", "Mix", None)
    pid2 = MetadataDB.create_playlist("col_b", "Mix", None)
    assert pid1 != pid2


def test_list_playlists_returns_only_target_collection_sorted_updated_desc():
    a1 = MetadataDB.create_playlist("col_a", "First", None)
    a2 = MetadataDB.create_playlist("col_a", "Second", None)
    MetadataDB.create_playlist("col_b", "Other", None)

    import time
    time.sleep(1.05)  # SQLite datetime('now') is second resolution
    MetadataDB.touch_playlist(a1)

    rows = MetadataDB.list_playlists("col_a")
    assert [r["id"] for r in rows] == [a1, a2]
    assert all(r["collection_name"] == "col_a" for r in rows)


def test_get_playlist_row_returns_none_for_missing():
    assert MetadataDB.get_playlist_row(99999) is None


def test_update_playlist_name_and_description():
    pid = MetadataDB.create_playlist("col_a", "Old", "old desc")
    MetadataDB.update_playlist(pid, name="New", description="new desc")
    row = MetadataDB.get_playlist_row(pid)
    assert row["name"] == "New"
    assert row["description"] == "new desc"


def test_update_playlist_description_only_leaves_name():
    pid = MetadataDB.create_playlist("col_a", "Keep", "x")
    MetadataDB.update_playlist(pid, name=None, description="y")
    row = MetadataDB.get_playlist_row(pid)
    assert row["name"] == "Keep"
    assert row["description"] == "y"


def test_update_playlist_can_clear_description():
    pid = MetadataDB.create_playlist("col_a", "Keep", "had desc")
    MetadataDB.update_playlist(pid, name=None, description=None, clear_description=True)
    row = MetadataDB.get_playlist_row(pid)
    assert row["description"] is None


def test_update_playlist_rename_collision_raises():
    MetadataDB.create_playlist("col_a", "A", None)
    other = MetadataDB.create_playlist("col_a", "B", None)
    with pytest.raises(Exception):
        MetadataDB.update_playlist(other, name="A", description=None)


def test_delete_playlist_removes_row():
    pid = MetadataDB.create_playlist("col_a", "Bye", None)
    MetadataDB.delete_playlist(pid)
    assert MetadataDB.get_playlist_row(pid) is None


def test_delete_playlist_cascades_to_tracks():
    pid = MetadataDB.create_playlist("col_a", "WithTracks", None)
    conn = MetadataDB.get()
    conn.execute(
        "INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, 't1', 1)",
        (pid,),
    )
    conn.commit()
    MetadataDB.delete_playlist(pid)
    remaining = conn.execute(
        "SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = ?", (pid,)
    ).fetchone()[0]
    assert remaining == 0


def test_add_track_to_playlist_assigns_position():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    p1 = MetadataDB.add_track_to_playlist(pid, "t1")
    p2 = MetadataDB.add_track_to_playlist(pid, "t2")
    p3 = MetadataDB.add_track_to_playlist(pid, "t3")
    assert (p1, p2, p3) == (1, 2, 3)


def test_add_duplicate_track_raises():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    with pytest.raises(Exception):
        MetadataDB.add_track_to_playlist(pid, "t1")


def test_list_playlist_tracks_returns_in_position_order():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    MetadataDB.add_track_to_playlist(pid, "t2")
    MetadataDB.add_track_to_playlist(pid, "t3")
    rows = MetadataDB.list_playlist_tracks(pid)
    assert [r["track_id"] for r in rows] == ["t1", "t2", "t3"]
    assert [r["position"] for r in rows] == [1, 2, 3]


def test_remove_track_from_playlist_does_not_renumber():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    MetadataDB.add_track_to_playlist(pid, "t2")
    MetadataDB.add_track_to_playlist(pid, "t3")
    removed = MetadataDB.remove_track_from_playlist(pid, "t2")
    assert removed is True
    rows = MetadataDB.list_playlist_tracks(pid)
    assert [r["track_id"] for r in rows] == ["t1", "t3"]
    assert [r["position"] for r in rows] == [1, 3]


def test_remove_missing_track_returns_false():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    assert MetadataDB.remove_track_from_playlist(pid, "nope") is False


def test_reorder_playlist_renumbers_dense():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    MetadataDB.add_track_to_playlist(pid, "t2")
    MetadataDB.add_track_to_playlist(pid, "t3")
    MetadataDB.reorder_playlist(pid, ["t3", "t1", "t2"])
    rows = MetadataDB.list_playlist_tracks(pid)
    assert [r["track_id"] for r in rows] == ["t3", "t1", "t2"]
    assert [r["position"] for r in rows] == [1, 2, 3]


def test_reorder_with_set_mismatch_raises_value_error():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    MetadataDB.add_track_to_playlist(pid, "t2")
    with pytest.raises(ValueError) as exc:
        MetadataDB.reorder_playlist(pid, ["t1", "t999"])
    msg = str(exc.value)
    assert "missing" in msg or "unexpected" in msg


def test_track_exists_in_playlist():
    pid = MetadataDB.create_playlist("col_a", "M", None)
    MetadataDB.add_track_to_playlist(pid, "t1")
    assert MetadataDB.track_in_playlist(pid, "t1") is True
    assert MetadataDB.track_in_playlist(pid, "t999") is False


def test_playlists_containing_track_for_collection():
    p1 = MetadataDB.create_playlist("col_a", "A", None)
    p2 = MetadataDB.create_playlist("col_a", "B", None)
    p3 = MetadataDB.create_playlist("col_a", "C", None)
    MetadataDB.create_playlist("col_b", "D", None)
    MetadataDB.add_track_to_playlist(p1, "t1")
    MetadataDB.add_track_to_playlist(p3, "t1")
    ids = set(MetadataDB.playlists_containing_track("col_a", "t1"))
    assert ids == {p1, p3}
