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
