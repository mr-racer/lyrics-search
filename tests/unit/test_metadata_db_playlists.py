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
