"""Verify Plan 19 playlist tables exist after MetadataDB.init()."""
from app.resources.metadata_db import MetadataDB


def test_playlists_table_exists_after_init():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='playlists'"
    ).fetchone()
    assert row is not None, "playlists table should be created by MetadataDB.init()"


def test_playlist_tracks_table_exists_after_init():
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='playlist_tracks'"
    ).fetchone()
    assert row is not None


def test_playlists_unique_constraint_collection_name():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM playlists")
    conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('a', 'mix')")
    conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('b', 'mix')")
    import sqlite3
    try:
        conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('a', 'mix')")
        conn.commit()
        raise AssertionError("expected IntegrityError on duplicate (collection_name, name)")
    except sqlite3.IntegrityError:
        pass
    conn.execute("DELETE FROM playlists")
    conn.commit()


def test_playlist_tracks_cascade_delete():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM playlists")
    cur = conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('a', 'mix')")
    pid = cur.lastrowid
    conn.execute("INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, 't1', 1)", (pid,))
    conn.commit()
    conn.execute("DELETE FROM playlists WHERE id = ?", (pid,))
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = ?", (pid,)).fetchone()[0]
    assert remaining == 0, "playlist_tracks should cascade-delete when parent playlist is deleted"
