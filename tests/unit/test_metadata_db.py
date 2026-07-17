"""Consolidated unit tests for app.resources.metadata_db.

Merged from the former test_metadata_db_*.py suite: ai_enabled, artist_bios,
audiodb, axis_norm_stats, playlists, recency, stream_signals, thread_safety,
user_settings. Each former module's tests live in its own class below; the
shared fresh-DB autouse fixture is provided by the _IsolatedDB mixin.
"""
import threading
import time
from datetime import datetime

import pytest

from app.resources.metadata_db import MetadataDB


class _IsolatedDB:
    """Autouse fixture: give every test a fresh, isolated tmp-file MetadataDB.

    Hoisted from the identical `_isolated_db` fixtures in the ai_enabled,
    artist_bios, axis_norm_stats and stream_signals source modules.
    """

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        yield
        MetadataDB._reset_for_tests()


# ---- module-level helpers / data ----

STATS = {
    "version": "abc123def456",
    "n": 42,
    "mean": {"energy": 0.01, "vocal_lead": -0.02},
    "std": {"energy": 0.05, "vocal_lead": 0.04},
}


def _insert(track, session="s1", interacted=None, played=100.0):
    MetadataDB.record_playback_event(
        session_id=session, collection_name="colA", track_id=track,
        played_sec=played, total_dur=200.0, interacted=interacted,
    )


def _seed_user(user_id: str) -> None:
    """Create a real (Phase A) user row so settings columns get their defaults."""
    MetadataDB.create_user(
        user_id=user_id, email=f"{user_id}@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )


class TestSetSongLabel(_IsolatedDB):
    """set_song_label — Genius label credit into songs.label."""

    def test_creates_row_when_song_missing(self):
        MetadataDB.set_song_label("kanye-west-stronger", "Roc-A-Fella", "colA")
        rel = MetadataDB.get_song_relations_bulk(["kanye-west-stronger"])
        assert rel["kanye-west-stronger"]["label"] == "Roc-A-Fella"

    def test_updates_existing_row_without_clobbering(self):
        MetadataDB.add_song_facts_batch(
            "kanye-west-stronger", "colA", ["Produced by Kanye West."],
        )
        MetadataDB.set_song_relations(
            "kanye-west-stronger", '["Kanye West"]',
            '{"samples": [{"song": "Harder Better Faster Stronger", "artist": "Daft Punk"}], "sampled_by": []}',
        )
        MetadataDB.set_song_label("kanye-west-stronger", "Roc-A-Fella", "colA")

        rel = MetadataDB.get_song_relations_bulk(["kanye-west-stronger"])
        assert rel["kanye-west-stronger"] == {
            "producer": "Kanye West",
            "label": "Roc-A-Fella",
            "samples": ["Daft Punk — Harder Better Faster Stronger"],
            "sampled_by": None,
        }
        conn = MetadataDB._connect()
        title = conn.execute(
            "SELECT title FROM songs WHERE slug = ?", ("kanye-west-stronger",),
        ).fetchone()[0]
        assert title == "kanye west stronger"  # existing title untouched by label upsert

    def test_overwrites_previous_label(self):
        MetadataDB.set_song_label("kanye-west-stronger", "Old Label", "colA")
        MetadataDB.set_song_label("kanye-west-stronger", "Roc-A-Fella", "colA")
        rel = MetadataDB.get_song_relations_bulk(["kanye-west-stronger"])
        assert rel["kanye-west-stronger"]["label"] == "Roc-A-Fella"

    def test_does_not_clobber_existing_artist_name(self):
        conn = MetadataDB._connect()
        conn.execute(
            "INSERT INTO artists (slug, name, collection_name) VALUES (?, ?, ?)",
            ("kanye", "Kanye West", "colA"),
        )
        conn.commit()
        MetadataDB.set_song_label("kanye-west-stronger", "Roc-A-Fella", "colA")
        name = conn.execute(
            "SELECT name FROM artists WHERE slug = ?", ("kanye",),
        ).fetchone()[0]
        assert name == "Kanye West"


class TestAiEnabled(_IsolatedDB):
    """The ai_enabled column on collection_settings."""

    def test_ai_enabled_column_exists(self):
        conn = MetadataDB.get()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collection_settings)")]
        assert "ai_enabled" in cols

    def test_ai_enabled_defaults_to_1_when_row_created_without_explicit_value(self):
        """Pre-migration semantics: existing flows (set_collection_text_model)
        create rows without touching ai_enabled — column DEFAULT 1 keeps them on."""
        MetadataDB.set_collection_text_model("colA", "all-MiniLM-L6-v2")
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT ai_enabled FROM collection_settings WHERE collection_name = ?",
            ("colA",),
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_get_returns_true_when_row_missing(self):
        """Pre-migration collections without a settings row should default to True.
        Some collections were indexed without ever calling set_collection_text_model."""
        assert MetadataDB.get_collection_ai_enabled("never_seen") is True

    def test_set_then_get_persists(self):
        MetadataDB.set_collection_ai_enabled("colA", False)
        assert MetadataDB.get_collection_ai_enabled("colA") is False
        MetadataDB.set_collection_ai_enabled("colA", True)
        assert MetadataDB.get_collection_ai_enabled("colA") is True

    def test_set_creates_row_when_missing(self):
        """First call should INSERT, second should UPDATE (no UNIQUE conflict)."""
        MetadataDB.set_collection_ai_enabled("colB", False)
        conn = MetadataDB.get()
        rows = conn.execute(
            "SELECT collection_name, ai_enabled FROM collection_settings WHERE collection_name = ?",
            ("colB",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 0

    def test_set_isolated_per_collection(self):
        MetadataDB.set_collection_ai_enabled("colA", False)
        MetadataDB.set_collection_ai_enabled("colB", True)
        assert MetadataDB.get_collection_ai_enabled("colA") is False
        assert MetadataDB.get_collection_ai_enabled("colB") is True


class TestArtistBios(_IsolatedDB):
    """artist_bios table + accessors."""

    def test_artist_bios_table_exists(self):
        conn = MetadataDB.get()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)")]
        assert set(cols) == {"artist_slug", "collection_name", "lang", "bio_text", "generated_at"}

    def test_artist_bios_pk(self):
        """PRIMARY KEY is (artist_slug, collection_name, lang) — same shape as sonic_vibes."""
        conn = MetadataDB.get()
        pk_cols = [r[1] for r in conn.execute("PRAGMA table_info(artist_bios)") if r[5] > 0]
        assert pk_cols == ["artist_slug", "collection_name", "lang"]

    def test_set_and_get_artist_bio(self):
        MetadataDB.set_artist_bio("dua-lipa", "col_a", "en", "Born in London…")
        assert MetadataDB.get_artist_bio("dua-lipa", "col_a", "en") == "Born in London…"
        # other (slug, collection, lang) tuples isolated
        assert MetadataDB.get_artist_bio("dua-lipa", "col_a", "ru") is None
        assert MetadataDB.get_artist_bio("dua-lipa", "col_b", "en") is None

    def test_set_artist_bio_upsert(self):
        """Same (slug, collection, lang) → overwrite, not duplicate row."""
        MetadataDB.set_artist_bio("x", "c", "en", "old")
        MetadataDB.set_artist_bio("x", "c", "en", "new")
        assert MetadataDB.get_artist_bio("x", "c", "en") == "new"
        conn = MetadataDB.get()
        n = conn.execute("SELECT COUNT(*) FROM artist_bios").fetchone()[0]
        assert n == 1

    def test_delete_artist_bios(self):
        MetadataDB.set_artist_bio("a", "col_a", "en", "x")
        MetadataDB.set_artist_bio("b", "col_a", "en", "y")
        MetadataDB.set_artist_bio("c", "col_b", "en", "z")
        n = MetadataDB.delete_artist_bios("col_a")
        assert n == 2
        assert MetadataDB.get_artist_bio("a", "col_a", "en") is None
        # col_b row untouched
        assert MetadataDB.get_artist_bio("c", "col_b", "en") == "z"


class TestAudiodb:
    """init() migrates the artists table to add audiodb columns."""

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "musix.db")
        monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        yield
        MetadataDB._reset_for_tests()

    def test_migration_adds_audiodb_columns_on_fresh_db(self):
        conn = MetadataDB._connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(artists)").fetchall()}
        expected = {
            "audiodb_bio", "mood", "country_code", "country", "label",
            "cutout_path", "thumb_path", "audiodb_mbid", "audiodb_fetched_at",
        }
        assert expected.issubset(cols)

    def test_migration_is_idempotent(self):
        # Re-running init() (which calls the migration) should not raise.
        MetadataDB._reset_for_tests()
        MetadataDB.init()  # First call (via fixture) already migrated
        MetadataDB.init()  # Second call should be a no-op
        conn = MetadataDB._connect()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(artists)").fetchall()}
        assert "audiodb_bio" in cols

    def test_migration_preserves_existing_rows(self):
        # Insert a row before the new columns are added (simulating pre-migration state
        # would require manual DB manipulation; here we just confirm new columns are
        # nullable for existing rows).
        MetadataDB.upsert_artist(slug="kanye-west", name="Kanye West", collection_name="test")
        conn = MetadataDB._connect()
        row = conn.execute(
            "SELECT audiodb_bio, mood, audiodb_fetched_at FROM artists WHERE slug=?",
            ("kanye-west",),
        ).fetchone()
        assert row == (None, None, None)

    def test_upsert_audiodb_inserts_new_row(self):
        # No prior artist row; upsert should INSERT one with name=slug fallback.
        MetadataDB.upsert_artist_audiodb(
            slug="dua-lipa", collection_name="test",
            audiodb_bio="Dua Lipa is a singer.", mood="energetic",
            country_code="GB", country="London, UK", label="Warner",
            cutout_path="/covers/artists/abc.png",
            thumb_path="/covers/artists/def.png",
            audiodb_mbid="12345",
        )
        row = MetadataDB.get_artist_audiodb("dua-lipa", "test")
        assert row["audiodb_bio"] == "Dua Lipa is a singer."
        assert row["mood"] == "energetic"
        assert row["country_code"] == "GB"
        assert row["audiodb_mbid"] == "12345"
        assert row["audiodb_fetched_at"]  # ISO string, truthy

    def test_upsert_audiodb_updates_existing_row(self):
        MetadataDB.upsert_artist(slug="kanye-west", name="Kanye West", collection_name="test")
        MetadataDB.upsert_artist_audiodb(
            slug="kanye-west", collection_name="test",
            audiodb_bio="bio v1", mood="introspective",
            country_code="US", country="Chicago, USA", label=None,
            cutout_path=None, thumb_path=None, audiodb_mbid=None,
        )
        # Update with new bio
        MetadataDB.upsert_artist_audiodb(
            slug="kanye-west", collection_name="test",
            audiodb_bio="bio v2", mood="introspective",
            country_code="US", country="Chicago, USA", label=None,
            cutout_path=None, thumb_path=None, audiodb_mbid=None,
        )
        row = MetadataDB.get_artist_audiodb("kanye-west", "test")
        assert row["audiodb_bio"] == "bio v2"

    def test_get_audiodb_returns_none_when_no_row(self):
        assert MetadataDB.get_artist_audiodb("ghost-artist", "test") is None

    def test_get_audiodb_returns_none_when_artist_has_no_audiodb_data(self):
        # Row exists from upsert_artist but audiodb_fetched_at is NULL — get returns dict
        # with all None values, NOT None itself, so the caller can distinguish missing-row
        # from never-fetched-row.
        MetadataDB.upsert_artist(slug="nobody", name="Nobody", collection_name="test")
        row = MetadataDB.get_artist_audiodb("nobody", "test")
        assert row is not None
        assert row["audiodb_bio"] is None
        assert row["audiodb_fetched_at"] is None

    def test_upsert_audiodb_handles_cross_collection_slug_collision(self):
        """Same slug in two collections: the audiodb data overwrites in place rather than
        raising UNIQUE constraint failed: artists.slug."""
        # Collection A — slug exists with name set
        MetadataDB.upsert_artist(slug="multi-coll", name="Multi Coll", collection_name="A")
        # Collection B — same slug, audiodb enrichment should not raise
        MetadataDB.upsert_artist_audiodb(
            slug="multi-coll", collection_name="B",
            audiodb_bio="Bio from B", mood="reflective",
            country_code="US", country=None, label=None,
            cutout_path=None, thumb_path=None, audiodb_mbid=None,
        )
        # Row exists (the schema is single-PK, so it's the same row regardless of collection)
        row = MetadataDB.get_artist_audiodb("multi-coll", "A")
        # Note: the ON CONFLICT updates the row regardless of which collection it was queried
        # under, so the bio is "Bio from B" even when fetched with collection=A.
        assert row is not None
        assert row["audiodb_bio"] == "Bio from B"


class TestAxisNormStats(_IsolatedDB):
    """The axis_norm_stats / stream_liked_share columns (Stream RecSys)."""

    def test_columns_exist(self):
        conn = MetadataDB.get()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collection_settings)")]
        assert "axis_norm_stats" in cols
        assert "stream_liked_share" in cols

    def test_set_then_get_roundtrip(self):
        MetadataDB.set_axis_norm_stats("colA", STATS)
        assert MetadataDB.get_axis_norm_stats("colA") == STATS

    def test_get_returns_none_when_row_missing(self):
        assert MetadataDB.get_axis_norm_stats("never_seen") is None

    def test_get_returns_none_when_column_null(self):
        """Row created by another flow (text_model) leaves axis_norm_stats NULL."""
        MetadataDB.set_collection_text_model("colA", "some-model")
        assert MetadataDB.get_axis_norm_stats("colA") is None

    def test_corrupt_json_returns_none(self):
        conn = MetadataDB.get()
        conn.execute(
            "INSERT INTO collection_settings (collection_name, axis_norm_stats) VALUES (?, ?)",
            ("colA", "{not json"),
        )
        conn.commit()
        assert MetadataDB.get_axis_norm_stats("colA") is None

    def test_set_upserts_without_clobbering_other_columns(self):
        MetadataDB.set_collection_text_model("colA", "some-model")
        MetadataDB.set_axis_norm_stats("colA", STATS)
        assert MetadataDB.get_collection_text_model("colA") == "some-model"
        assert MetadataDB.get_axis_norm_stats("colA")["n"] == 42

    def test_set_overwrites_previous_stats(self):
        MetadataDB.set_axis_norm_stats("colA", STATS)
        MetadataDB.set_axis_norm_stats("colA", {**STATS, "n": 100})
        assert MetadataDB.get_axis_norm_stats("colA")["n"] == 100


class TestPlaylists:
    """MetadataDB playlist classmethods (Plan 19)."""

    @pytest.fixture(autouse=True)
    def reset_playlists(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM playlists")
        conn.commit()
        yield
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM playlists")
        conn.commit()

    def test_create_playlist_returns_row_with_id_and_timestamps(self):
        pid = MetadataDB.create_playlist("col_a", "Late night", "ночное вождение")
        assert isinstance(pid, int) and pid > 0
        row = MetadataDB.get_playlist_row(pid)
        assert row is not None
        assert row["id"] == pid
        assert row["collection_name"] == "col_a"
        assert row["name"] == "Late night"
        assert row["description"] == "ночное вождение"
        assert row["created_at"] and row["updated_at"]

    def test_create_playlist_duplicate_name_same_collection_raises(self):
        MetadataDB.create_playlist("col_a", "Mix", None)
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            MetadataDB.create_playlist("col_a", "Mix", None)

    def test_create_playlist_same_name_different_collection_ok(self):
        pid1 = MetadataDB.create_playlist("col_a", "Mix", None)
        pid2 = MetadataDB.create_playlist("col_b", "Mix", None)
        assert pid1 != pid2

    def test_list_playlists_returns_only_target_collection_sorted_updated_desc(self):
        a1 = MetadataDB.create_playlist("col_a", "First", None)
        a2 = MetadataDB.create_playlist("col_a", "Second", None)
        MetadataDB.create_playlist("col_b", "Other", None)

        time.sleep(1.05)  # SQLite datetime('now') is second resolution
        MetadataDB.touch_playlist(a1)

        rows = MetadataDB.list_playlists("col_a")
        assert [r["id"] for r in rows] == [a1, a2]
        assert all(r["collection_name"] == "col_a" for r in rows)

    def test_get_playlist_row_returns_none_for_missing(self):
        assert MetadataDB.get_playlist_row(99999) is None

    def test_update_playlist_name_and_description(self):
        pid = MetadataDB.create_playlist("col_a", "Old", "old desc")
        MetadataDB.update_playlist(pid, name="New", description="new desc")
        row = MetadataDB.get_playlist_row(pid)
        assert row["name"] == "New"
        assert row["description"] == "new desc"

    def test_update_playlist_description_only_leaves_name(self):
        pid = MetadataDB.create_playlist("col_a", "Keep", "x")
        MetadataDB.update_playlist(pid, name=None, description="y")
        row = MetadataDB.get_playlist_row(pid)
        assert row["name"] == "Keep"
        assert row["description"] == "y"

    def test_update_playlist_can_clear_description(self):
        pid = MetadataDB.create_playlist("col_a", "Keep", "had desc")
        MetadataDB.update_playlist(pid, name=None, description=None, clear_description=True)
        row = MetadataDB.get_playlist_row(pid)
        assert row["description"] is None

    def test_update_playlist_rename_collision_raises(self):
        MetadataDB.create_playlist("col_a", "A", None)
        other = MetadataDB.create_playlist("col_a", "B", None)
        with pytest.raises(Exception):
            MetadataDB.update_playlist(other, name="A", description=None)

    def test_delete_playlist_removes_row(self):
        pid = MetadataDB.create_playlist("col_a", "Bye", None)
        MetadataDB.delete_playlist(pid)
        assert MetadataDB.get_playlist_row(pid) is None

    def test_delete_playlist_cascades_to_tracks(self):
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

    def test_add_track_to_playlist_assigns_position(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        p1 = MetadataDB.add_track_to_playlist(pid, "t1")
        p2 = MetadataDB.add_track_to_playlist(pid, "t2")
        p3 = MetadataDB.add_track_to_playlist(pid, "t3")
        assert (p1, p2, p3) == (1, 2, 3)

    def test_add_duplicate_track_raises(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        with pytest.raises(Exception):
            MetadataDB.add_track_to_playlist(pid, "t1")

    def test_list_playlist_tracks_returns_in_position_order(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        MetadataDB.add_track_to_playlist(pid, "t2")
        MetadataDB.add_track_to_playlist(pid, "t3")
        rows = MetadataDB.list_playlist_tracks(pid)
        assert [r["track_id"] for r in rows] == ["t1", "t2", "t3"]
        assert [r["position"] for r in rows] == [1, 2, 3]

    def test_remove_track_from_playlist_does_not_renumber(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        MetadataDB.add_track_to_playlist(pid, "t2")
        MetadataDB.add_track_to_playlist(pid, "t3")
        removed = MetadataDB.remove_track_from_playlist(pid, "t2")
        assert removed is True
        rows = MetadataDB.list_playlist_tracks(pid)
        assert [r["track_id"] for r in rows] == ["t1", "t3"]
        assert [r["position"] for r in rows] == [1, 3]

    def test_remove_missing_track_returns_false(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        assert MetadataDB.remove_track_from_playlist(pid, "nope") is False

    def test_reorder_playlist_renumbers_dense(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        MetadataDB.add_track_to_playlist(pid, "t2")
        MetadataDB.add_track_to_playlist(pid, "t3")
        MetadataDB.reorder_playlist(pid, ["t3", "t1", "t2"])
        rows = MetadataDB.list_playlist_tracks(pid)
        assert [r["track_id"] for r in rows] == ["t3", "t1", "t2"]
        assert [r["position"] for r in rows] == [1, 2, 3]

    def test_reorder_with_set_mismatch_raises_value_error(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        MetadataDB.add_track_to_playlist(pid, "t2")
        with pytest.raises(ValueError) as exc:
            MetadataDB.reorder_playlist(pid, ["t1", "t999"])
        msg = str(exc.value)
        assert "missing" in msg or "unexpected" in msg

    def test_track_exists_in_playlist(self):
        pid = MetadataDB.create_playlist("col_a", "M", None)
        MetadataDB.add_track_to_playlist(pid, "t1")
        assert MetadataDB.track_in_playlist(pid, "t1") is True
        assert MetadataDB.track_in_playlist(pid, "t999") is False

    def test_playlists_containing_track_for_collection(self):
        p1 = MetadataDB.create_playlist("col_a", "A", None)
        MetadataDB.create_playlist("col_a", "B", None)
        p3 = MetadataDB.create_playlist("col_a", "C", None)
        MetadataDB.create_playlist("col_b", "D", None)
        MetadataDB.add_track_to_playlist(p1, "t1")
        MetadataDB.add_track_to_playlist(p3, "t1")
        ids = set(MetadataDB.playlists_containing_track("col_a", "t1"))
        assert ids == {p1, p3}


class TestRecency:
    """get_play_recency_map."""

    @pytest.fixture(autouse=True)
    def _db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "m.db")
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        yield
        MetadataDB._reset_for_tests()

    def test_recency_map_keeps_latest_per_track(self):
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t1", played_sec=100.0, total_dur=200.0)
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t1", played_sec=120.0, total_dur=200.0)
        MetadataDB.record_playback_event(session_id="s", collection_name="c",
                                         track_id="t2", played_sec=90.0, total_dur=200.0)
        m = MetadataDB.get_play_recency_map("c")
        assert set(m.keys()) == {"t1", "t2"}
        assert isinstance(m["t1"], str) and "T" in m["t1"]  # ISO format

    def test_recency_map_scoped_by_collection(self):
        MetadataDB.record_playback_event(session_id="s", collection_name="c1",
                                         track_id="t1", played_sec=100.0, total_dur=200.0)
        assert MetadataDB.get_play_recency_map("c2") == {}


class TestGetPlaybackSignals(_IsolatedDB):
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


class TestGetReactionsWithUpdatedAt(_IsolatedDB):
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


class TestPruneOrphanedTracks(_IsolatedDB):
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


class TestThreadSafety:
    """MetadataDB must survive concurrent access from FastAPI's request threadpool.

    Regression for the server-mode auth loop: every request runs get_current_user
    → get_user_by_id in a threadpool worker. With a single shared sqlite3
    connection the concurrent statement use raced inside CPython's sqlite3 module
    and produced sqlite3.InterfaceError ("bad parameter or other API misuse") →
    500, or silently wrong/empty rows → "unknown user id" → 401 → the frontend
    logged the user out in a loop.
    """

    @pytest.fixture(autouse=True)
    def reset_db(self, tmp_path, monkeypatch):
        import app.resources.metadata_db as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "threads_test.db")
        MetadataDB._instance = None
        MetadataDB.close()
        MetadataDB.init()
        yield
        MetadataDB.close()

    def test_concurrent_user_lookups_do_not_corrupt(self, tmp_path):
        MetadataDB.create_user(
            user_id="u-threads", email="threads@example.com",
            password_hash="x", role="owner", created_at=1.0,
        )

        n_threads = 8
        iterations = 400
        errors: list[BaseException] = []
        barrier = threading.Barrier(n_threads)

        def hammer(i: int):
            try:
                barrier.wait()
                for k in range(iterations):
                    row = MetadataDB.get_user_by_id("u-threads")
                    assert row is not None, "existing user resolved to None"
                    assert row["id"] == "u-threads"
                    row2 = MetadataDB.get_user_by_email("threads@example.com")
                    assert row2 is not None and row2["id"] == "u-threads"
                    # Mix in a write the way real traffic does (login timestamps).
                    if k % 50 == i % 50:
                        MetadataDB.update_last_login("u-threads", float(k))
            except BaseException as e:  # noqa: BLE001 — collect everything for the report
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert not errors, f"{len(errors)} thread(s) failed; first: {errors[0]!r}"


class TestUserSettings:
    """Per-user settings round-trip: text_model_name + clap_enabled.

    ADAPTED for merged Phase A: the `users` table already exists with NOT NULL
    `email`/`password_hash` (created by Phase A). Per-user settings are COLUMNS on
    that table (added via _ensure_columns), keyed by the real `users.id`. Tests
    seed a real user via MetadataDB.create_user; there is no bare settings-row
    upsert (creating users is Phase A's job).
    """

    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        import app.resources.metadata_db as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "settings.db")
        MetadataDB._instance = None
        MetadataDB.init()
        yield
        MetadataDB.close()

    def test_new_user_has_default_settings(self):
        _seed_user("acct-1")
        s = MetadataDB.get_user_settings("acct-1")
        assert s["text_model_name"] == "jinaai/jina-embeddings-v2-small-en"
        assert s["clap_enabled"] is True

    def test_update_user_settings_persists_text_model_name(self):
        _seed_user("acct-1")
        MetadataDB.update_user_settings("acct-1", text_model_name="Qwen/Qwen3-Embedding-0.6B")
        s = MetadataDB.get_user_settings("acct-1")
        assert s["text_model_name"] == "Qwen/Qwen3-Embedding-0.6B"
        assert s["clap_enabled"] is True

    def test_update_user_settings_persists_clap_enabled(self):
        _seed_user("acct-1")
        MetadataDB.update_user_settings("acct-1", clap_enabled=False)
        s = MetadataDB.get_user_settings("acct-1")
        assert s["clap_enabled"] is False
        # text_model_name untouched by a clap-only update
        assert s["text_model_name"] == "jinaai/jina-embeddings-v2-small-en"

    def test_update_user_settings_noop_when_all_none(self):
        _seed_user("acct-1")
        MetadataDB.update_user_settings("acct-1")  # no fields → no-op, no error
        s = MetadataDB.get_user_settings("acct-1")
        assert s["text_model_name"] == "jinaai/jina-embeddings-v2-small-en"
        assert s["clap_enabled"] is True

    def test_get_user_settings_unknown_user_returns_none(self):
        assert MetadataDB.get_user_settings("does-not-exist") is None

    def test_init_is_idempotent(self):
        """Re-init must not raise on the ALTER TABLE column adds."""
        MetadataDB.init()
        MetadataDB.init()  # second call: _ensure_columns silently skips existing cols


class TestTasteSignals(_IsolatedDB):
    """Append-only огонёк/вода journal (taste_signals)."""

    def test_record_and_read_session_scoped(self):
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t2", kind="water")
        MetadataDB.record_taste_signal(
            session_id="s2", collection_name="colA", track_id="t3", kind="fire")
        sess = MetadataDB.get_taste_signals("colA", session_id="s1")
        assert {(t, k) for t, k, _ in sess} == {("t1", "fire"), ("t2", "water")}

    def test_all_signals_across_sessions(self):
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        MetadataDB.record_taste_signal(
            session_id="s2", collection_name="colA", track_id="t2", kind="fire")
        allsig = MetadataDB.get_taste_signals("colA")
        assert {(t, k) for t, k, _ in allsig} == {("t1", "fire"), ("t2", "fire")}

    def test_scoped_by_collection(self):
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colB", track_id="t2", kind="fire")
        assert {t for t, _, _ in MetadataDB.get_taste_signals("colA")} == {"t1"}

    def test_append_only_keeps_repeat_fires(self):
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        rows = MetadataDB.get_taste_signals("colA")
        assert [t for t, _, _ in rows] == ["t1", "t1"]

    def test_created_at_is_parseable_iso(self):
        MetadataDB.record_taste_signal(
            session_id="s1", collection_name="colA", track_id="t1", kind="fire")
        _, _, created = MetadataDB.get_taste_signals("colA")[0]
        datetime.fromisoformat(created)  # must not raise


class TestRandomFacts(_IsolatedDB):
    """get_random_facts — the home page's daily pool. Refined (AI-shortened)
    facts in the requested language win; refined English is the second
    choice; the raw English pool only tops up what's left."""

    def _seed_artist(self, slug="radiohead", name="Radiohead", coll="c"):
        MetadataDB.upsert_artist(slug, name, coll)

    def test_raw_pool_when_no_refined(self):
        self._seed_artist()
        MetadataDB.add_artist_fact("radiohead", "c", "Raw fact one")
        out = MetadataDB.get_random_facts("c", limit=3, lang="ru")
        assert [f["fact"] for f in out] == ["Raw fact one"]
        assert out[0]["type"] == "artist"
        assert out[0]["context"] == "Radiohead"

    def test_prefers_refined_in_requested_lang(self):
        self._seed_artist()
        MetadataDB.add_artist_fact("radiohead", "c", "Raw fact")
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="radiohead", collection_name="c",
            lang="ru", refined=["Отшлифованный факт"])
        out = MetadataDB.get_random_facts("c", limit=1, lang="ru")
        assert out == [{"fact": "Отшлифованный факт",
                        "context": "Radiohead", "type": "artist",
                        "artist": "Radiohead", "artist_slug": "radiohead",
                        "title": None}]

    def test_falls_back_to_refined_en_before_raw(self):
        self._seed_artist()
        MetadataDB.add_artist_fact("radiohead", "c", "Raw fact")
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="radiohead", collection_name="c",
            lang="en", refined=["Refined EN"])
        out = MetadataDB.get_random_facts("c", limit=1, lang="ru")
        assert out[0]["fact"] == "Refined EN"

    def test_explicit_empty_refined_blocks_raw_resurfacing(self):
        # AI ran and kept nothing for this artist — its raw facts must not
        # sneak back in through the raw top-up.
        self._seed_artist()
        MetadataDB.add_artist_fact("radiohead", "c", "Rejected raw fact")
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="radiohead", collection_name="c",
            lang="ru", refined=[])
        assert MetadataDB.get_random_facts("c", limit=3, lang="ru") == []

    def test_song_refined_context_joins_artist_and_title(self):
        self._seed_artist()
        MetadataDB.upsert_song("radiohead-creep", "Creep", "radiohead", "c")
        MetadataDB.set_refined_facts(
            scope="song", scope_key="radiohead-creep", collection_name="c",
            lang="ru", refined=["Факт о песне"])
        out = MetadataDB.get_random_facts("c", limit=1, lang="ru")
        assert out == [{"fact": "Факт о песне",
                        "context": "Radiohead — Creep", "type": "song",
                        "artist": "Radiohead", "artist_slug": "radiohead",
                        "title": "Creep"}]

    def test_one_fact_per_entity(self):
        self._seed_artist()
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="radiohead", collection_name="c",
            lang="ru", refined=["Факт 1", "Факт 2", "Факт 3"])
        out = MetadataDB.get_random_facts("c", limit=3, lang="ru")
        assert len(out) == 1   # one entity → at most one pick

    def test_scoped_to_collection(self):
        self._seed_artist(coll="other")
        MetadataDB.set_refined_facts(
            scope="artist", scope_key="radiohead", collection_name="other",
            lang="ru", refined=["X"])
        assert MetadataDB.get_random_facts("c", limit=3, lang="ru") == []
