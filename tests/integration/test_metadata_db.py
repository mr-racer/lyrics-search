"""Integration tests for MetadataDB — full CRUD + schema against SQLite.

Consolidated from the former test_metadata_db_*.py modules; each former
module is now a class (TestInstanceConfig, TestInvites, TestMigration,
TestPlaylistsSchema, TestSonic, TestUsers) so fixtures/helpers stay scoped.
"""

import sqlite3
from pathlib import Path

import pytest

import app.resources.metadata_db as mod
from app.resources.metadata_db import MetadataDB


# --------------------------------------------------------------------------- #
# Module-level helpers (hoisted from the merged source modules)               #
# --------------------------------------------------------------------------- #

def _seed_owner():
    """Reset invites/users and create an owner row (from invites tests)."""
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM invites")
    conn.execute("DELETE FROM users")
    MetadataDB.create_user(
        user_id="owner-1", email="o@x.y", password_hash="h",
        role="owner", created_at=1700000000.0,
    )


def _columns(db_path: Path, table: str) -> set[str]:
    """Return the set of column names for a table (from migration tests)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}
    finally:
        conn.close()


class TestArtists:
    def test_init_creates_tables(self):
        """init() should have already created tables (autouse fixture)."""
        conn = MetadataDB.get()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in cur.fetchall()]
        assert "artists" in tables
        assert "artist_facts" in tables
        assert "songs" in tables
        assert "song_facts" in tables

    def test_upsert_artist(self):
        MetadataDB.upsert_artist("test-artist", "Test Artist", "col1")
        slug = MetadataDB.get_artist_slug("Test Artist", "col1")
        assert slug == "test-artist"

    def test_upsert_artist_updates_on_conflict(self):
        MetadataDB.upsert_artist("test-artist", "Old Name", "col1")
        MetadataDB.upsert_artist("test-artist", "New Name", "col2")
        slug = MetadataDB.get_artist_slug("New Name", "col2")
        assert slug == "test-artist"

    def test_get_artist_slug_unknown(self):
        slug = MetadataDB.get_artist_slug("Nonexistent", "col1")
        assert slug is None


class TestArtistFacts:
    def test_add_and_get_fact(self):
        MetadataDB.upsert_artist("artist-a", "Artist A", "col1")
        MetadataDB.add_artist_fact(
            "artist-a", "col1", "Fact one", source="test"
        )
        facts = MetadataDB.get_artist_facts("artist-a", "col1")
        assert "Fact one" in facts

    def test_add_fact_creates_artist(self):
        """add_artist_fact should create the artist row if missing."""
        MetadataDB.add_artist_fact(
            "new-artist", "col1", "Solo fact", source="test"
        )
        facts = MetadataDB.get_artist_facts("new-artist", "col1")
        assert "Solo fact" in facts

    def test_add_batch(self):
        MetadataDB.upsert_artist("batch-artist", "Batch Artist", "col1")
        MetadataDB.add_artist_facts_batch(
            "batch-artist", "col1", ["F1", "F2", "F3"], source="test"
        )
        facts = MetadataDB.get_artist_facts("batch-artist", "col1")
        assert len(facts) == 3

    def test_facts_scoped_by_collection(self):
        MetadataDB.upsert_artist("scoped", "Scoped", "col_a")
        MetadataDB.add_artist_fact("scoped", "col_a", "Col A fact", source="test")
        # The slug exists only in col_a
        facts_b = MetadataDB.get_artist_facts("scoped", "col_b")
        assert facts_b == []

    def test_get_all_artist_facts_by_collection(self):
        MetadataDB.upsert_artist("all-a", "All A", "col_all")
        MetadataDB.add_artist_facts_batch(
            "all-a", "col_all", ["Fact 1", "Fact 2"], source="test"
        )
        MetadataDB.upsert_artist("all-b", "All B", "col_all")
        MetadataDB.add_artist_facts_batch(
            "all-b", "col_all", ["Fact X"], source="test"
        )
        result = MetadataDB.get_all_artist_facts_by_collection("col_all")
        assert "all-a" in result
        assert "all-b" in result
        assert "Fact 1" in result["all-a"]
        assert "Fact 2" in result["all-a"]

    def test_get_all_empty_collection(self):
        result = MetadataDB.get_all_artist_facts_by_collection("nonexistent")
        assert result == {}


class TestSongs:
    def test_upsert_song(self):
        MetadataDB.upsert_artist("song-artist", "Song Artist", "col1")
        MetadataDB.upsert_song(
            "song-artist-hello", "Hello", "song-artist", "col1"
        )

    def test_upsert_song_updates_on_conflict(self):
        MetadataDB.upsert_artist("upd-artist", "Upd Artist", "col1")
        MetadataDB.upsert_song(
            "upd-artist-s1", "Old Title", "upd-artist", "col1"
        )
        MetadataDB.upsert_song(
            "upd-artist-s1", "New Title", "upd-artist", "col1"
        )


class TestSongFacts:
    def test_add_and_get_song_fact(self):
        MetadataDB.upsert_artist("sf-artist", "SF Artist", "col1")
        MetadataDB.upsert_song(
            "sf-artist-song", "Song", "sf-artist", "col1"
        )
        MetadataDB.add_song_fact(
            "sf-artist-song", "col1", "Song fact", source="test"
        )
        facts = MetadataDB.get_song_facts("sf-artist-song", "col1")
        assert "Song fact" in facts

    def test_add_song_fact_creates_song_row(self):
        """add_song_fact creates song row if missing (artist must exist first)."""
        MetadataDB.upsert_artist("auto", "Auto", "col1")
        MetadataDB.add_song_fact(
            "auto-auto-song", "col1", "Auto fact", source="test"
        )
        facts = MetadataDB.get_song_facts("auto-auto-song", "col1")
        assert "Auto fact" in facts

    def test_add_song_facts_batch(self):
        MetadataDB.upsert_artist("batch-s", "Batch S", "col1")
        MetadataDB.upsert_song(
            "batch-s-s1", "S1", "batch-s", "col1"
        )
        MetadataDB.add_song_facts_batch(
            "batch-s-s1", "col1", ["SF1", "SF2"], source="test"
        )
        facts = MetadataDB.get_song_facts("batch-s-s1", "col1")
        assert len(facts) == 2

    def test_get_all_song_facts_by_collection(self):
        MetadataDB.upsert_artist("gs-artist", "GS Artist", "col_gs")
        MetadataDB.upsert_song("gs-artist-s1", "S1", "gs-artist", "col_gs")
        MetadataDB.add_song_facts_batch(
            "gs-artist-s1", "col_gs", ["F1"], source="test"
        )
        result = MetadataDB.get_all_song_facts_by_collection("col_gs")
        assert "gs-artist-s1" in result

    def test_get_all_song_facts_empty(self):
        result = MetadataDB.get_all_song_facts_by_collection("empty_col")
        assert result == {}


class TestConvenienceHelpers:
    def test_ensure_artist(self):
        slug = MetadataDB.ensure_artist("Ensure Me", "col_e")
        assert slug == "ensure-me"

    def test_ensure_song(self):
        artist_slug, song_slug = MetadataDB.ensure_song(
            "Song Artist", "Song Title", "col_s"
        )
        assert artist_slug == "song-artist"
        assert song_slug == "song-artist-song-title"

    def test_ensure_song_idempotent(self):
        _, s1 = MetadataDB.ensure_song("Idem", "Title", "col_i")
        _, s2 = MetadataDB.ensure_song("Idem", "Title", "col_i")
        assert s1 == s2

    def test_close_resets_singleton(self):
        MetadataDB.close()
        # After close, _instance is None — next get() creates a new connection
        assert MetadataDB._instance is None


class TestInstanceConfig:
    """Phase A instance_config table — single-row, mode-locked (was
    test_metadata_db_instance_config.py)."""

    def test_instance_config_table_exists_after_init(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='instance_config'"
        ).fetchone()
        assert row is not None

    def test_instance_config_single_row_enforced(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.execute(
            "INSERT INTO instance_config (id, mode, created_at) VALUES (1, 'sharing', 1700000000)"
        )
        try:
            conn.execute(
                "INSERT INTO instance_config (id, mode, created_at) VALUES (2, 'server', 1700000001)"
            )
            conn.commit()
            raise AssertionError("expected IntegrityError — only id=1 row allowed")
        except sqlite3.IntegrityError:
            pass
        conn.execute("DELETE FROM instance_config")
        conn.commit()

    def test_instance_config_mode_check_constraint(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        try:
            conn.execute(
                "INSERT INTO instance_config (id, mode, created_at) VALUES (1, 'invalid', 1700000000)"
            )
            conn.commit()
            raise AssertionError("expected IntegrityError on invalid mode")
        except sqlite3.IntegrityError:
            pass
        conn.execute("DELETE FROM instance_config")
        conn.commit()

    def test_get_instance_config_returns_none_when_empty(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        assert MetadataDB.get_instance_config() is None

    def test_set_and_get_instance_config(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1700000000.0)
        row = MetadataDB.get_instance_config()
        assert row == {"mode": "sharing", "created_at": 1700000000.0}

    def test_set_instance_config_rejects_second_write(self):
        """Per spec §4.2: mode is locked at first write. Second set_ raises."""
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1700000000.0)
        try:
            MetadataDB.set_instance_config(mode="server", created_at=1700000100.0)
            raise AssertionError("expected IntegrityError on second instance_config write")
        except sqlite3.IntegrityError:
            pass


class TestInvites:
    """Phase A invites table — create/consume/list (was
    test_metadata_db_invites.py)."""

    def test_invites_table_exists_after_init(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='invites'"
        ).fetchone()
        assert row is not None

    def test_invites_unused_partial_index_exists(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_invites_unused'"
        ).fetchone()
        assert row is not None

    def test_create_invite_round_trip(self):
        _seed_owner()
        MetadataDB.create_invite(
            code="abc123XYZ_-x",
            created_by="owner-1",
            created_at=1700000000.0,
            expires_at=1700000000.0 + 7 * 86400,
        )
        row = MetadataDB.get_invite("abc123XYZ_-x")
        assert row is not None
        assert row["created_by"] == "owner-1"
        assert row["expires_at"] - row["created_at"] == 7 * 86400
        assert row["consumed_by"] is None
        assert row["consumed_at"] is None

    def test_get_invite_unknown_code_returns_none(self):
        _seed_owner()
        assert MetadataDB.get_invite("nope") is None

    def test_consume_invite_marks_row(self):
        _seed_owner()
        MetadataDB.create_user(
            user_id="member-1", email="m@x.y", password_hash="h",
            role="member", created_at=1700000100.0,
        )
        MetadataDB.create_invite(
            code="code1",
            created_by="owner-1",
            created_at=1700000000.0,
            expires_at=1700000000.0 + 86400,
        )
        claimed = MetadataDB.consume_invite("code1", consumed_by="member-1", consumed_at=1700000150.0)
        assert claimed is True
        row = MetadataDB.get_invite("code1")
        assert row["consumed_by"] == "member-1"
        assert row["consumed_at"] == 1700000150.0

    def test_consume_invite_is_atomic_second_claim_fails(self):
        """The `consumed_at IS NULL` predicate makes consume a compare-and-swap:
        a second claim of the same code returns False and does not overwrite."""
        _seed_owner()
        MetadataDB.create_user(
            user_id="m-a", email="a@x.y", password_hash="h",
            role="member", created_at=1700000100.0,
        )
        MetadataDB.create_user(
            user_id="m-b", email="b@x.y", password_hash="h",
            role="member", created_at=1700000110.0,
        )
        MetadataDB.create_invite("race", "owner-1", 1700000000.0, 1700100000.0)
        first = MetadataDB.consume_invite("race", consumed_by="m-a", consumed_at=1700000150.0)
        second = MetadataDB.consume_invite("race", consumed_by="m-b", consumed_at=1700000160.0)
        assert first is True
        assert second is False
        # The first claimant's stamp is preserved.
        row = MetadataDB.get_invite("race")
        assert row["consumed_by"] == "m-a"
        assert row["consumed_at"] == 1700000150.0

    def test_consume_unknown_invite_returns_false(self):
        _seed_owner()
        assert MetadataDB.consume_invite("ghost", consumed_by="x", consumed_at=1.0) is False

    def test_list_invites_filters_consumed(self):
        _seed_owner()
        MetadataDB.create_user(
            user_id="m1", email="m1@x.y", password_hash="h",
            role="member", created_at=1700000100.0,
        )
        MetadataDB.create_invite("active", "owner-1", 1700000000.0, 1700100000.0)
        MetadataDB.create_invite("used",   "owner-1", 1700000010.0, 1700100010.0)
        MetadataDB.consume_invite("used", consumed_by="m1", consumed_at=1700000150.0)

        open_only = MetadataDB.list_invites(include_consumed=False)
        assert [r["code"] for r in open_only] == ["active"]

        everything = MetadataDB.list_invites(include_consumed=True)
        assert sorted(r["code"] for r in everything) == ["active", "used"]

    def test_delete_invite_removes_row(self):
        _seed_owner()
        MetadataDB.create_invite("kill", "owner-1", 1700000000.0, 1700100000.0)
        assert MetadataDB.get_invite("kill") is not None
        MetadataDB.delete_invite("kill")
        assert MetadataDB.get_invite("kill") is None


class TestMigration:
    """MusicBrainz scaffold columns added idempotently (was
    test_metadata_db_migration.py). These tests repoint DB_PATH to their own
    temp file and reset the singleton directly."""

    def test_musicbrainz_columns_present_on_fresh_init(self, tmp_path, monkeypatch):
        """Fresh DB init should add the MusicBrainz columns."""
        db_path = tmp_path / "musix.db"
        monkeypatch.setattr(mod, "DB_PATH", db_path)

        MetadataDB._reset_for_tests()  # implemented in this task
        MetadataDB.init()

        cols = _columns(db_path, "songs")
        assert {"producers", "label", "samples_json", "mbid"}.issubset(cols)

        art_cols = _columns(db_path, "artists")
        assert "mbid" in art_cols

    def test_musicbrainz_migration_idempotent(self, tmp_path, monkeypatch):
        """Running init() twice must not raise (no duplicate column errors)."""
        db_path = tmp_path / "musix.db"
        monkeypatch.setattr(mod, "DB_PATH", db_path)

        MetadataDB._reset_for_tests()
        MetadataDB.init()
        # Second init: should be a no-op for migrations.
        MetadataDB._reset_for_tests()
        MetadataDB.init()  # must not raise

        cols = _columns(db_path, "songs")
        assert {"producers", "label", "samples_json", "mbid"}.issubset(cols)

    def test_migration_preserves_existing_data(self, tmp_path, monkeypatch):
        """Adding columns to a pre-existing songs row must not lose data."""
        db_path = tmp_path / "musix.db"
        monkeypatch.setattr(mod, "DB_PATH", db_path)

        # Phase 1: init without the migration (simulate pre-Plan-3 state).
        # The pre-Plan-3 songs table has the real columns but lacks the MB columns
        # (producers, label, samples_json, mbid) that this migration adds.
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE songs (
              song_key TEXT PRIMARY KEY,
              title TEXT,
              artist TEXT,
              collection_name TEXT
            )
        """)
        conn.execute("INSERT INTO songs (song_key, title, artist) VALUES ('s1', 'T', 'A')")
        conn.commit()
        conn.close()

        # Phase 2: run init() which should add columns without touching the row.
        MetadataDB._reset_for_tests()
        MetadataDB.init()

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT title, artist, producers, label, samples_json, mbid FROM songs").fetchone()
            assert row[0] == "T"
            assert row[1] == "A"
            assert row[2] is None  # producers — not populated
            assert row[3] is None  # label
            assert row[4] is None  # samples_json
            assert row[5] is None  # mbid
        finally:
            conn.close()


class TestPlaylistsSchema:
    """Plan 19 playlist tables exist + constraints (was
    test_metadata_db_playlists_schema.py)."""

    def test_playlists_table_exists_after_init(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='playlists'"
        ).fetchone()
        assert row is not None, "playlists table should be created by MetadataDB.init()"

    def test_playlist_tracks_table_exists_after_init(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='playlist_tracks'"
        ).fetchone()
        assert row is not None

    def test_playlists_unique_constraint_collection_name(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM playlists")
        conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('a', 'mix')")
        conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('b', 'mix')")
        try:
            conn.execute("INSERT INTO playlists (collection_name, name) VALUES ('a', 'mix')")
            conn.commit()
            raise AssertionError("expected IntegrityError on duplicate (collection_name, name)")
        except sqlite3.IntegrityError:
            pass
        conn.execute("DELETE FROM playlists")
        conn.commit()

    def test_playlist_tracks_cascade_delete(self):
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


class TestSonic:
    """Sonic Descriptor columns on songs table (was
    test_metadata_db_sonic.py). Uses its own temp_db fixture that repoints
    DB_PATH/DB_DIR and resets the singleton."""

    @pytest.fixture
    def temp_db(self, monkeypatch, tmp_path):
        """Patch DB_PATH to a temp file, reset singleton, init schema."""
        db_path = tmp_path / "metadata_test.db"
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_path)
        monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
        MetadataDB._instance = None
        MetadataDB.init()
        yield
        MetadataDB._instance = None

    def test_columns_present(self, temp_db):
        conn = MetadataDB.get()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(songs)")}
        assert "sonic_tags_json" in cols
        assert "sonic_class" in cols
        assert "sonic_class_confidence" in cols
        assert "audio_signature" in cols

    def test_upsert_and_get_sonic_descriptor(self, temp_db):
        MetadataDB.upsert_artist("radiohead", "Radiohead", "test_collection")
        MetadataDB.upsert_song("karma-police", "Karma Police", "radiohead", "test_collection")
        tags = [{"tag": "anxious", "score": 0.72}, {"tag": "atmospheric", "score": 0.68}]
        MetadataDB.upsert_sonic_descriptor(
            song_slug="karma-police",
            tags=tags,
            sonic_class="Indie melancholic",
            confidence=0.81,
        )
        desc = MetadataDB.get_sonic_descriptor("karma-police")
        assert desc["sonic_class"] == "Indie melancholic"
        assert desc["sonic_class_confidence"] == 0.81
        assert desc["tags"] == tags

    def test_get_sonic_descriptor_returns_none_for_unknown(self, temp_db):
        desc = MetadataDB.get_sonic_descriptor("never-existed")
        assert desc is None

    def test_upsert_sonic_descriptor_overwrites_previous(self, temp_db):
        MetadataDB.upsert_artist("a", "A", "c")
        MetadataDB.upsert_song("s", "S", "a", "c")
        MetadataDB.upsert_sonic_descriptor("s", tags=[{"tag": "warm", "score": 0.5}], sonic_class="X", confidence=0.4)
        MetadataDB.upsert_sonic_descriptor("s", tags=[{"tag": "cold", "score": 0.9}], sonic_class="Y", confidence=0.9)
        desc = MetadataDB.get_sonic_descriptor("s")
        assert desc["tags"] == [{"tag": "cold", "score": 0.9}]
        assert desc["sonic_class"] == "Y"


class TestUsers:
    """Phase A users table — CRUD + constraints (was
    test_metadata_db_users.py)."""

    def test_users_table_exists_after_init(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        assert row is not None, "users table should be created by MetadataDB.init()"

    def test_users_email_unique_constraint(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) "
            "VALUES ('id-1', 'owner@example.com', 'h', 'owner', 1700000000)"
        )
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, created_at) "
                "VALUES ('id-2', 'owner@example.com', 'h', 'member', 1700000001)"
            )
            conn.commit()
            raise AssertionError("expected IntegrityError on duplicate email")
        except sqlite3.IntegrityError:
            pass
        conn.execute("DELETE FROM users")
        conn.commit()

    def test_users_role_check_constraint(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, created_at) "
                "VALUES ('id-x', 'x@y.z', 'h', 'admin', 1700000000)"
            )
            conn.commit()
            raise AssertionError("expected IntegrityError on invalid role")
        except sqlite3.IntegrityError:
            pass
        conn.execute("DELETE FROM users")
        conn.commit()

    def test_create_user_and_get_by_email(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.commit()
        MetadataDB.create_user(
            user_id="uid-1",
            email="alice@example.com",
            password_hash="$argon2id$...",
            role="owner",
            created_at=1700000000.0,
        )
        row = MetadataDB.get_user_by_email("alice@example.com")
        assert row is not None
        assert row["id"] == "uid-1"
        assert row["email"] == "alice@example.com"
        assert row["role"] == "owner"
        assert row["password_hash"] == "$argon2id$..."
        assert row["last_login_at"] is None

    def test_create_user_normalizes_email_case(self):
        """Caller normalizes case; verify storage preserves what was passed."""
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.commit()
        MetadataDB.create_user(
            user_id="uid-2", email="bob@example.com", password_hash="h",
            role="member", created_at=1700000000.0,
        )
        # Lookup with same casing returns the row
        assert MetadataDB.get_user_by_email("bob@example.com") is not None
        # Lookup with different casing returns None (caller's responsibility to normalize)
        assert MetadataDB.get_user_by_email("BOB@example.com") is None

    def test_get_user_by_id_round_trip(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.commit()
        MetadataDB.create_user(
            user_id="uid-3", email="c@d.e", password_hash="h",
            role="member", created_at=1700000000.0,
        )
        row = MetadataDB.get_user_by_id("uid-3")
        assert row is not None and row["email"] == "c@d.e"
        assert MetadataDB.get_user_by_id("missing") is None

    def test_update_last_login_sets_timestamp(self):
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM users")
        conn.commit()
        MetadataDB.create_user(
            user_id="uid-4", email="z@y.x", password_hash="h",
            role="member", created_at=1700000000.0,
        )
        assert MetadataDB.get_user_by_id("uid-4")["last_login_at"] is None
        MetadataDB.update_last_login("uid-4", 1700001234.5)
        assert MetadataDB.get_user_by_id("uid-4")["last_login_at"] == 1700001234.5
