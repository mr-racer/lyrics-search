"""Integration tests for MetadataDB — full CRUD against SQLite."""

from app.resources.metadata_db import MetadataDB


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
