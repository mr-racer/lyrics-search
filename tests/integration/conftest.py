"""Integration test fixtures — clean MetadataDB per test, FastAPI client."""

import shutil

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(scope="session")
def _schema_template(tmp_path_factory):
    """Build the SQLite schema ONCE per session; tests copy the file.

    ``MetadataDB.init()`` replays every CREATE TABLE and every additive ALTER
    TABLE (the re-runs raise "duplicate column" and are swallowed one at a
    time) — about 50 ms. Paid by an autouse fixture, that was ~25 s of the
    suite's wall clock spent building the same schema several hundred times.
    Copying a prepared file is a few milliseconds and yields a byte-identical
    database.
    """
    import app.resources.metadata_db as mod

    template = tmp_path_factory.mktemp("schema") / "template.db"
    original_path = mod.DB_PATH
    MetadataDB.close()
    mod.DB_PATH = template
    try:
        MetadataDB.init()
        conn = MetadataDB._connect()
        # Fold the WAL back into the main file so the copy below is complete.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        MetadataDB.close()
        mod.DB_PATH = original_path
    return template


@pytest.fixture(autouse=True)
def clean_metadata_db(tmp_path, monkeypatch, _schema_template):
    """Use a temp SQLite DB for each integration test, then reset singleton."""
    import app.resources.metadata_db as mod

    original_path = mod.DB_PATH
    mod.DB_PATH = tmp_path / "test_metadata.db"
    shutil.copyfile(_schema_template, mod.DB_PATH)
    seeded_db = str(mod.DB_PATH)
    MetadataDB._instance = None
    # Allow SQLite connection to be used across threads (FastAPI TestClient spawns workers)
    original_connect = MetadataDB._connect
    @classmethod
    def _patched_connect(cls):
        if cls._instance is None:
            mod.DB_DIR.mkdir(parents=True, exist_ok=True)
            import sqlite3
            conn = sqlite3.connect(str(mod.DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")  # mirror prod: survive concurrent writes
            cls._instance = conn
            # ONLY the file this fixture copied from the template arrives with a
            # schema. Tests that repoint DB_PATH at a database of their own
            # (test_yandex_importer, test_backfill, …) must still get a real
            # init() — marking those ready would hand them an empty database.
            if str(mod.DB_PATH) == seeded_db:
                cls._schema_ready.add(seeded_db)
        return cls._instance
    MetadataDB._connect = _patched_connect
    MetadataDB.init()
    # The effective-producer view is cached module-wide (TTL) — a fresh DB per
    # test with reused collection names would otherwise read the previous
    # test's cached rows.
    from app.services import track_credits_service
    track_credits_service.clear_credited_cache()
    yield
    track_credits_service.clear_credited_cache()
    MetadataDB.close()
    mod.DB_PATH = original_path
    MetadataDB._connect = original_connect


@pytest.fixture
def initialized_db():
    """Pre-populated MetadataDB with test data."""
    MetadataDB.upsert_artist("the-weeknd", "The Weeknd", "test_collection")
    MetadataDB.add_artist_facts_batch(
        "the-weeknd",
        "test_collection",
        ["Fact 1 about The Weeknd", "Fact 2 about The Weeknd"],
        source="test",
    )
    MetadataDB.upsert_song(
        "the-weeknd-blinding-lights",
        "Blinding Lights",
        "the-weeknd",
        "test_collection",
    )
    MetadataDB.add_song_facts_batch(
        "the-weeknd-blinding-lights",
        "test_collection",
        ["Song fact 1", "Song fact 2"],
        source="test",
    )
    yield


@pytest.fixture
def fastapi_app():
    """Create FastAPI app with mocked services for API tests."""
    from app.api.main import create_app

    app = create_app()
    # Services will be None (simulating Qdrant unavailable)
    # Override as needed in individual tests
    yield app


@pytest.fixture
def client(fastapi_app):
    """FastAPI TestClient."""
    from fastapi.testclient import TestClient

    with TestClient(fastapi_app) as c:
        yield c
