"""Integration test fixtures — clean MetadataDB per test, FastAPI client."""

import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def clean_metadata_db(tmp_path, monkeypatch):
    """Use a temp SQLite DB for each integration test, then reset singleton."""
    import app.resources.metadata_db as mod

    original_path = mod.DB_PATH
    mod.DB_PATH = tmp_path / "test_metadata.db"
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
            cls._instance = conn
        return cls._instance
    MetadataDB._connect = _patched_connect
    MetadataDB.init()
    yield
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
