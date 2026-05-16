"""Unit tests for the ai_enabled column on collection_settings."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


def test_ai_enabled_column_exists():
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(collection_settings)")]
    assert "ai_enabled" in cols


def test_ai_enabled_defaults_to_1_when_row_created_without_explicit_value():
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
