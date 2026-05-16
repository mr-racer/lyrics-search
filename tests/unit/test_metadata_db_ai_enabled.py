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


def test_get_returns_true_when_row_missing():
    """Pre-migration collections without a settings row should default to True.
    Some collections were indexed without ever calling set_collection_text_model."""
    assert MetadataDB.get_collection_ai_enabled("never_seen") is True


def test_set_then_get_persists():
    MetadataDB.set_collection_ai_enabled("colA", False)
    assert MetadataDB.get_collection_ai_enabled("colA") is False
    MetadataDB.set_collection_ai_enabled("colA", True)
    assert MetadataDB.get_collection_ai_enabled("colA") is True


def test_set_creates_row_when_missing():
    """First call should INSERT, second should UPDATE (no UNIQUE conflict)."""
    MetadataDB.set_collection_ai_enabled("colB", False)
    conn = MetadataDB.get()
    rows = conn.execute(
        "SELECT collection_name, ai_enabled FROM collection_settings WHERE collection_name = ?",
        ("colB",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == 0


def test_set_isolated_per_collection():
    MetadataDB.set_collection_ai_enabled("colA", False)
    MetadataDB.set_collection_ai_enabled("colB", True)
    assert MetadataDB.get_collection_ai_enabled("colA") is False
    assert MetadataDB.get_collection_ai_enabled("colB") is True
