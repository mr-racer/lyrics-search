"""Unit tests for the axis_norm_stats / stream_liked_share columns (Stream RecSys)."""
import pytest

from app.resources.metadata_db import MetadataDB


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield
    MetadataDB._reset_for_tests()


STATS = {
    "version": "abc123def456",
    "n": 42,
    "mean": {"energy": 0.01, "vocal_lead": -0.02},
    "std": {"energy": 0.05, "vocal_lead": 0.04},
}


def test_columns_exist():
    conn = MetadataDB.get()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(collection_settings)")]
    assert "axis_norm_stats" in cols
    assert "stream_liked_share" in cols


def test_set_then_get_roundtrip():
    MetadataDB.set_axis_norm_stats("colA", STATS)
    assert MetadataDB.get_axis_norm_stats("colA") == STATS


def test_get_returns_none_when_row_missing():
    assert MetadataDB.get_axis_norm_stats("never_seen") is None


def test_get_returns_none_when_column_null():
    """Row created by another flow (text_model) leaves axis_norm_stats NULL."""
    MetadataDB.set_collection_text_model("colA", "some-model")
    assert MetadataDB.get_axis_norm_stats("colA") is None


def test_corrupt_json_returns_none():
    conn = MetadataDB.get()
    conn.execute(
        "INSERT INTO collection_settings (collection_name, axis_norm_stats) VALUES (?, ?)",
        ("colA", "{not json"),
    )
    conn.commit()
    assert MetadataDB.get_axis_norm_stats("colA") is None


def test_set_upserts_without_clobbering_other_columns():
    MetadataDB.set_collection_text_model("colA", "some-model")
    MetadataDB.set_axis_norm_stats("colA", STATS)
    assert MetadataDB.get_collection_text_model("colA") == "some-model"
    assert MetadataDB.get_axis_norm_stats("colA")["n"] == 42


def test_set_overwrites_previous_stats():
    MetadataDB.set_axis_norm_stats("colA", STATS)
    MetadataDB.set_axis_norm_stats("colA", {**STATS, "n": 100})
    assert MetadataDB.get_axis_norm_stats("colA")["n"] == 100
