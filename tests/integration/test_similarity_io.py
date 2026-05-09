"""Integration tests for similarity_service file I/O."""

import json
from pathlib import Path

from app.services.similarity_service import save_top_pairs, load_top_pairs


class TestSimilarityIO:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        data = {
            "similar": [{"song": "A", "top_similar": []}],
            "dissimilar": [{"song": "A", "top_dissimilar": []}],
        }
        path = save_top_pairs(data["similar"], data["dissimilar"], "test_col")
        assert Path(path).exists()

        result = load_top_pairs("test_col")
        assert result is not None
        assert result["collection_name"] == "test_col"
        assert "similar" in result
        assert "dissimilar" in result
        assert "computed_at" in result

    def test_load_returns_none_for_missing(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        result = load_top_pairs("nonexistent")
        assert result is None

    def test_round_trip_preserves_data(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        similar = [
            {
                "song": "A - B",
                "track_id": "id1",
                "cover_art_path": "/cover/1.jpg",
                "top_similar": [
                    {"name": "C - D", "track_id": "id2", "cover_art_path": "/c/2.jpg", "score": 95.5}
                ],
            }
        ]
        dissimilar = [
            {
                "song": "A - B",
                "track_id": "id1",
                "cover_art_path": "/cover/1.jpg",
                "top_dissimilar": [
                    {"name": "E - F", "track_id": "id3", "cover_art_path": "/c/3.jpg", "score": 10.2}
                ],
            }
        ]
        save_top_pairs(similar, dissimilar, "round_trip")
        result = load_top_pairs("round_trip")

        assert result["similar"][0]["song"] == "A - B"
        assert result["similar"][0]["top_similar"][0]["score"] == 95.5
        assert result["dissimilar"][0]["top_dissimilar"][0]["score"] == 10.2

    def test_file_uses_collection_name(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        save_top_pairs([], [], "my_col")
        assert (tmp_path / "my_col.json").exists()


class TestLoadTopPairsStructure:
    def test_loaded_data_has_expected_keys(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        save_top_pairs([], [], "keys_col")
        result = load_top_pairs("keys_col")
        for key in ("similar", "dissimilar", "collection_name", "computed_at"):
            assert key in result
