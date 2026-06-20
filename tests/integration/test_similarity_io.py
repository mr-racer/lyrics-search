"""Integration tests for similarity_service file I/O."""

import json
import os
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


class TestLoadTopPairsIndex:
    def _seed(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        mod.clear_load_memo()
        similar = [{
            "song": "A - x", "track_id": "t1", "cover_art_path": None,
            "top_similar": [
                {"name": "B - y", "track_id": "t2", "cover_art_path": None, "score": 90.0},
            ],
        }]
        dissimilar = [{
            "song": "A - x", "track_id": "t1", "cover_art_path": None,
            "top_dissimilar": [
                {"name": "C - z", "track_id": "t3", "cover_art_path": None, "score": 5.0},
            ],
        }]
        mod.save_top_pairs(similar, dissimilar, "idx_col")
        return mod

    def test_index_maps_track_to_both_lists(self, tmp_path, monkeypatch):
        mod = self._seed(tmp_path, monkeypatch)
        idx = mod.load_top_pairs_index("idx_col")
        assert idx is not None
        assert idx["t1"]["similar"][0]["track_id"] == "t2"
        assert idx["t1"]["dissimilar"][0]["track_id"] == "t3"

    def test_index_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        import app.services.similarity_service as mod
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        mod.clear_load_memo()
        assert mod.load_top_pairs_index("nope") is None

    def test_index_invalidates_on_mtime_change(self, tmp_path, monkeypatch):
        mod = self._seed(tmp_path, monkeypatch)
        first = mod.load_top_pairs_index("idx_col")
        assert "t1" in first
        # Rewrite the cache with a different track_id and an explicitly newer mtime.
        mod.save_top_pairs(
            [{"song": "Q - w", "track_id": "t9", "cover_art_path": None,
              "top_similar": [{"name": "R - e", "track_id": "t8",
                               "cover_art_path": None, "score": 77.0}]}],
            [{"song": "Q - w", "track_id": "t9", "cover_art_path": None,
              "top_dissimilar": []}],
            "idx_col",
        )
        path = tmp_path / "idx_col.json"
        future = path.stat().st_mtime + 100
        os.utime(path, (future, future))
        second = mod.load_top_pairs_index("idx_col")
        assert "t9" in second
        assert "t1" not in second
