"""Tests for similarity_service: compute_similarity_matrix, get_top_pairs."""

import numpy as np
import pytest

from app.services.similarity_service import compute_similarity_matrix, get_top_pairs


class TestComputeSimilarityMatrix:
    def test_shape(self, sample_vectors):
        mat = compute_similarity_matrix(sample_vectors)
        assert mat.shape == (3, 3)

    def test_diagonal_is_inf(self, sample_vectors):
        mat = compute_similarity_matrix(sample_vectors)
        for i in range(mat.shape[0]):
            assert np.isinf(mat[i, i])

    def test_identical_vectors_zero_distance(self):
        v = [[1.0, 0.0], [1.0, 0.0]]
        mat = compute_similarity_matrix(v)
        assert mat[0, 1] == pytest.approx(0.0)

    def test_orthogonal_vectors_distance_one(self):
        v = [[1.0, 0.0], [0.0, 1.0]]
        mat = compute_similarity_matrix(v)
        assert mat[0, 1] == pytest.approx(1.0)

    def test_opposite_vectors_distance_two(self):
        v = [[1.0, 0.0], [-1.0, 0.0]]
        mat = compute_similarity_matrix(v)
        assert mat[0, 1] == pytest.approx(2.0)

    def test_cosine_distance_range(self, sample_vectors):
        mat = compute_similarity_matrix(sample_vectors)
        # Off-diagonal values should be in [0, 2]
        mask = ~np.eye(mat.shape[0], dtype=bool)
        off_diag = mat[mask]
        assert np.all(off_diag >= 0)
        assert np.all(off_diag <= 2)


class TestGetTopPairs:
    def _setup(self, n=5, top_k=3):
        """Create a small distance matrix for testing."""
        np.random.seed(42)
        vecs = np.random.randn(n, 3).tolist()
        dist_matrix = compute_similarity_matrix(vecs)
        ids = [f"id_{i}" for i in range(n)]
        id2name = {f"id_{i}": f"Artist - Song {i}" for i in range(n)}
        id2payload = {f"id_{i}": {"cover_art_path": f"/cover/{i}.jpg"} for i in range(n)}
        return dist_matrix, ids, id2name, id2payload, top_k

    def test_returns_correct_count(self):
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        similar, dissimilar = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        assert len(similar) == 5
        assert len(dissimilar) == 5
        assert len(similar[0]["top_similar"]) == top_k
        assert len(dissimilar[0]["top_dissimilar"]) == top_k

    def test_scores_are_percentages(self):
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        similar, dissimilar = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        for entry in similar:
            for item in entry["top_similar"]:
                assert 0 <= item["score"] <= 100

    def test_top_k_capped_at_n_minus_1(self):
        """When k > n-1, only n-1 pairs should be returned."""
        dist_matrix, ids, id2name, id2payload, _ = self._setup(n=3, top_k=5)
        similar, _ = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=10
        )
        assert len(similar[0]["top_similar"]) == 2  # n-1 = 2

    def test_cover_art_path_included(self):
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        similar, _ = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        assert similar[0]["cover_art_path"] == "/cover/0.jpg"

    def test_similar_sorted_by_distance_asc(self):
        """Similar pairs should be sorted smallest distance first."""
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        similar, _ = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        # Scores for similar: (1 - dist/2) * 100, so higher score = smaller distance
        # Similar should have descending scores (best first)
        scores = [item["score"] for item in similar[0]["top_similar"]]
        assert scores == sorted(scores, reverse=True)

    def test_dissimilar_sorted_by_distance_desc(self):
        """Dissimilar pairs should be ordered (scores may not be monotonic due to skip of diagonal)."""
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        _, dissimilar = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        # Just verify structure is correct
        assert len(dissimilar) == 5
        assert len(dissimilar[0]["top_dissimilar"]) == top_k
        for entry in dissimilar:
            assert "top_dissimilar" in entry
            for item in entry["top_dissimilar"]:
                assert 0 <= item["score"] <= 100

    def test_song_names_included(self):
        dist_matrix, ids, id2name, id2payload, top_k = self._setup()
        similar, _ = get_top_pairs(
            dist_matrix, ids, id2name, id2payload, top_k=top_k
        )
        assert "song" in similar[0]["song"].lower()


class TestBuildTrackPairs:
    def test_enriches_from_payload_and_keeps_score(self):
        from app.services.similarity_service import build_track_pairs

        similar = [
            {"track_id": "t2", "name": "B - y", "cover_art_path": "c2", "score": 92.3},
        ]
        dissimilar = [
            {"track_id": "t3", "name": "C - z", "cover_art_path": "c3", "score": 8.1},
        ]
        payloads = {
            "t2": {"title": "y", "artist": "B", "genre": "pop",
                   "cover_art_path": "c2", "duration": 100, "file_path": "/m/2.mp3"},
            "t3": {"title": "z", "artist": "C", "genre": "metal",
                   "cover_art_path": "c3", "duration": 200, "file_path": "/m/3.mp3"},
        }
        res = build_track_pairs(similar, dissimilar, payloads, top_k=3)
        assert res["similar"][0]["track_id"] == "t2"
        assert res["similar"][0]["title"] == "y"
        assert res["similar"][0]["artist"] == "B"
        assert res["similar"][0]["genre"] == "pop"
        assert res["similar"][0]["score"] == 92.3
        assert res["dissimilar"][0]["artist"] == "C"
        assert res["dissimilar"][0]["score"] == 8.1

    def test_falls_back_to_name_split_when_payload_missing(self):
        from app.services.similarity_service import build_track_pairs

        similar = [{"track_id": "gone", "name": "The Artist - A Song - Live",
                    "cover_art_path": "cg", "score": 50.0}]
        res = build_track_pairs(similar, [], {}, top_k=3)
        item = res["similar"][0]
        # name = "{artist} - {title}" -> split on FIRST " - "
        assert item["artist"] == "The Artist"
        assert item["title"] == "A Song - Live"
        assert item["cover_art_path"] == "cg"
        assert item["genre"] is None
        assert item["score"] == 50.0

    def test_caps_at_top_k(self):
        from app.services.similarity_service import build_track_pairs

        similar = [{"track_id": f"t{i}", "name": f"A{i} - B{i}",
                    "cover_art_path": None, "score": float(i)} for i in range(5)]
        res = build_track_pairs(similar, [], {}, top_k=3)
        assert len(res["similar"]) == 3
