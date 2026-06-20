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


class TestGetTopPairsBuildTimeAlbumFilter:
    """Build-time same-album exclusion: the cached `top_similar` list must hold
    nearest DIFFERENT-album neighbours (scanning deeper to backfill), so the
    read-time same-album drop never starves the player rail below top_k."""

    def _matrix(self):
        # seed = id_0; its two NEAREST (id_1, id_2) share id_0's album, the
        # farther id_3/id_4 are different albums. The diagonal must be inf.
        m = np.array([
            [np.inf, 0.10, 0.20, 0.50, 0.60],
            [0.10, np.inf, 0.30, 0.70, 0.80],
            [0.20, 0.30, np.inf, 0.40, 0.90],
            [0.50, 0.70, 0.40, np.inf, 0.15],
            [0.60, 0.80, 0.90, 0.15, np.inf],
        ], dtype=float)
        ids = [f"id_{i}" for i in range(5)]
        id2name = {f"id_{i}": f"Artist - Song {i}" for i in range(5)}
        albums = {0: "Seed", 1: "Seed", 2: "Seed", 3: "Other", 4: "OtherTwo"}
        id2payload = {
            f"id_{i}": {"cover_art_path": f"/c/{i}.jpg", "album": albums[i]}
            for i in range(5)
        }
        return m, ids, id2name, id2payload

    def test_excludes_same_album_from_similar_when_enabled(self):
        m, ids, id2name, id2payload = self._matrix()
        similar, _ = get_top_pairs(
            m, ids, id2name, id2payload, top_k=2, drop_same_album_similar=True
        )
        # nearest are id_1/id_2 (same album as seed) → skipped; backfill to id_3/id_4
        s0 = [n["track_id"] for n in similar[0]["top_similar"]]
        assert s0 == ["id_3", "id_4"]

    def test_default_keeps_same_album(self):
        m, ids, id2name, id2payload = self._matrix()
        similar, _ = get_top_pairs(m, ids, id2name, id2payload, top_k=2)
        s0 = [n["track_id"] for n in similar[0]["top_similar"]]
        assert s0 == ["id_1", "id_2"]  # nearest regardless of album

    def test_dissimilar_unaffected_by_album_flag(self):
        m, ids, id2name, id2payload = self._matrix()
        _, diss_on = get_top_pairs(
            m, ids, id2name, id2payload, top_k=2, drop_same_album_similar=True
        )
        _, diss_off = get_top_pairs(m, ids, id2name, id2payload, top_k=2)
        on0 = [n["track_id"] for n in diss_on[0]["top_dissimilar"]]
        off0 = [n["track_id"] for n in diss_off[0]["top_dissimilar"]]
        assert on0 == off0  # contrast keeps same-album → flag has no effect

    def test_keeps_as_many_as_exist_when_album_dominates(self):
        # Only id_3 differs in album from the seed → similar can hold just 1,
        # even with top_k=3 (we never invent different-album neighbours).
        m, ids, id2name, id2payload = self._matrix()
        id2payload["id_4"]["album"] = "Seed"  # now id_1,id_2,id_4 all same album
        similar, _ = get_top_pairs(
            m, ids, id2name, id2payload, top_k=3, drop_same_album_similar=True
        )
        s0 = [n["track_id"] for n in similar[0]["top_similar"]]
        assert s0 == ["id_3"]


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

    def test_drops_same_album_from_similar_only(self):
        from app.services.similarity_service import build_track_pairs
        similar = [
            {"track_id": "a", "name": "X - a", "cover_art_path": None, "score": 90.0},
            {"track_id": "b", "name": "X - b", "cover_art_path": None, "score": 80.0},
        ]
        dissimilar = [
            {"track_id": "c", "name": "X - c", "cover_art_path": None, "score": 5.0},
        ]
        payloads = {
            "a": {"title": "a", "artist": "X", "album": "Same", "duration": 200},
            "b": {"title": "b", "artist": "X", "album": "Other", "duration": 200},
            "c": {"title": "c", "artist": "X", "album": "Same", "duration": 200},
        }
        res = build_track_pairs(
            similar, dissimilar, payloads, top_k=3,
            seed_album="same", drop_same_album_similar=True, min_duration=30.0,
        )
        sim_ids = [t["track_id"] for t in res["similar"]]
        assert sim_ids == ["b"]                       # "a" dropped (same album)
        assert res["dissimilar"][0]["track_id"] == "c"  # contrast keeps same album

    def test_drops_short_tracks(self):
        from app.services.similarity_service import build_track_pairs
        similar = [
            {"track_id": "s", "name": "X - s", "cover_art_path": None, "score": 90.0},
            {"track_id": "l", "name": "X - l", "cover_art_path": None, "score": 80.0},
        ]
        payloads = {
            "s": {"title": "s", "artist": "X", "album": "A", "duration": 10},
            "l": {"title": "l", "artist": "X", "album": "A", "duration": 200},
        }
        res = build_track_pairs(similar, [], payloads, top_k=3, min_duration=30.0)
        assert [t["track_id"] for t in res["similar"]] == ["l"]

    def test_backfills_to_top_k_after_filtering(self):
        from app.services.similarity_service import build_track_pairs
        similar = [
            {"track_id": f"t{i}", "name": f"X - {i}", "cover_art_path": None, "score": float(90 - i)}
            for i in range(6)
        ]
        payloads = {
            "t0": {"title": "0", "artist": "X", "album": "Seed", "duration": 200},  # same album → drop
            "t1": {"title": "1", "artist": "X", "album": "B", "duration": 200},
            "t2": {"title": "2", "artist": "X", "album": "C", "duration": 5},        # short → drop
            "t3": {"title": "3", "artist": "X", "album": "D", "duration": 200},
            "t4": {"title": "4", "artist": "X", "album": "E", "duration": 200},
            "t5": {"title": "5", "artist": "X", "album": "F", "duration": 200},
        }
        res = build_track_pairs(
            similar, [], payloads, top_k=3,
            seed_album="seed", drop_same_album_similar=True, min_duration=30.0,
        )
        assert [t["track_id"] for t in res["similar"]] == ["t1", "t3", "t4"]
