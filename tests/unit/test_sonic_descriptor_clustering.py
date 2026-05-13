"""Unit tests for HDBSCAN clustering in SonicDescriptorService."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from app.services.sonic_descriptor_service import SonicDescriptorService


@pytest.fixture
def svc(tmp_path):
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    return SonicDescriptorService(cluster_dir=cluster_dir)


def _make_points(vectors, ids=None):
    """Build mock Qdrant scroll output from a list of vectors."""
    pts = []
    for i, v in enumerate(vectors):
        p = MagicMock()
        p.id = ids[i] if ids else f"t{i}"
        p.payload = {"slug": f"slug-{i}", "title": f"Track {i}", "artist": "Artist", "cover_art_path": None}
        p.vector = {"audio": v.tolist()}
        pts.append(p)
    return pts


def test_cluster_library_writes_assignments_and_representatives(svc, tmp_path):
    # Two well-separated synthetic clusters of 8 tracks each in 8D
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[5, 0, 0, 0, 0, 0, 0, 0], scale=0.1, size=(8, 8))
    cluster_b = rng.normal(loc=[0, 5, 0, 0, 0, 0, 0, 0], scale=0.1, size=(8, 8))
    vectors = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    fake_qdrant = MagicMock()
    fake_qdrant.scroll.side_effect = [(_make_points(vectors), None)]

    result = svc.cluster_library(qdrant=fake_qdrant, collection="test", audio_vector_name="audio", min_cluster_size=4)

    assert result["n_tracks"] == 16
    assert result["n_clusters"] >= 2  # HDBSCAN finds the two blobs (+ maybe noise)

    assignments_path = svc.cluster_dir / "test_assignments.json"
    assert assignments_path.exists()
    assignments = json.loads(assignments_path.read_text())
    assert set(assignments.keys()) == {f"slug-{i}" for i in range(16)}

    reps = svc.get_cluster_representatives("test")
    assert len(reps) >= 2
    # Each cluster_id is non-negative (noise=-1 excluded from representatives)
    assert all(r["cluster_id"] >= 0 for r in reps)
    for r in reps:
        assert 1 <= len(r["representative_tracks"]) <= 5
