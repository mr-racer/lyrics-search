"""Unit tests for MLP classifier training and prediction."""

import json
from unittest.mock import MagicMock

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

import joblib
import numpy as np
import pytest

from app.services.sonic_descriptor_service import SonicDescriptorService


@pytest.fixture
def svc(tmp_path):
    cluster_dir = tmp_path / "clusters"
    classifier_dir = tmp_path / "classifier"
    cluster_dir.mkdir()
    classifier_dir.mkdir()
    return SonicDescriptorService(cluster_dir=cluster_dir, classifier_dir=classifier_dir)


def _stage_two_clusters(svc, collection: str, n_per: int = 20):
    """Build synthetic assignments + labels and the matching vectors."""
    rng = np.random.default_rng(7)
    a = rng.normal(loc=[5, 0, 0, 0], scale=0.2, size=(n_per, 4))
    b = rng.normal(loc=[0, 5, 0, 0], scale=0.2, size=(n_per, 4))
    X = np.vstack([a, b]).astype(np.float32)
    slugs = [f"slug-{i}" for i in range(2 * n_per)]

    assignments = {slugs[i]: 0 for i in range(n_per)}
    assignments.update({slugs[i + n_per]: 1 for i in range(n_per)})

    (svc.cluster_dir / f"{collection}_assignments.json").write_text(json.dumps(assignments))
    svc.save_cluster_labels(collection=collection, labels={0: "Cluster A", 1: "Cluster B"})

    # Build a fake qdrant that returns these vectors for the matching slugs
    points = []
    for i, s in enumerate(slugs):
        p = MagicMock()
        p.id = f"id-{i}"
        p.payload = {"slug": s}
        p.vector = {"audio": X[i].tolist()}
        points.append(p)
    fake_qdrant = MagicMock()
    fake_qdrant.scroll.side_effect = [(points, None)]
    return fake_qdrant, X, slugs


def test_train_classifier_saves_model_and_meta(svc, tmp_path):
    fake_qdrant, X, slugs = _stage_two_clusters(svc, collection="train_test")
    result = svc.train_classifier(qdrant=fake_qdrant, collection="train_test", audio_vector_name="audio")
    assert result["status"] == "ready"
    assert result["accuracy"] >= 0.9  # well-separated synthetic data
    assert "Cluster A" in result["classes"]
    assert "Cluster B" in result["classes"]

    model_path = svc.classifier_dir / "train_test.joblib"
    meta_path = svc.classifier_dir / "train_test_meta.json"
    assert model_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["classes"] == result["classes"]
    assert meta["accuracy"] == result["accuracy"]


def test_train_classifier_aborts_when_no_labels(svc):
    fake_qdrant = MagicMock()
    with pytest.raises(RuntimeError, match="No cluster labels"):
        svc.train_classifier(qdrant=fake_qdrant, collection="empty", audio_vector_name="audio")


def test_predict_class_returns_label_and_confidence(svc):
    fake_qdrant, X, slugs = _stage_two_clusters(svc, collection="predict_test")
    svc.train_classifier(qdrant=fake_qdrant, collection="predict_test", audio_vector_name="audio")

    # Vector very close to cluster A's center [5,0,0,0]
    v = np.array([4.9, 0.1, 0.0, 0.0], dtype=np.float32)
    label, conf = svc.predict_class(collection="predict_test", audio_vector=v)
    assert label == "Cluster A"
    assert 0.0 < conf <= 1.0
    assert conf > 0.5  # high confidence for synthetic well-separated data


def test_predict_class_returns_none_when_no_classifier(svc):
    label, conf = svc.predict_class(collection="never_trained", audio_vector=np.zeros(4, dtype=np.float32))
    assert label is None
    assert conf is None


def test_predict_class_respects_confidence_threshold(svc, tmp_path):
    fake_qdrant, X, slugs = _stage_two_clusters(svc, collection="thresh_test")
    svc.train_classifier(qdrant=fake_qdrant, collection="thresh_test", audio_vector_name="audio")

    # Set a deliberately high threshold; ambiguous vector should return None for label
    svc.min_class_confidence = 0.99
    ambiguous = np.array([2.5, 2.5, 0.0, 0.0], dtype=np.float32)
    label, conf = svc.predict_class(collection="thresh_test", audio_vector=ambiguous)
    # Confidence below threshold → label suppressed, but conf still reported
    assert label is None
    assert conf is not None and conf < 0.99


def test_apply_classifier_bulk_persists_sonic_class(svc, tmp_path, monkeypatch):
    fake_qdrant, X, slugs = _stage_two_clusters(svc, collection="bulk_test")
    svc.train_classifier(qdrant=fake_qdrant, collection="bulk_test", audio_vector_name="audio")

    # Re-stage qdrant to return same points for bulk apply pass
    points = []
    for i, s in enumerate(slugs):
        p = MagicMock()
        p.id = f"id-{i}"
        p.payload = {"slug": s}
        p.vector = {"audio": X[i].tolist()}
        points.append(p)
    fake_qdrant.scroll.side_effect = [(points, None)]

    persisted: list[tuple] = []
    monkeypatch.setattr(
        "app.resources.metadata_db.MetadataDB.upsert_sonic_descriptor",
        lambda song_slug, tags=None, sonic_class=None, confidence=None, audio_signature=None: persisted.append((song_slug, sonic_class, confidence)),
    )

    n = svc.apply_classifier_bulk(qdrant=fake_qdrant, collection="bulk_test", audio_vector_name="audio")
    assert n == 40
    # Cluster A members should predict "Cluster A"
    cluster_a_slugs = {s for s in slugs[:20]}
    cluster_a_predictions = [(slug, cls) for slug, cls, _ in persisted if slug in cluster_a_slugs]
    assert all(cls == "Cluster A" for _, cls in cluster_a_predictions)
