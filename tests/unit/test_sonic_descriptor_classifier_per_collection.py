"""Per-collection classifier cache: no cross-pollination, thread-safe lazy load."""

import json
import threading
from unittest.mock import MagicMock

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

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


def _stage(svc, collection, n_per=20, label_prefix="Class"):
    rng = np.random.default_rng(7 + hash(collection) % 100)
    a = rng.normal(loc=[5, 0, 0, 0], scale=0.2, size=(n_per, 4))
    b = rng.normal(loc=[0, 5, 0, 0], scale=0.2, size=(n_per, 4))
    X = np.vstack([a, b]).astype(np.float32)
    slugs = [f"slug-{i}" for i in range(2 * n_per)]
    assignments = {slugs[i]: 0 for i in range(n_per)}
    assignments.update({slugs[i + n_per]: 1 for i in range(n_per)})
    (svc.cluster_dir / f"{collection}_assignments.json").write_text(json.dumps(assignments))
    svc.save_cluster_labels(collection=collection, labels={0: f"{label_prefix} A", 1: f"{label_prefix} B"})

    points = []
    for i, s in enumerate(slugs):
        p = MagicMock()
        p.id = f"id-{i}"
        p.payload = {"slug": s}
        p.vector = {"audio": X[i].tolist()}
        points.append(p)
    q = MagicMock()
    q.scroll.side_effect = [(points, None)]
    return q, X, slugs


def test_two_collections_get_distinct_cached_classifiers(svc):
    """Calling predict_class on two collections must hit two distinct cached models."""
    q_a, _, _ = _stage(svc, collection="col-A", label_prefix="Alpha")
    svc.train_classifier(qdrant=q_a, collection="col-A", audio_vector_name="audio")
    q_b, _, _ = _stage(svc, collection="col-B", label_prefix="Beta")
    svc.train_classifier(qdrant=q_b, collection="col-B", audio_vector_name="audio")

    v = np.array([4.9, 0.1, 0.0, 0.0], dtype=np.float32)
    label_a, _ = svc.predict_class(collection="col-A", audio_vector=v)
    label_b, _ = svc.predict_class(collection="col-B", audio_vector=v)
    assert label_a == "Alpha A"
    assert label_b == "Beta A"

    # Both cached separately
    assert "col-A" in svc._classifier_cache
    assert "col-B" in svc._classifier_cache
    assert svc._classifier_cache["col-A"] is not svc._classifier_cache["col-B"]


def test_concurrent_lazy_load_for_same_collection_is_thread_safe(svc):
    """Eight threads call predict_class on the same untrained-then-trained
    collection. joblib.load must execute exactly once."""
    q, _, _ = _stage(svc, collection="race-col")
    svc.train_classifier(qdrant=q, collection="race-col", audio_vector_name="audio")
    # Clear cache so the next predict_class triggers a load
    svc._classifier_cache.clear()

    load_calls: list[int] = []

    # Wrap joblib.load via a side-effecting counter to count actual disk loads.
    import joblib
    real_joblib_load = joblib.load
    def counting_load(path):
        load_calls.append(1)
        return real_joblib_load(path)
    joblib.load = counting_load

    try:
        v = np.array([4.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results: list[tuple] = []

        def _worker():
            results.append(svc.predict_class(collection="race-col", audio_vector=v))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(load_calls) == 1, f"expected 1 joblib.load call, got {len(load_calls)}"
        # _stage() default label_prefix is "Class", so cluster 0 → "Class A";
        # v sits at cluster-0's centroid [5,0,0,0].
        assert all(r[0] == "Class A" for r in results)
    finally:
        joblib.load = real_joblib_load
