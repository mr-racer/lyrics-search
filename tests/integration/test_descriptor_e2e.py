"""End-to-end smoke: tag → cluster → label → train → predict → persist."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.sonic_descriptor_service import SonicDescriptorService


@pytest.fixture
def staged(tmp_path):
    # Vocab + cached prompt embeddings (8D)
    vocab = {"version": 1, "groups": {"e": ["punchy", "ambient", "warm", "dark"]}}
    vocab_path = tmp_path / "prompts.json"
    vocab_path.write_text(json.dumps(vocab))
    emb_path = tmp_path / "embs.npy"
    np.save(emb_path, np.eye(4, 8, dtype=np.float32))  # 4 prompts in 8D

    svc = SonicDescriptorService(
        prompt_vocab_path=vocab_path,
        embeddings_path=emb_path,
        cluster_dir=tmp_path / "clusters",
        classifier_dir=tmp_path / "classifier",
        top_k_tags=2,
    )

    rng = np.random.default_rng(42)
    a = rng.normal(loc=[5, 0, 0, 0, 0, 0, 0, 0], scale=0.15, size=(15, 8))
    b = rng.normal(loc=[0, 5, 0, 0, 0, 0, 0, 0], scale=0.15, size=(15, 8))
    X = np.vstack([a, b]).astype(np.float32)
    slugs = [f"track-{i}" for i in range(30)]

    points = []
    for i, s in enumerate(slugs):
        p = MagicMock()
        p.id = f"id-{i}"
        p.payload = {"slug": s, "title": f"T{i}", "artist": "A", "cover_art_path": None}
        p.vector = {"audio": X[i].tolist()}
        points.append(p)
    qdrant = MagicMock()
    # Reusable side_effect for multiple scroll calls (clustering, training, apply)
    qdrant.scroll.side_effect = lambda **kw: (points, None)
    return svc, qdrant, X, slugs


def test_full_descriptor_pipeline(staged, monkeypatch):
    svc, qdrant, X, slugs = staged

    persisted: dict[str, dict] = {}

    def fake_upsert(song_slug, tags=None, sonic_class=None, confidence=None, audio_signature=None):
        persisted.setdefault(song_slug, {})
        if tags is not None:
            persisted[song_slug]["tags"] = tags
        if sonic_class is not None:
            persisted[song_slug]["sonic_class"] = sonic_class
        if confidence is not None:
            persisted[song_slug]["confidence"] = confidence

    monkeypatch.setattr(
        "app.resources.metadata_db.MetadataDB.upsert_sonic_descriptor",
        fake_upsert,
    )

    # 1. Tag everyone
    n_tagged = svc.compute_tags_bulk(qdrant=qdrant, collection="e2e", audio_vector_name="audio")
    assert n_tagged == 30
    assert "tags" in persisted["track-0"]

    # 2. Cluster
    result = svc.cluster_library(qdrant=qdrant, collection="e2e", audio_vector_name="audio", min_cluster_size=4)
    assert result["n_clusters"] >= 2

    # 3. Label (manual — simulating curator)
    reps = svc.get_cluster_representatives("e2e")
    # First cluster ids found (excluding noise) — give each a fake name
    cluster_ids = sorted({r["cluster_id"] for r in reps})
    labels_map = {cid: f"Class-{cid}" for cid in cluster_ids}
    svc.save_cluster_labels(collection="e2e", labels=labels_map)

    # 4. Train
    status = svc.train_classifier(qdrant=qdrant, collection="e2e", audio_vector_name="audio")
    assert status["status"] == "ready"
    assert status["accuracy"] >= 0.8

    # 5. Apply in bulk
    n_pred = svc.apply_classifier_bulk(qdrant=qdrant, collection="e2e", audio_vector_name="audio")
    assert n_pred == 30
    assert "sonic_class" in persisted["track-0"]
    # Cluster A members and Cluster B members should fall into distinct classes
    class_assignments = {persisted[s]["sonic_class"] for s in slugs}
    assert len(class_assignments) == 2
