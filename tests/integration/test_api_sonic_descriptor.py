"""Integration tests for Sonic Descriptor API endpoints."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Ensure sonic_descriptor_service exists on app.state so the sonic-prompts
    # endpoints can be exercised even when the lifespan hasn't run (or Qdrant is
    # unavailable). The service is cheap to construct and only touches files.
    if not getattr(app.state, "sonic_descriptor_service", None):
        from app.services.sonic_descriptor_service import SonicDescriptorService
        app.state.sonic_descriptor_service = SonicDescriptorService()
    return TestClient(app)


def test_get_sonic_descriptor_unknown_returns_404(client, monkeypatch):
    monkeypatch.setattr(
        "app.resources.metadata_db.MetadataDB.get_sonic_descriptor",
        lambda slug: None,
    )
    r = client.get("/api/v1/library/sonic-descriptor/never-existed")
    assert r.status_code == 404


def test_get_sonic_descriptor_returns_persisted(client, monkeypatch):
    monkeypatch.setattr(
        "app.resources.metadata_db.MetadataDB.get_sonic_descriptor",
        lambda slug: {
            "tags": [{"tag": "anxious", "score": 0.72}],
            "sonic_class": "Indie melancholic",
            "sonic_class_confidence": 0.81,
            "audio_signature": None,
        },
    )
    r = client.get("/api/v1/library/sonic-descriptor/karma-police")
    assert r.status_code == 200
    body = r.json()
    assert body["sonic_class"] == "Indie melancholic"
    assert body["tags"][0]["tag"] == "anxious"


def test_get_sonic_prompts_returns_vocab(client, tmp_path, monkeypatch):
    vocab = {"version": 1, "groups": {"e": ["punchy", "ambient"]}}
    p = tmp_path / "prompts.json"
    import json
    p.write_text(json.dumps(vocab))
    monkeypatch.setattr("app.services.sonic_descriptor_service.DEFAULT_VOCAB_PATH", p)
    # Reset cached service if any
    if hasattr(app.state, "sonic_descriptor_service") and app.state.sonic_descriptor_service is not None:
        app.state.sonic_descriptor_service.prompt_vocab_path = p
        app.state.sonic_descriptor_service._prompts = None

    r = client.get("/api/v1/library/sonic-prompts")
    assert r.status_code == 200
    body = r.json()
    assert "groups" in body
    assert body["groups"]["e"] == ["punchy", "ambient"]


def test_put_sonic_prompts_updates_vocab_and_invalidates_cache(client, tmp_path, monkeypatch):
    vocab_path = tmp_path / "prompts.json"
    emb_path = tmp_path / "embs.npy"
    import json
    import numpy as np
    vocab_path.write_text(json.dumps({"version": 1, "groups": {"e": ["punchy"]}}))
    np.save(emb_path, np.zeros((1, 2), dtype=np.float32))

    if hasattr(app.state, "sonic_descriptor_service") and app.state.sonic_descriptor_service is not None:
        svc = app.state.sonic_descriptor_service
        svc.prompt_vocab_path = vocab_path
        svc.embeddings_path = emb_path
        svc._prompts = None
        svc._prompt_embeddings = None

    new_vocab = {"version": 2, "groups": {"e": ["explosive", "drifting"]}}
    r = client.put("/api/v1/library/sonic-prompts", json=new_vocab)
    assert r.status_code == 200
    assert json.loads(vocab_path.read_text()) == new_vocab
    # Embeddings cache should be deleted
    assert not emb_path.exists()


def test_post_cluster_discovery_returns_job_id(client, monkeypatch):
    captured: dict = {}

    async def fake_run_clustering(collection: str):
        captured["called_with"] = collection
        return {"n_tracks": 10, "n_clusters": 2}

    monkeypatch.setattr(
        "app.api.routes.library._run_cluster_discovery_job",
        fake_run_clustering,
    )

    r = client.post("/api/v1/library/cluster-discovery?collection=test_col")
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "pending"


def test_get_cluster_job_status_returns_state(client, monkeypatch):
    from app.api.routes import library as lib_routes
    lib_routes._CLUSTER_JOBS["job-xyz"] = {
        "status": "completed",
        "result": {"n_tracks": 10, "n_clusters": 2},
        "error": None,
    }
    r = client.get("/api/v1/library/cluster-jobs/job-xyz")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["result"]["n_clusters"] == 2


def test_get_cluster_job_unknown_returns_404(client):
    r = client.get("/api/v1/library/cluster-jobs/does-not-exist")
    assert r.status_code == 404


def test_get_cluster_representatives_returns_list(client, monkeypatch):
    fake_reps = [
        {
            "cluster_id": 0,
            "size": 12,
            "representative_tracks": [
                {"track_id": "t1", "title": "Song", "artist": "Artist", "cover_art_path": None},
            ],
            "current_label": "Lo-fi indie",
        }
    ]
    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.get_cluster_representatives",
        lambda self, collection: fake_reps,
    )
    r = client.get("/api/v1/library/clusters/representatives?collection=test_col")
    assert r.status_code == 200
    assert r.json() == fake_reps


def test_post_cluster_labels_persists(client, monkeypatch):
    captured: dict = {}

    def fake_save(self, collection, labels):
        captured["collection"] = collection
        captured["labels"] = labels

    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.save_cluster_labels",
        fake_save,
    )

    r = client.post(
        "/api/v1/library/clusters/labels",
        json={"collection": "test_col", "labels": {"0": "Lo-fi indie", "1": "Cinematic drone"}},
    )
    assert r.status_code == 200
    assert captured["collection"] == "test_col"
    assert captured["labels"] == {0: "Lo-fi indie", 1: "Cinematic drone"}


def test_get_sonic_clusters_merges_labels_with_assignments(client, monkeypatch):
    fake_labels = {0: "Lo-fi indie", 1: "Cinematic drone"}
    fake_assignments = {"slug-a": 0, "slug-b": 0, "slug-c": 1, "slug-d": -1}

    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.load_cluster_labels",
        lambda self, collection: fake_labels,
    )
    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.load_cluster_assignments",
        lambda self, collection: fake_assignments,
    )

    r = client.get("/api/v1/library/sonic-clusters?collection=test_col")
    assert r.status_code == 200
    body = r.json()
    by_id = {c["cluster_id"]: c for c in body}
    assert by_id[0]["label"] == "Lo-fi indie"
    assert by_id[0]["size"] == 2
    assert sorted(by_id[0]["member_slugs"]) == ["slug-a", "slug-b"]
    assert -1 not in by_id


def test_get_sonic_clusters_empty_when_no_labels(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.load_cluster_labels",
        lambda self, collection: {},
    )
    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.load_cluster_assignments",
        lambda self, collection: {},
    )
    r = client.get("/api/v1/library/sonic-clusters?collection=anything")
    assert r.status_code == 200
    assert r.json() == []


def test_post_train_classifier_returns_job_id(client, monkeypatch):
    captured: dict = {}

    async def fake_train(collection: str):
        captured["collection"] = collection
        return {"status": "ready", "accuracy": 0.85, "classes": ["A", "B"]}

    monkeypatch.setattr(
        "app.api.routes.library._run_classifier_training_job",
        fake_train,
    )

    r = client.post("/api/v1/library/sonic-classifier/train?collection=test_col")
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body


def test_get_classifier_status_passes_through(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.sonic_descriptor_service.SonicDescriptorService.get_classifier_status",
        lambda self, collection: {
            "status": "ready",
            "trained_at": 12345.0,
            "accuracy": 0.78,
            "classes": ["Lo-fi indie", "Drone"],
        },
    )
    r = client.get("/api/v1/library/sonic-classifier/status?collection=test_col")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["classes"] == ["Lo-fi indie", "Drone"]
