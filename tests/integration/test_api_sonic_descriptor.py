"""Integration tests for Sonic Descriptor API endpoints."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ._auth_helper import authenticate_test_client


@pytest.fixture
def client():
    # Ensure sonic_descriptor_service exists on app.state so the sonic-prompts
    # endpoints can be exercised even when the lifespan hasn't run (or Qdrant is
    # unavailable). The service is cheap to construct and only touches files.
    if not getattr(app.state, "sonic_descriptor_service", None):
        from app.services.sonic_descriptor_service import SonicDescriptorService
        app.state.sonic_descriptor_service = SonicDescriptorService()
    c = TestClient(app)
    authenticate_test_client(c, app)
    return c


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
