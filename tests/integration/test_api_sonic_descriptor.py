"""Integration tests for Sonic Descriptor API endpoints."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
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
