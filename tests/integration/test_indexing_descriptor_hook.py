"""Verify that indexing triggers tag computation and class prediction per track."""

import sys, types
sys.modules.setdefault("laion_clap", types.ModuleType("laion_clap"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_index_one_track_persists_descriptor(monkeypatch, tmp_path):
    """After indexing a track, MetadataDB should have its sonic_tags_json populated."""
    from app.services.sonic_descriptor_service import SonicDescriptorService

    # Stage a real service against tmp paths
    vocab_path = tmp_path / "prompts.json"
    vocab_path.write_text('{"version":1,"groups":{"e":["punchy","ambient"]}}')
    embs_path = tmp_path / "embs.npy"
    np.save(embs_path, np.eye(2, dtype=np.float32))
    svc = SonicDescriptorService(
        prompt_vocab_path=vocab_path,
        embeddings_path=embs_path,
        cluster_dir=tmp_path / "c",
        classifier_dir=tmp_path / "cls",
        top_k_tags=2,
    )

    persisted: list[tuple] = []
    monkeypatch.setattr(
        "app.resources.metadata_db.MetadataDB.upsert_sonic_descriptor",
        lambda song_slug, tags=None, sonic_class=None, confidence=None, audio_signature=None: persisted.append((song_slug, tags, sonic_class)),
    )

    # Direct call to the hook — index_track_descriptor is what the indexing pipeline will invoke
    audio_vec = np.array([0.9, 0.1], dtype=np.float32)
    svc.index_track_descriptor(collection="my_col", slug="abc", audio_vector=audio_vec)

    assert len(persisted) == 1
    slug, tags, sclass = persisted[0]
    assert slug == "abc"
    assert tags[0]["tag"] == "punchy"  # closest to (0.9, 0.1)
    assert sclass is None  # no classifier trained → class is None
