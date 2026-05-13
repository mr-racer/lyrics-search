"""Sonic Descriptor Layer service.

Translates opaque CLAP audio embeddings into interpretable descriptors:
- Adjective tags via zero-shot CLAP-text-prompt similarity (no training)
- sonic_class via HDBSCAN clustering + user-curated labels + sklearn MLPClassifier

Outputs are persisted to ``songs.sonic_tags_json`` / ``sonic_class`` / ``sonic_class_confidence``.

The service is designed as a long-lived singleton (one per app lifetime). Heavy resources
(prompt embeddings, trained classifier, cluster assignments) are cached lazily.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VOCAB_PATH = ROOT / "cache" / "sonic_prompts.json"
DEFAULT_EMBEDDINGS_PATH = ROOT / "cache" / "sonic_prompts_embeddings.npy"
DEFAULT_CLUSTER_DIR = ROOT / "cache" / "sonic_clusters"
DEFAULT_CLASSIFIER_DIR = ROOT / "cache" / "sonic_classifier"


class SonicDescriptorService:
    """Compute and persist interpretable descriptors for tracks.

    Usage::

        svc = SonicDescriptorService()
        tags = svc.compute_tags(track_id="abc", audio_vector=np.array(...))
        # → [SonicTag(tag="anxious", score=0.72), ...]
    """

    def __init__(
        self,
        prompt_vocab_path: Path = DEFAULT_VOCAB_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        cluster_dir: Path = DEFAULT_CLUSTER_DIR,
        classifier_dir: Path = DEFAULT_CLASSIFIER_DIR,
        top_k_tags: int = 5,
        min_class_confidence: float = 0.4,
    ) -> None:
        self.prompt_vocab_path = Path(prompt_vocab_path)
        self.embeddings_path = Path(embeddings_path)
        self.cluster_dir = Path(cluster_dir)
        self.classifier_dir = Path(classifier_dir)
        self.top_k_tags = top_k_tags
        self.min_class_confidence = min_class_confidence

        # Lazy-loaded caches
        self._prompts: Optional[list[str]] = None
        self._prompt_embeddings: Optional[np.ndarray] = None

    def load_prompt_vocab(self) -> list[str]:
        """Return flat list of prompts from the vocab JSON, cached after first read."""
        if self._prompts is not None:
            return self._prompts
        with self.prompt_vocab_path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        flat: list[str] = []
        for group_prompts in doc.get("groups", {}).values():
            flat.extend(group_prompts)
        self._prompts = flat
        logger.info("[SonicDescriptor] loaded %d prompts from %s", len(flat), self.prompt_vocab_path)
        return flat

    def load_or_compute_prompt_embeddings(self) -> np.ndarray:
        """Return CLAP text embeddings for the vocabulary, computing & caching on first call.

        Cache invalidation is tied to the vocab file: callers who edit ``sonic_prompts.json``
        should also delete ``sonic_prompts_embeddings.npy`` (or the PUT endpoint does it).
        """
        if self._prompt_embeddings is not None:
            return self._prompt_embeddings

        if self.embeddings_path.exists():
            arr = np.load(self.embeddings_path)
            self._prompt_embeddings = arr
            logger.info("[SonicDescriptor] loaded cached prompt embeddings %s from %s", arr.shape, self.embeddings_path)
            return arr

        from app.resources.model_registry import ModelRegistry
        clap = ModelRegistry.load_clap()
        prompts = self.load_prompt_vocab()
        emb = clap.get_text_embedding(prompts, use_tensor=False)
        arr = np.asarray(emb, dtype=np.float32)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, arr)
        self._prompt_embeddings = arr
        logger.info("[SonicDescriptor] computed and cached %s prompt embeddings to %s", arr.shape, self.embeddings_path)
        return arr

    def compute_tags(self, audio_vector: np.ndarray) -> list[dict]:
        """Return top-K {tag, score} for the given audio embedding via CLAP-prompt cosine sim.

        ``audio_vector`` should already be normalized (CLAP output is). Prompt embeddings are
        normalized inside this method to make scores interpretable as cosines in [-1, 1] —
        in practice CLAP outputs are unit-norm so scores fall in [0, 1].
        """
        prompts = self.load_prompt_vocab()
        prompt_embs = self.load_or_compute_prompt_embeddings()

        # Normalize prompt embeddings row-wise
        norms = np.linalg.norm(prompt_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero
        prompt_unit = prompt_embs / norms

        # Normalize audio vector
        audio = audio_vector.astype(np.float32)
        a_norm = np.linalg.norm(audio)
        if a_norm > 0:
            audio = audio / a_norm

        sims = prompt_unit @ audio  # shape (N_prompts,)
        order = np.argsort(-sims)[: self.top_k_tags]
        return [{"tag": prompts[i], "score": float(sims[i])} for i in order]

    def compute_tags_bulk(
        self,
        qdrant,
        collection: str,
        audio_vector_name: str = "audio",
        batch_size: int = 500,
    ) -> int:
        """Scroll all points in ``collection``, compute tags per track, persist to MetadataDB.

        Returns the number of tracks processed.
        """
        from app.resources.metadata_db import MetadataDB

        offset = None
        n_processed = 0
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                offset=offset,
                limit=batch_size,
                with_payload=["slug", "title", "artist"],
                with_vectors=[audio_vector_name],
            )
            if not points:
                break
            for p in points:
                vec = (p.vector or {}).get(audio_vector_name) if isinstance(p.vector, dict) else None
                if vec is None:
                    continue
                slug = (p.payload or {}).get("slug") or str(p.id)
                tags = self.compute_tags(np.asarray(vec, dtype=np.float32))
                MetadataDB.upsert_sonic_descriptor(song_slug=slug, tags=tags)
                n_processed += 1
            if next_offset is None:
                break
            offset = next_offset
        logger.info("[SonicDescriptor] bulk-tagged %d tracks in collection %s", n_processed, collection)
        return n_processed
