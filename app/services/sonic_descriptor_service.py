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
import threading
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

    # Per-collection lazy classifier cache to avoid disk reads on every
    # prediction. Per-collection isolation is enforced by the dict's keying
    # (already correct). Phase B adds a lock so two concurrent first-time
    # predict_class calls for the SAME collection don't both joblib.load the
    # model (which would burn IO + RAM loading the same file twice).
    _classifier_cache: dict[str, tuple] = {}
    _classifier_cache_lock: threading.Lock = threading.Lock()

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

    def index_track_descriptor(
        self,
        collection: str,
        slug: str,
        audio_vector: np.ndarray,
    ) -> None:
        """Convenience hook for the indexing pipeline: compute tags + class for one track, persist.

        Used by ``LibraryService`` after each CLAP-encoded track is upserted to Qdrant.
        Safe to call without a trained classifier — class will be persisted as NULL.
        """
        from app.resources.metadata_db import MetadataDB
        tags = self.compute_tags(audio_vector=audio_vector)
        label, conf = self.predict_class(collection=collection, audio_vector=audio_vector)
        MetadataDB.upsert_sonic_descriptor(
            song_slug=slug,
            tags=tags,
            sonic_class=label,
            confidence=conf,
        )

    def compute_tags_bulk(
        self,
        qdrant,
        collection: str,
        audio_vector_name: str = "clap",
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

    def cluster_library(
        self,
        qdrant,
        collection: str,
        audio_vector_name: str = "clap",
        min_cluster_size: int = 5,
    ) -> dict:
        """Run HDBSCAN over all CLAP vectors in collection, persist assignments + representatives.

        Returns ``{"n_tracks": int, "n_clusters": int}``.
        """
        import hdbscan

        # 1. Collect all vectors + slugs (need both for persistence + representatives)
        vectors: list[np.ndarray] = []
        slugs: list[str] = []
        payloads: list[dict] = []

        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                offset=offset,
                limit=500,
                with_payload=["slug", "title", "artist", "cover_art_path"],
                with_vectors=[audio_vector_name],
            )
            if not points:
                break
            for p in points:
                vec = (p.vector or {}).get(audio_vector_name) if isinstance(p.vector, dict) else None
                if vec is None:
                    continue
                vectors.append(np.asarray(vec, dtype=np.float32))
                pl = p.payload or {}
                slugs.append(pl.get("slug") or str(p.id))
                payloads.append({
                    "track_id": str(p.id),
                    "title": pl.get("title") or "",
                    "artist": pl.get("artist") or "",
                    "cover_art_path": pl.get("cover_art_path"),
                })
            if next_offset is None:
                break
            offset = next_offset

        if not vectors:
            logger.warning("[SonicDescriptor] cluster_library: collection %s is empty", collection)
            return {"n_tracks": 0, "n_clusters": 0}

        X = np.vstack(vectors).astype(np.float32)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)

        # 2. Persist assignments {slug: cluster_id}. Noise (label=-1) preserved but
        #    excluded from representatives.
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        assignments_path = self.cluster_dir / f"{collection}_assignments.json"
        assignments = {slugs[i]: int(labels[i]) for i in range(len(slugs))}
        assignments_path.write_text(json.dumps(assignments))

        # 3. Compute representatives (top-5 closest to centroid per cluster)
        clusters_meta = self._compute_representatives(X, labels, payloads, top_n=5)
        reps_path = self.cluster_dir / f"{collection}_representatives.json"
        reps_path.write_text(json.dumps(clusters_meta))

        unique_labels = set(int(l) for l in labels) - {-1}
        logger.info(
            "[SonicDescriptor] clustered %d tracks into %d clusters (collection=%s)",
            len(slugs), len(unique_labels), collection,
        )
        return {"n_tracks": len(slugs), "n_clusters": len(unique_labels)}

    def _compute_representatives(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        payloads: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """For each non-noise cluster, return top-N tracks closest to its centroid."""
        out: list[dict] = []
        for cid in sorted({int(l) for l in labels} - {-1}):
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                continue
            centroid = X[idxs].mean(axis=0)
            dists = np.linalg.norm(X[idxs] - centroid, axis=1)
            order = np.argsort(dists)[:top_n]
            reps = [payloads[idxs[k]] for k in order]
            out.append({"cluster_id": cid, "size": int(len(idxs)), "representative_tracks": reps})
        return out

    def get_cluster_representatives(self, collection: str) -> list[dict]:
        """Read persisted representatives JSON for a collection.

        Returns empty list if cluster_library was never run for this collection.
        Merges current_label from labels JSON if present.
        """
        reps_path = self.cluster_dir / f"{collection}_representatives.json"
        if not reps_path.exists():
            return []
        reps = json.loads(reps_path.read_text())

        labels_path = self.cluster_dir / f"{collection}_labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        for r in reps:
            r["current_label"] = labels.get(str(r["cluster_id"]))
        return reps

    def save_cluster_labels(self, collection: str, labels: dict[int, str]) -> None:
        """Persist user-assigned cluster names. Overwrites any existing labels file."""
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        path = self.cluster_dir / f"{collection}_labels.json"
        # JSON-safe keys (stringify ints)
        path.write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False))
        logger.info("[SonicDescriptor] saved %d cluster labels for %s", len(labels), collection)

    def load_cluster_labels(self, collection: str) -> dict[int, str]:
        """Return labels dict with int keys, or empty dict if no labels file exists."""
        path = self.cluster_dir / f"{collection}_labels.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text())
        return {int(k): v for k, v in raw.items()}

    def load_cluster_assignments(self, collection: str) -> dict[str, int]:
        """Return {slug: cluster_id} mapping for a collection, or {} if no clustering done."""
        path = self.cluster_dir / f"{collection}_assignments.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def train_classifier(
        self,
        qdrant,
        collection: str,
        audio_vector_name: str = "clap",
        test_size: float = 0.2,
        random_state: int = 0,
    ) -> dict:
        """Train sklearn MLP on (CLAP vector, cluster_label) pairs for this collection.

        Returns ``{"status": "ready"|"failed", "accuracy": float, "trained_at": ts, "classes": [...]}``.
        Persists model to ``classifier_dir/<collection>.joblib`` and meta to ``<collection>_meta.json``.
        """
        import time
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier

        labels_map = self.load_cluster_labels(collection)
        if not labels_map:
            raise RuntimeError(f"No cluster labels found for collection '{collection}'. Run curator first.")

        assignments_path = self.cluster_dir / f"{collection}_assignments.json"
        if not assignments_path.exists():
            raise RuntimeError(f"No cluster assignments for '{collection}'. Run cluster_library first.")
        assignments: dict[str, int] = json.loads(assignments_path.read_text())

        # Build labeled set: only tracks whose cluster has a user-assigned label, exclude noise (-1)
        labeled_slugs: list[str] = []
        labeled_y: list[str] = []
        for slug, cid in assignments.items():
            if cid == -1:
                continue
            label = labels_map.get(int(cid))
            if label is None:
                continue  # cluster not yet labeled
            labeled_slugs.append(slug)
            labeled_y.append(label)

        if len(labeled_slugs) < 2:
            raise RuntimeError(f"Need at least 2 labeled tracks to train; got {len(labeled_slugs)}.")

        # Fetch vectors for labeled slugs only
        slug_to_vec: dict[str, np.ndarray] = {}
        offset = None
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                offset=offset,
                limit=500,
                with_payload=["slug"],
                with_vectors=[audio_vector_name],
            )
            if not points:
                break
            for p in points:
                pl_slug = (p.payload or {}).get("slug") or str(p.id)
                vec = (p.vector or {}).get(audio_vector_name) if isinstance(p.vector, dict) else None
                if vec is not None and pl_slug in set(labeled_slugs):
                    slug_to_vec[pl_slug] = np.asarray(vec, dtype=np.float32)
            if next_offset is None:
                break
            offset = next_offset

        X = np.vstack([slug_to_vec[s] for s in labeled_slugs if s in slug_to_vec])
        y = [labeled_y[i] for i, s in enumerate(labeled_slugs) if s in slug_to_vec]

        if len(set(y)) < 2:
            raise RuntimeError(f"Need at least 2 distinct class labels to train; got {set(y)}.")

        # Train/test split (stratified if possible)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
        )

        model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=random_state)
        model.fit(X_train, y_train)
        accuracy = float(model.score(X_test, y_test))

        # Persist
        self.classifier_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.classifier_dir / f"{collection}.joblib"
        joblib.dump(model, model_path)

        classes = sorted(set(y))
        meta = {
            "status": "ready",
            "trained_at": time.time(),
            "accuracy": accuracy,
            "classes": classes,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        meta_path = self.classifier_dir / f"{collection}_meta.json"
        meta_path.write_text(json.dumps(meta))
        # Invalidate cached classifier so subsequent predict_class calls reload from disk
        self._classifier_cache.pop(collection, None)
        logger.info(
            "[SonicDescriptor] trained classifier for '%s': accuracy=%.3f, classes=%s",
            collection, accuracy, classes,
        )
        return meta

    def _load_classifier(self, collection: str) -> Optional[tuple]:
        """Return (model, classes_list) tuple, or None if no trained classifier exists.

        Thread-safe lazy load via double-checked locking: a cheap lock-free dict
        lookup first, then lock + re-check only on a miss so eight concurrent
        first-time callers for the same collection trigger exactly one
        ``joblib.load`` (Phase B §6.3).
        """
        cached = self._classifier_cache.get(collection)
        if cached is not None:
            return cached
        model_path = self.classifier_dir / f"{collection}.joblib"
        meta_path = self.classifier_dir / f"{collection}_meta.json"
        if not model_path.exists() or not meta_path.exists():
            return None
        with self._classifier_cache_lock:
            cached = self._classifier_cache.get(collection)
            if cached is not None:
                return cached
            import joblib
            model = joblib.load(model_path)
            meta = json.loads(meta_path.read_text())
            self._classifier_cache[collection] = (model, meta["classes"])
            return self._classifier_cache[collection]

    def predict_class(
        self,
        collection: str,
        audio_vector: np.ndarray,
    ) -> tuple[Optional[str], Optional[float]]:
        """Predict sonic_class for an audio vector. Returns (label, confidence).

        Returns (None, None) if no classifier exists for this collection.
        Returns (None, conf) if conf < ``self.min_class_confidence`` (label suppressed but score reported).
        """
        loaded = self._load_classifier(collection)
        if loaded is None:
            return None, None
        model, _classes = loaded
        v = audio_vector.reshape(1, -1)
        probs = model.predict_proba(v)[0]
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])
        # model.classes_ holds the class labels (string array)
        label = str(model.classes_[top_idx])
        if conf < self.min_class_confidence:
            return None, conf
        return label, conf

    def get_classifier_status(self, collection: str) -> dict:
        """Return readiness info for the classifier on this collection."""
        meta_path = self.classifier_dir / f"{collection}_meta.json"
        if not meta_path.exists():
            return {"status": "untrained", "trained_at": None, "accuracy": None, "classes": []}
        meta = json.loads(meta_path.read_text())
        return {
            "status": meta.get("status", "ready"),
            "trained_at": meta.get("trained_at"),
            "accuracy": meta.get("accuracy"),
            "classes": meta.get("classes", []),
        }

    def apply_classifier_bulk(
        self,
        qdrant,
        collection: str,
        audio_vector_name: str = "clap",
        batch_size: int = 500,
    ) -> int:
        """Predict sonic_class for every track in collection and persist to MetadataDB.

        Skips tracks where the classifier returns (None, None) (no classifier trained).
        Returns the count of tracks processed.
        """
        from app.resources.metadata_db import MetadataDB

        # If no classifier exists, return 0 — nothing to do
        if self._load_classifier(collection) is None:
            logger.info("[SonicDescriptor] apply_classifier_bulk: no classifier for %s", collection)
            return 0

        offset = None
        n = 0
        while True:
            points, next_offset = qdrant.scroll(
                collection_name=collection,
                offset=offset,
                limit=batch_size,
                with_payload=["slug"],
                with_vectors=[audio_vector_name],
            )
            if not points:
                break
            for p in points:
                vec = (p.vector or {}).get(audio_vector_name) if isinstance(p.vector, dict) else None
                if vec is None:
                    continue
                slug = (p.payload or {}).get("slug") or str(p.id)
                label, conf = self.predict_class(collection=collection, audio_vector=np.asarray(vec, dtype=np.float32))
                MetadataDB.upsert_sonic_descriptor(song_slug=slug, sonic_class=label, confidence=conf)
                n += 1
            if next_offset is None:
                break
            offset = next_offset
        logger.info("[SonicDescriptor] applied classifier to %d tracks in %s", n, collection)
        return n
