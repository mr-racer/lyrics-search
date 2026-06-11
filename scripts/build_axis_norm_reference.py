"""Build the reference axis-normalisation stats from an indexed Qdrant collection.

The reference keeps z-scores honest for small collections via shrinkage
(stream recsys design §8). Run ONCE against a large, diverse, already-indexed
collection — CLAP audio vectors are read straight from Qdrant, audio is NOT
re-encoded (only the text prompts pass through the CLAP text tower).

Usage
-----
  python -m scripts.build_axis_norm_reference --collection acct_<owner_id>
  python -m scripts.build_axis_norm_reference --collection big_lib --dry-run

Output: app/resources/data/axis_norm_reference.json (override with --output).
The file embeds axis_version(); when prompts or the CLAP checkpoint change,
readers ignore the stale file and this script must be re-run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient

from app.resources.clap_features import (
    AXIS_NAMES,
    AXIS_NORM_REFERENCE_PATH,
    axes_for_clap_vectors,
    axis_version,
    compute_axis_text_embeddings,
)

logger = logging.getLogger("build_axis_norm_reference")

BATCH = 256
MIN_TRACKS = 2  # std (ddof=1) needs at least two samples


def _iter_clap_vectors(qdrant: QdrantClient, collection: str) -> Iterable[list]:
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=collection,
            limit=BATCH,
            with_payload=False,
            with_vectors=["clap"],
            offset=offset,
        )
        if not points:
            return
        for p in points:
            vec = p.vector.get("clap") if isinstance(p.vector, dict) else None
            if vec:
                yield vec
        if next_offset is None:
            return
        offset = next_offset


def build_reference(qdrant: QdrantClient, collection: str, clap_model=None) -> dict:
    """Compute axis mean/std over every CLAP vector in ``collection``."""
    vectors = [np.asarray(v, dtype=np.float32) for v in _iter_clap_vectors(qdrant, collection)]
    if len(vectors) < MIN_TRACKS:
        raise SystemExit(
            f"collection {collection!r} has only {len(vectors)} CLAP vector(s) — "
            f"need >= {MIN_TRACKS}. Was it indexed with CLAP enabled?"
        )
    logger.info("collected %d CLAP vectors from %r", len(vectors), collection)

    if clap_model is None:
        from app.resources.model_registry import ModelRegistry
        clap_model = ModelRegistry.load_clap()
    text_emb = compute_axis_text_embeddings(clap_model)

    axes_dicts = axes_for_clap_vectors(np.stack(vectors), text_emb)
    arr = np.array([[d[a] for a in AXIS_NAMES] for d in axes_dicts], dtype=float)
    return {
        "version": axis_version(),
        "n": int(arr.shape[0]),
        "source_collection": collection,
        "mean": {a: float(v) for a, v in zip(AXIS_NAMES, arr.mean(axis=0))},
        "std": {a: float(v) for a, v in zip(AXIS_NAMES, arr.std(axis=0, ddof=1))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True,
                        help="Indexed Qdrant collection to use as the reference library")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--output", type=Path, default=AXIS_NORM_REFERENCE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the stats without writing the file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    qdrant = QdrantClient(url=args.qdrant_url)
    stats = build_reference(qdrant, args.collection)

    payload = json.dumps(stats, indent=2)
    if args.dry_run:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    logger.info("reference written: %s (n=%d, version=%s)",
                args.output, stats["n"], stats["version"])


if __name__ == "__main__":
    main()
