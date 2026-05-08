"""Similarity analysis service — compute top-similar / top-dissimilar track pairs."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .job_tracker import IndexStage, IndexStatus

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "top_pairs"


def compute_similarity_matrix(vectors: List[List[float]]) -> np.ndarray:
    """Compute pairwise cosine distance matrix.

    Args:
        vectors: List of CLAP vectors (each is a list of floats).

    Returns:
        NxN distance matrix with inf on diagonal (self-similarity ignored).
    """
    dist_matrix = pairwise_distances(vectors, metric="cosine")
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix


def get_top_pairs(
    dist_matrix: np.ndarray,
    ids: List[str],
    id2name: Dict[str, str],
    id2payload: Dict[str, dict],
    top_k: int = 5,
) -> tuple:
    """Extract top-K most similar and most dissimilar pairs per track.

    Returns:
        (similar_list, dissimilar_list) — each is a list of dicts:
        [
          {
            "song": "Artist - Title",
            "track_id": "<track_id>",
            "cover_art_path": "<path or null>",
            "top_similar": [
              {
                "name": "Artist - Title",
                "track_id": "<track_id>",
                "cover_art_path": "<path or null>",
                "score": 92.3
              },
              ...
            ]
          },
          ...
        ]
    """
    n = dist_matrix.shape[0]
    k = min(top_k, n - 1)

    similar: List[dict] = []
    dissimilar: List[dict] = []

    for i in range(n):
        sorted_idx = np.argsort(dist_matrix[i])

        # Similar: smallest distances
        sim_idx = sorted_idx[:k]
        sim_scores = (1.0 - dist_matrix[i, sim_idx] / 2.0) * 100.0

        # Dissimilar: largest distances (skip inf on diagonal), most-different first
        diss_idx = sorted_idx[-(k + 1):-1][::-1]
        diss_scores = (1.0 - dist_matrix[i, diss_idx] / 2.0) * 100.0

        similar.append(
            {
                "song": id2name[ids[i]],
                "track_id": ids[i],
                "cover_art_path": id2payload.get(ids[i], {}).get("cover_art_path"),
                "top_similar": [
                    {
                        "name": id2name[ids[j]],
                        "track_id": ids[j],
                        "cover_art_path": id2payload.get(ids[j], {}).get("cover_art_path"),
                        "score": round(float(s), 1),
                    }
                    for j, s in zip(sim_idx, sim_scores)
                ],
            }
        )
        dissimilar.append(
            {
                "song": id2name[ids[i]],
                "track_id": ids[i],
                "cover_art_path": id2payload.get(ids[i], {}).get("cover_art_path"),
                "top_dissimilar": [
                    {
                        "name": id2name[ids[j]],
                        "track_id": ids[j],
                        "cover_art_path": id2payload.get(ids[j], {}).get("cover_art_path"),
                        "score": round(float(s), 1),
                    }
                    for j, s in zip(diss_idx, diss_scores)
                ],
            }
        )

    return similar, dissimilar


def save_top_pairs(similar: List[dict], dissimilar: List[dict], collection_name: str) -> str:
    """Save top pairs to cache file.

    Returns:
        Path to the cached JSON file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{collection_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "similar": similar,
                "dissimilar": dissimilar,
                "collection_name": collection_name,
                "computed_at": time.time(),
            },
            f,
            ensure_ascii=False,
        )
    return str(path)


def load_top_pairs(collection_name: str) -> Optional[dict]:
    """Load cached top pairs for a collection.

    Returns:
        Dict with 'similar', 'dissimilar', 'collection_name', 'computed_at' — or None if not cached.
    """
    path = CACHE_DIR / f"{collection_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def analyze_collection(
    qdrant_client,
    collection_name: str,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Run full similarity analysis on a Qdrant collection.

    Scrolls all points, computes CLAP distance matrix, extracts top-5 pairs,
    and saves to cache.

    Args:
        qdrant_client: QdrantClient instance.
        collection_name: Collection to analyze.
        progress_callback: Async callback(stage, current, total, message).

    Returns:
        Path to the cached JSON file.

    Raises:
        ValueError: If collection has no CLAP vectors.
    """
    if progress_callback:
        await progress_callback(
            IndexStage.ANALYSIS, 0, 1, "Загрузка векторов из Qdrant..."
        )

    # ── Step 1: Scroll all points with CLAP vectors + payload ──
    points = []
    offset = None
    while True:
        response, next_offset = qdrant_client.scroll(
            collection_name,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(response)
        if next_offset is None:
            break
        offset = next_offset

    if not points:
        raise ValueError(f"Collection '{collection_name}' is empty")

    # Extract CLAP vectors — skip points that lack them
    vectors_map: Dict[str, List[float]] = {}
    id2name: Dict[str, str] = {}
    id2payload: Dict[str, dict] = {}

    for pt in points:
        vec = pt.vector.get("clap") if isinstance(pt.vector, dict) else None
        if vec is None:
            continue
        vectors_map[pt.id] = vec
        pl = pt.payload or {}
        artist = pl.get("artist", "Unknown")
        title = pl.get("title", "Unknown")
        id2name[pt.id] = f"{artist} - {title}"
        id2payload[pt.id] = pl

    if not vectors_map:
        raise ValueError(
            f"No CLAP vectors found in collection '{collection_name}'. "
            "Re-index with audio embedding enabled."
        )

    ids = list(vectors_map.keys())
    vectors = [vectors_map[i] for i in ids]

    if progress_callback:
        await progress_callback(
            IndexStage.ANALYSIS,
            len(vectors),
            len(vectors),
            f"Загружено {len(vectors)} векторов",
        )

    # ── Step 2: Compute distance matrix ──
    if progress_callback:
        await progress_callback(
            IndexStage.ANALYSIS, 0, 1, "Вычисление матрицы расстояний..."
        )

    dist_matrix = compute_similarity_matrix(vectors)

    if progress_callback:
        await progress_callback(
            IndexStage.ANALYSIS, 0, 1, "Подбор топ-5 пар..."
        )

    # ── Step 3: Extract top pairs ──
    similar, dissimilar = get_top_pairs(dist_matrix, ids, id2name, id2payload, top_k=5)

    # ── Step 4: Save to cache ──
    cache_path = save_top_pairs(similar, dissimilar, collection_name)

    if progress_callback:
        await progress_callback(
            IndexStage.ANALYSIS,
            1,
            1,
            f"Анализ завершён, кэш: {cache_path}",
        )

    logger.info(
        "[SimilarityService] Analysis complete for '%s': %d tracks, cache at %s",
        collection_name,
        len(vectors),
        cache_path,
    )

    return cache_path
