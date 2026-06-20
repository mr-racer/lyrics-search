"""Similarity analysis service — compute top-similar / top-dissimilar track pairs."""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .job_tracker import IndexStage
from app.services._payload_coerce import coerce_float, coerce_year

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


def build_track_pairs(
    similar_neighbors: List[dict],
    dissimilar_neighbors: List[dict],
    payload_by_id: Dict[str, dict],
    top_k: int = 3,
    *,
    seed_album: Optional[str] = None,
    drop_same_album_similar: bool = False,
    min_duration: float = 0.0,
) -> dict:
    """Enrich the top-K neighbour lists with live Qdrant payload + cache score.

    Each neighbour comes from the cache as {track_id, name, cover_art_path, score}.
    We prefer the live ``payload_by_id`` (clean title/artist/genre, full fields for
    queue parity) and fall back to splitting the cached ``name`` ("Artist - Title"
    on the first " - ") when a track was deleted since the analysis ran.

    Returns {"similar": [<dict>], "dissimilar": [<dict>]} — each item has keys
    track_id, title, artist, album, genre, cover_art_path, duration_sec,
    file_path, producer, label, samples, year, score.

    Optional filtering (applied before the top_k cap):
    - ``seed_album`` + ``drop_same_album_similar=True``: drop same-album neighbours
      from the similar list only (contrast keeps same-album).
    - ``min_duration``: drop neighbours whose duration is between 0 and min_duration
      (exclusive). Missing/zero durations are treated as "unknown" and kept.
    """
    seed_alb = (seed_album or "").strip().lower()

    def _too_short(pl):
        if not min_duration:
            return False
        d = coerce_float((pl or {}).get("duration"))
        return d is not None and 0.0 < d < min_duration

    def _same_album(pl):
        a = ((pl or {}).get("album") or "").strip().lower()
        return bool(a) and bool(seed_alb) and a == seed_alb

    def _enrich(neighbors: List[dict], *, is_similar: bool) -> List[dict]:
        out: List[dict] = []
        for nb in neighbors:                       # no pre-slice — filter then cap
            tid = nb.get("track_id")
            pl = payload_by_id.get(tid) if tid is not None else None
            if pl is not None:
                if _too_short(pl):
                    continue
                if is_similar and drop_same_album_similar and _same_album(pl):
                    continue
                out.append({
                    "track_id": str(tid),
                    "title": pl.get("title") or "",
                    "artist": pl.get("artist") or "",
                    "album": pl.get("album"),
                    "genre": pl.get("genre"),
                    "cover_art_path": pl.get("cover_art_path") or nb.get("cover_art_path"),
                    "duration_sec": coerce_float(pl.get("duration")) or 0.0,
                    "file_path": pl.get("file_path") or "",
                    "producer": pl.get("producer"),
                    "label": pl.get("label"),
                    "samples": pl.get("samples"),
                    "year": coerce_year(pl.get("year")),
                    "score": nb.get("score"),
                })
            else:
                name = nb.get("name") or ""
                artist, sep, title = name.partition(" - ")
                if not sep:  # no separator → treat the whole string as the title
                    artist, title = "", name
                out.append({
                    "track_id": str(tid) if tid is not None else "",
                    "title": title,
                    "artist": artist,
                    "album": None,
                    "genre": None,
                    "cover_art_path": nb.get("cover_art_path"),
                    "duration_sec": 0.0,
                    "file_path": "",
                    "producer": None,
                    "label": None,
                    "samples": None,
                    "year": None,
                    "score": nb.get("score"),
                })
            if len(out) >= top_k:
                break
        return out

    return {
        "similar": _enrich(similar_neighbors, is_similar=True),
        "dissimilar": _enrich(dissimilar_neighbors, is_similar=False),
    }


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


# Parsed-file memo: the top-pairs JSON is large (the full similar/dissimilar
# matrix), so re-reading+parsing it on every /library/top-pairs hit is wasteful.
# Keyed by absolute path (CACHE_DIR is monkeypatched in tests) and invalidated by
# mtime — a recompute rewrites the file, so a stale parse is never served.
_LOAD_MEMO: dict[str, tuple[float, dict]] = {}
_LOAD_MEMO_MAX = 8

# Derived index memo: {track_id: {"similar": [...], "dissimilar": [...]}}, built
# over the already-parsed top-pairs file and invalidated by the same mtime. Lets
# the per-track player endpoint do an O(1) lookup instead of scanning the lists.
_INDEX_MEMO: dict[str, tuple[float, dict]] = {}


def clear_load_memo() -> None:
    """Drop the parsed-file + index memos (recompute invalidates by mtime; tests
    call this for isolation)."""
    _LOAD_MEMO.clear()
    _INDEX_MEMO.clear()


def load_top_pairs(collection_name: str) -> Optional[dict]:
    """Load cached top pairs for a collection.

    Returns:
        Dict with 'similar', 'dissimilar', 'collection_name', 'computed_at' — or None if not cached.
    """
    path = CACHE_DIR / f"{collection_name}.json"
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _LOAD_MEMO.pop(key, None)
        return None
    hit = _LOAD_MEMO.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _LOAD_MEMO[key] = (mtime, data)
    while len(_LOAD_MEMO) > _LOAD_MEMO_MAX:
        _LOAD_MEMO.pop(next(iter(_LOAD_MEMO)))
    return data


def load_top_pairs_index(collection_name: str) -> Optional[dict]:
    """Return {track_id: {"similar": [...], "dissimilar": [...]}} for O(1) lookup.

    Built over the memoized parsed file and cached in ``_INDEX_MEMO``, keyed by the
    same file mtime — a recompute rewrites the file, so a stale index is never
    served. Returns None when the collection has no cache file.
    """
    path = CACHE_DIR / f"{collection_name}.json"
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _INDEX_MEMO.pop(key, None)
        return None
    hit = _INDEX_MEMO.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    data = load_top_pairs(collection_name)
    if data is None:
        _INDEX_MEMO.pop(key, None)
        return None
    index: dict = {}
    for entry in data.get("similar", []):
        tid = entry.get("track_id")
        if tid is not None:
            index.setdefault(tid, {})["similar"] = entry.get("top_similar", [])
    for entry in data.get("dissimilar", []):
        tid = entry.get("track_id")
        if tid is not None:
            index.setdefault(tid, {})["dissimilar"] = entry.get("top_dissimilar", [])
    _INDEX_MEMO[key] = (mtime, index)
    while len(_INDEX_MEMO) > _LOAD_MEMO_MAX:
        _INDEX_MEMO.pop(next(iter(_INDEX_MEMO)))
    return index


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
        pl = pt.payload or {}
        dur = coerce_float(pl.get("duration"))
        if dur is not None and 0.0 < dur < 30.0:
            continue
        vectors_map[pt.id] = vec
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
            IndexStage.ANALYSIS, 0, 1, "Подбор топ-12 пар..."
        )

    # ── Step 3: Extract top pairs ──
    similar, dissimilar = get_top_pairs(dist_matrix, ids, id2name, id2payload, top_k=12)

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
