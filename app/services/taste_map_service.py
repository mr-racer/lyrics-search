"""Taste Map — 2D projection + clustering of a user's library for the
"Сонар вкуса" stats visual.

Pulls CLAP audio vectors (never lyrics), reduces them to 2D with PCA, and groups
the projection into sonic "islands" via a small numpy k-means. Each island is
named by its dominant genre (plain listener language — no jargon, no LLM). The
result is cached per (collection, lang) and recomputed when the collection's
point count changes (i.e. on reindex).

This is intentionally a self-contained path: the existing taste-island clusterer
(`stream_service.long_term_profile`) groups *listening signals*, not the whole
library, so it can't drive a full-library map.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter

import numpy as np

from app.resources.qdrant_utils import scroll_all

logger = logging.getLogger(__name__)

# collection,lang -> (point_count, TasteMapResponse)
_CACHE: dict[tuple[str, str], tuple[int, object]] = {}
_LOCK = threading.Lock()

_DISPLAY_FIELDS = ["title", "artist", "genre", "cover_art_path"]
_MIN_TRACKS = 8


def invalidate(collection_name: str | None = None) -> None:
    """Drop cached maps (all, or for one collection)."""
    with _LOCK:
        if collection_name is None:
            _CACHE.clear()
        else:
            for k in [k for k in _CACHE if k[0] == collection_name]:
                _CACHE.pop(k, None)


def _clap_of(point):
    v = getattr(point, "vector", None)
    if isinstance(v, dict):
        v = v.get("clap")
    return v


def _kmeans(X, k, *, iters=24, seed=0):
    """Tiny Lloyd's k-means with k-means++ seeding (deterministic). X is (n, d)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    first = int(rng.integers(n))
    idx = [first]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(d2.sum())
        probs = (d2 / total) if total > 0 else None
        nxt = int(rng.choice(n, p=probs))
        idx.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(1))
    C = X[idx].astype(np.float64).copy()
    labels = np.full(n, -1)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)   # n x k
        nl = d.argmin(1)
        if np.array_equal(nl, labels):
            break
        labels = nl
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels, C


def build(*, qdrant_client, collection_name: str, lang: str = "en"):
    from app.domain.models import TasteMapResponse, TasteMapPoint, TasteMapCluster

    try:
        count = qdrant_client.get_collection(collection_name).points_count or 0
    except Exception:
        return TasteMapResponse()

    key = (collection_name, lang)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == count:
            return cached[1]

    ids: list[str] = []
    vecs: list = []
    disp: list[dict] = []
    try:
        for p in scroll_all(
            qdrant_client, collection_name, batch_size=512,
            with_payload=_DISPLAY_FIELDS, with_vectors=["clap"],
        ):
            v = _clap_of(p)
            if not v:
                continue
            ids.append(str(p.id))
            vecs.append(v)
            disp.append(p.payload or {})
    except Exception:
        logger.exception("[taste-map] scroll failed for %s", collection_name)
        return TasteMapResponse()

    n = len(ids)
    if n < _MIN_TRACKS:
        result = TasteMapResponse()
        with _LOCK:
            _CACHE[key] = (count, result)
        return result

    X = np.asarray(vecs, dtype=np.float32)
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(X)
    except Exception:
        logger.exception("[taste-map] PCA failed for %s", collection_name)
        return TasteMapResponse()

    coords = np.asarray(coords, dtype=np.float64)
    # uniform robust scale to ~[-1, 1] (preserve aspect; ignore far outliers)
    s = float(np.percentile(np.abs(coords), 98)) or 1.0
    coords = np.clip(coords / s, -1.05, 1.05)

    k = int(min(8, max(3, round(n / 40))))
    labels, C = _kmeans(coords, k, seed=0)

    mixed = "Разное" if lang == "ru" else "Mixed"
    clusters = []
    for j in range(C.shape[0]):
        mask = labels == j
        size = int(mask.sum())
        if size == 0:
            continue
        members = [ids[i] for i in range(n) if labels[i] == j]
        gen = Counter(
            (disp[i].get("genre") or "").strip()
            for i in range(n)
            if labels[i] == j and (disp[i].get("genre") or "").strip()
        )
        name = gen.most_common(1)[0][0] if gen else mixed
        pts = coords[mask]
        d = ((pts - C[j]) ** 2).sum(1)
        spread = float(np.sqrt(d.mean())) if size else 0.1
        order = np.argsort(d)
        samples = [members[int(o)] for o in order[:4]]
        clusters.append(TasteMapCluster(
            id=int(j), name=name, size=size,
            cx=round(float(C[j][0]), 4), cy=round(float(C[j][1]), 4),
            spread=round(max(0.08, min(spread, 0.9)), 4),
            sample_track_ids=samples,
        ))

    points = [
        TasteMapPoint(
            track_id=ids[i],
            x=round(float(coords[i][0]), 4), y=round(float(coords[i][1]), 4),
            cluster=int(labels[i]),
            title=disp[i].get("title") or "—",
            artist=disp[i].get("artist") or "—",
        )
        for i in range(n)
    ]
    result = TasteMapResponse(points=points, clusters=clusters)
    with _LOCK:
        _CACHE[key] = (count, result)
    return result
