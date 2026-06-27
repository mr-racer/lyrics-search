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

import numpy as np

from app.resources.qdrant_utils import scroll_all

logger = logging.getLogger(__name__)

# collection,lang -> (point_count, TasteMapResponse)
_CACHE: dict[tuple[str, str], tuple[int, object]] = {}
_LOCK = threading.Lock()

_DISPLAY_FIELDS = ["title", "artist", "sonic_axes", "cover_art_path"]
_MIN_TRACKS = 8

# Each sonic axis is BIPOLAR; an island is labelled by the pole (high|low) it
# leans toward, z-scored against the project's reference axis stats (see build()).
_AXIS_WORDS = {
    "ru": {
        "energy": ("Энергичная", "Спокойная"),
        "vocal_lead": ("Вокальная", "Инструментальная"),
        "spacious": ("Просторная", "Камерная"),
        "experimental": ("Экспериментальная", "Ровная"),
        "brightness": ("Яркая", "Тёмная"),
        "acousticness": ("Акустичная", "Электронная"),
    },
    "en": {
        "energy": ("Energetic", "Calm"),
        "vocal_lead": ("Vocal", "Instrumental"),
        "spacious": ("Spacious", "Intimate"),
        "experimental": ("Experimental", "Steady"),
        "brightness": ("Bright", "Dark"),
        "acousticness": ("Acoustic", "Electronic"),
    },
}
_Z_THRESHOLD = 0.5   # how far a cluster must lean (in σ) from the library to earn a word


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

    # ── Sonic naming ──────────────────────────────────────────────────────
    # Z-score each track's axes with the SAME blended reference stats the
    # recommendations radar uses (collection stats shrunk toward the bundled
    # reference) — no per-library mean/std recomputed here. Then label each
    # island by the 1-2 axes whose mean z most exceeds the threshold.
    from app.resources.clap_features import (
        AXIS_NAMES, blend_axis_stats, load_axis_norm_reference,
    )
    from app.resources.metadata_db import MetadataDB

    words = _AXIS_WORDS.get(lang, _AXIS_WORDS["en"])
    baseline = "Разное" if lang == "ru" else "Mixed"
    axis_stats = blend_axis_stats(
        MetadataDB.get_axis_norm_stats(collection_name), load_axis_norm_reference(),
    )
    a_mean = (axis_stats or {}).get("mean") or {}
    a_std = (axis_stats or {}).get("std") or {}

    na = len(AXIS_NAMES)
    Z = np.full((n, na), np.nan, dtype=np.float64)   # per-track z (NaN = no signal)
    if axis_stats:
        for i in range(n):
            sa = disp[i].get("sonic_axes")
            if not isinstance(sa, dict):
                continue
            for ai, an in enumerate(AXIS_NAMES):
                raw = sa.get(an)
                if not isinstance(raw, (int, float)):
                    continue
                s = a_std.get(an, 0.0)
                Z[i, ai] = (raw - a_mean.get(an, 0.0)) / s if s > 1e-9 else 0.0
    finite = np.isfinite(Z)

    def _island_name(mask):
        fm = finite[mask]
        ccnt = fm.sum(0)
        if int(ccnt.sum()) == 0:
            return baseline
        czm = np.where(ccnt > 0, np.where(fm, Z[mask], 0.0).sum(0) / np.maximum(ccnt, 1), 0.0)
        order = [int(a) for a in np.argsort(-np.abs(czm))]
        strong = [a for a in order if ccnt[a] > 0 and abs(czm[a]) >= _Z_THRESHOLD]
        if not strong:
            return baseline

        def _w(a):
            hi, lo = words[AXIS_NAMES[a]]
            return hi if czm[a] >= 0 else lo

        if len(strong) >= 2:
            sep = " и " if lang == "ru" else " & "
            return f"{_w(strong[0])}{sep}{_w(strong[1]).lower()}"
        return _w(strong[0])

    clusters = []
    for j in range(C.shape[0]):
        mask = labels == j
        size = int(mask.sum())
        if size == 0:
            continue
        members = [ids[i] for i in range(n) if labels[i] == j]
        name = _island_name(mask)
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
