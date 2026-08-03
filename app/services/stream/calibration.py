"""CLAP cosine → percentile calibration (design §5).

CLAP cosines sit in a narrow band, and where that band sits depends on the
library: 0.85 can mean «near-duplicate» in one collection and «vaguely the same
genre» in another. Every absolute threshold in the recsys (anchor merge, skip
forgiveness, explore negativity) therefore had to be hand-tuned and still
travelled badly.

This module samples the collection once, builds a 101-point quantile table of
the pairwise cosine distribution, and exposes ``sim_pct(cos) ∈ [0, 1]`` — «how
similar is this pair *by the standards of this library*». Thresholds expressed
as percentiles mean the same thing everywhere.

The table lives in ``collection_settings.clap_calibration`` and is rebuilt when
the track count drifts past CALIB_DRIFT or the table ages past CALIB_TTL_DAYS.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime

import numpy as np

from app.resources.metadata_db import MetadataDB
from app.resources.qdrant_utils import scroll_all

logger = logging.getLogger(__name__)

CALIB_VERSION = 1
CALIB_SAMPLE = 512        # tracks sampled → 512×512 pairwise cosines
CALIB_MIN_TRACKS = 32     # below this a distribution is meaningless
CALIB_SCROLL_CAP = 4000   # points scanned before sampling (bounds the scroll)
CALIB_DRIFT = 0.10        # rebuild once the track count moved this much
CALIB_TTL_DAYS = 7.0
N_QUANTILES = 101         # percentile 0..100 inclusive


class Calibration:
    """Cosine → percentile mapping for one collection.

    ``source`` is ``"table"`` when backed by a real quantile table and
    ``"raw"`` for the degraded fallback (no table, and none buildable): there
    ``sim_pct`` returns ``(cos + 1) / 2`` so every consumer keeps working, just
    more bluntly. Callers surface ``source`` in diagnostics.
    """

    __slots__ = ("quantiles", "source", "n_tracks")

    def __init__(self, quantiles: list[float] | None, *, source: str, n_tracks: int = 0):
        self.quantiles = quantiles
        self.source = source
        self.n_tracks = n_tracks

    def sim_pct(self, cos: float) -> float:
        """Map a cosine onto [0, 1] — the fraction of random pairs it beats."""
        q = self.quantiles
        if not q:
            return max(0.0, min(1.0, (float(cos) + 1.0) / 2.0))
        if cos <= q[0]:
            return 0.0
        if cos >= q[-1]:
            return 1.0
        # q is sorted ascending; np.searchsorted + linear interpolation inside
        # the bracketing bin gives a smooth, monotone mapping.
        i = int(np.searchsorted(q, cos, side="right"))
        lo, hi = q[i - 1], q[i]
        frac = 0.0 if hi <= lo else (cos - lo) / (hi - lo)
        return (i - 1 + frac) / (len(q) - 1)

    def sim_pct_matrix(self, cosines) -> np.ndarray:
        """Vectorised ``sim_pct`` for an array of cosines."""
        arr = np.asarray(cosines, dtype=np.float32)
        q = self.quantiles
        if not q:
            return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        grid = np.asarray(q, dtype=np.float32)
        pcts = np.linspace(0.0, 1.0, len(q), dtype=np.float32)
        return np.interp(arr, grid, pcts).astype(np.float32)


_RAW = Calibration(None, source="raw")


def _is_fresh(table: dict, n_tracks: int | None, now: datetime) -> bool:
    """A stored table is reusable while its version, track count and age hold."""
    if table.get("version") != CALIB_VERSION:
        return False
    if not table.get("quantiles"):
        return False
    built = table.get("built_at")
    if built:
        try:
            age_days = (now - datetime.fromisoformat(built)).total_seconds() / 86400.0
            if age_days > CALIB_TTL_DAYS:
                return False
        except (TypeError, ValueError):
            return False
    stored_n = int(table.get("n_tracks") or 0)
    if n_tracks and stored_n:
        if abs(n_tracks - stored_n) / max(stored_n, 1) > CALIB_DRIFT:
            return False
    return True


def load(
    qdrant_client, collection_name: str, *,
    n_tracks: int | None = None,
    build_if_missing: bool = True,
    now: datetime | None = None,
) -> Calibration:
    """Return this collection's calibration, building it when stale/absent.

    Never raises: a failed build degrades to the raw fallback so the stream
    keeps serving. ``n_tracks`` is the caller's already-known point count
    (``next_chunk`` counts the collection anyway) — pass it to skip a count call.
    """
    now = now or datetime.utcnow()
    stored = MetadataDB.get_clap_calibration(collection_name)
    if stored and _is_fresh(stored, n_tracks, now):
        return Calibration(
            [float(x) for x in stored["quantiles"]],
            source="table", n_tracks=int(stored.get("n_tracks") or 0),
        )
    if not build_if_missing:
        return _RAW
    table = build(qdrant_client, collection_name, now=now)
    if table is None:
        return _RAW
    return Calibration(table["quantiles"], source="table",
                       n_tracks=int(table.get("n_tracks") or 0))


def build(
    qdrant_client, collection_name: str, *,
    sample_size: int = CALIB_SAMPLE,
    now: datetime | None = None,
    rng=None,
) -> dict | None:
    """Sample the collection, compute the quantile table, persist it.

    Returns the stored table dict, or ``None`` when the collection is too small
    or Qdrant is unreachable (callers fall back to raw cosines).
    """
    now = now or datetime.utcnow()
    rng = rng or random
    vectors = _sample_vectors(qdrant_client, collection_name, sample_size, rng)
    if vectors is None or len(vectors) < CALIB_MIN_TRACKS:
        logger.info("[calib] %s: not enough vectors (%s) — staying on raw cosine",
                    collection_name, 0 if vectors is None else len(vectors))
        return None

    quantiles = quantiles_from_vectors(vectors)
    table = {
        "version": CALIB_VERSION,
        "n_tracks": _count(qdrant_client, collection_name) or len(vectors),
        "n_sample": len(vectors),
        "built_at": now.isoformat(),
        "quantiles": quantiles,
    }
    try:
        MetadataDB.set_clap_calibration(collection_name, table)
    except Exception:
        logger.exception("[calib] failed to persist table for %s", collection_name)
    return table


def quantiles_from_vectors(vectors: np.ndarray) -> list[float]:
    """101 quantiles of the pairwise cosine distribution of unit vectors.

    Only the strict upper triangle is used — the diagonal is 1.0 by
    construction and would drag the top of the distribution to «identical».
    """
    sims = vectors @ vectors.T
    iu = np.triu_indices(len(vectors), k=1)
    flat = sims[iu]
    if flat.size == 0:
        return [0.0] * N_QUANTILES
    qs = np.quantile(flat, np.linspace(0.0, 1.0, N_QUANTILES))
    # np.quantile is already sorted, but float noise can produce a tiny
    # inversion; enforce monotonicity so sim_pct stays well-defined.
    qs = np.maximum.accumulate(qs)
    return [round(float(x), 6) for x in qs]


def _sample_vectors(qdrant_client, collection_name: str, sample_size: int, rng) -> np.ndarray | None:
    """Up to ``sample_size`` unit CLAP vectors, reservoir-free: scan a capped
    prefix of the collection, then sample from it.

    A capped prefix biases toward insertion order, which for a music library is
    roughly folder order — acceptable for a *distribution* estimate and far
    cheaper than a full scroll with vectors.
    """
    collected: list[np.ndarray] = []
    try:
        for point in scroll_all(qdrant_client, collection_name,
                                batch_size=256, with_payload=False,
                                with_vectors=["clap"]):
            vec = point.vector.get("clap") if isinstance(point.vector, dict) else point.vector
            if not vec:
                continue
            v = np.asarray(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                collected.append(v / n)
            if len(collected) >= CALIB_SCROLL_CAP:
                break
    except Exception:
        logger.exception("[calib] vector scroll failed for %s", collection_name)
        return None
    if not collected:
        return None
    if len(collected) > sample_size:
        collected = rng.sample(collected, sample_size)
    return np.stack(collected)


def _count(qdrant_client, collection_name: str) -> int | None:
    try:
        return qdrant_client.count(collection_name=collection_name, exact=True).count
    except Exception:
        return None
