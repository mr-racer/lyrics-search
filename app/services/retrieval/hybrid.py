"""Index a list of strings once, then rank it against any number of queries.

Building the index is the expensive half (two model passes over every document),
so a caller that will ask several questions of the same corpus should build one
retriever and reuse it — which is exactly what the assistant's web branch does
across iterations.

Nothing here reads configuration: ``alpha`` and the fusion weights arrive as
arguments. That keeps this package free of the assistant's config module, which
would otherwise be an import cycle (the assistant imports retrieval).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from app.resources.models import STATS, ModelError
from app.services.retrieval.bm25 import BM25
from app.services.retrieval.hub import DEFAULT_HUB

logger = logging.getLogger(__name__)

# The constant from the original RRF paper, and what Qdrant uses for the track
# search — keeping it identical means the two behave alike.
RRF_K = 60

# 1.0 puts the order entirely in the cross-encoder's hands; 0.8 keeps a fifth of
# the first-stage signal, which stops a single confident CE mistake from burying
# a chunk every other signal liked.
DEFAULT_ALPHA = 0.8

# The sparse leg is keyed "milco" after the model that produces it, because that
# is the name the calibrated duplicate thresholds are written in. Renaming it
# would silently disable the two-signal duplicate rule (an unrecognised signal
# is ignored, not trusted) — see ``diversity.is_duplicate``.
DEFAULT_WEIGHTS = {"dense": 1.0, "milco": 1.0, "bm25": 0.3}


@dataclass(slots=True)
class Ranked:
    """One scored candidate. ``index`` points into the retriever's own docs."""

    index: int
    rrf: float
    ce_prob: Optional[float]
    final: float
    ranks: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)


def rrf(rankings: dict, *, k: int = RRF_K,
        weights: Optional[dict] = None) -> dict:
    weights = weights or {}
    fused: dict = defaultdict(float)
    for name, order in rankings.items():
        w = weights.get(name, 1.0)
        for rank, idx in enumerate(order, start=1):
            fused[idx] += w / (k + rank)
    return dict(fused)


class HybridRetriever:
    """Dense + learned-sparse + BM25, fused by RRF and reranked."""

    def __init__(self, docs: list, *, hub=None):
        self.hub = hub or DEFAULT_HUB
        self.docs = list(docs)
        self._dense = self._encode("dense", "index", self.docs)
        self._sparse = self._encode("sparse", "index", self.docs)
        self._bm25 = BM25(self.docs)
        logger.info("[retrieval] indexed %d docs (dense=%s sparse=%s)",
                    len(self.docs), self._dense is not None,
                    self._sparse is not None)

    # ── the one place a missing leg is survivable ─────────────────────────

    def _encode(self, leg: str, where: str, texts: list, *,
                is_query: bool = False):
        """Encode with one leg, or drop that leg for this retriever.

        This class is the ONLY sanctioned place to turn a model failure into a
        missing signal, and it is sanctioned because RRF fuses whatever signals
        exist: two out of three still answers the question, and "ranked worse"
        beats "no answer" on a box whose card is shared with an LLM.

        What was missing was never the behaviour, it was the record. A leg lost
        here is counted against ``leg/where``, so a session that ranked on two
        signals instead of three shows up in ``GET /search/models/loaded``
        instead of only in how the answers read.
        """
        encode = (self.hub.encode_dense if leg == "dense"
                  else self.hub.encode_sparse)
        try:
            return encode(texts, is_query=is_query)
        except ModelError as e:
            STATS.degraded(leg, where)
            logger.warning("[retrieval] %s leg lost at '%s' (%d texts) — "
                           "ranking without it: %s", leg, where, len(texts), e)
            return None

    def __len__(self) -> int:
        return len(self.docs)

    def extend(self, docs: list) -> int:
        """Add documents, encoding ONLY the new ones. Returns how many landed.

        The web branch calls this once per iteration. Rebuilding the whole index
        instead would re-embed everything read so far on every round — correct,
        but three times the GPU work by the third iteration for vectors that
        cannot have changed.
        """
        docs = [d for d in docs if (d or "").strip()]
        if not docs:
            return 0

        new_dense = self._encode("dense", "extend", docs)
        new_sparse = self._encode("sparse", "extend", docs)
        try:
            import torch

            if self._dense is not None and new_dense is not None:
                self._dense = torch.cat([self._dense, new_dense], dim=0)
            elif new_dense is None:
                # A leg that fails mid-run must not leave the index ragged: half
                # the documents having dense vectors would silently reorder
                # everything. Drop the leg instead.
                self._dense = None
            if self._sparse is not None and new_sparse is not None:
                self._sparse = torch.cat([self._sparse, new_sparse],
                                         dim=0).coalesce()
            elif new_sparse is None:
                self._sparse = None
        except Exception:  # noqa: BLE001
            logger.warning("[retrieval] extend failed — rebuilding", exc_info=True)
            self.docs.extend(docs)
            self._dense = self._encode("dense", "rebuild", self.docs)
            self._sparse = self._encode("sparse", "rebuild", self.docs)
            self._bm25 = BM25(self.docs)
            return len(docs)

        self.docs.extend(docs)
        self._bm25 = BM25(self.docs)     # pure python, cheaper to rebuild
        return len(docs)

    @property
    def signals(self) -> list:
        out = ["bm25"]
        if self._dense is not None:
            out.insert(0, "dense")
        if self._sparse is not None:
            out.insert(-1, "milco")
        return out

    # ── the one public entry point ────────────────────────────────────────

    def search(self, query: str, *, ce_query: Optional[str] = None,
               min_prob: Optional[float] = None, limit: Optional[int] = None,
               alpha: float = DEFAULT_ALPHA,
               weights: Optional[dict] = None) -> list:
        """Rank every document against ``query``, best first.

        ``ce_query`` is the text the cross-encoder sees, when it should differ
        from the retrieval query — callers use a fuller, unfiltered statement
        there, because a year or a genre helps recall but only confuses a
        relevance judgement.

        ``min_prob`` drops everything the cross-encoder scored below it. With no
        cross-encoder available the filter cannot be applied and is skipped
        rather than silently dropping everything — degrading to "unfiltered but
        ranked" beats degrading to "empty".
        """
        if not self.docs or not (query or "").strip():
            return []

        weights = weights or DEFAULT_WEIGHTS

        rankings: dict = {}
        scores: dict = {}

        dense_scores = self._dense_scores(query)
        if dense_scores is not None:
            scores["dense"] = dense_scores
        sparse_scores = self._sparse_scores(query)
        if sparse_scores is not None:
            scores["milco"] = sparse_scores
        scores["bm25"] = self._bm25.scores(query)

        for name, values in scores.items():
            rankings[name] = sorted(range(len(values)), key=lambda i: -values[i])

        fused = rrf(rankings, weights=weights)
        order = sorted(range(len(self.docs)), key=lambda i: -fused.get(i, 0.0))
        rank_of = {name: {idx: r for r, idx in enumerate(o, 1)}
                   for name, o in rankings.items()}

        results = [
            Ranked(index=i, rrf=fused.get(i, 0.0), ce_prob=None,
                   final=fused.get(i, 0.0),
                   ranks={n: rank_of[n].get(i) for n in rankings},
                   scores={n: float(scores[n][i]) for n in scores})
            for i in order
        ]

        try:
            probs = self.hub.ce_probabilities(
                (ce_query or query), [self.docs[r.index] for r in results])
        except ModelError as e:
            # Degrading to "unfiltered but ranked" beats degrading to "empty":
            # ``min_prob`` cannot be applied without a probability to compare,
            # and dropping everything would be the wrong reading of "we could
            # not judge". Counted, because a run whose whole pack went ungated
            # is a materially different run.
            STATS.degraded("cross_encoder", "search")
            probs = None
            logger.warning("[retrieval] no cross-encoder — returning RRF order, "
                           "min_prob not applied: %s", e)
        if not probs:
            return results[:limit] if limit else results

        lo = min(r.rrf for r in results)
        hi = max(r.rrf for r in results)
        span = (hi - lo) or 1e-9
        for r, p in zip(results, probs):
            r.ce_prob = p
            r.final = alpha * p + (1 - alpha) * ((r.rrf - lo) / span)

        results.sort(key=lambda r: -r.final)
        if min_prob is not None:
            results = [r for r in results if (r.ce_prob or 0.0) >= min_prob]
        return results[:limit] if limit else results

    # ── document-to-document ──────────────────────────────────────────────

    def similarity_matrix(self, indices: list) -> dict:
        """Cosine between the given documents, per signal.

        No model call and no re-encoding: these are the same document vectors
        the search itself ranked with, so asking whether two passages say the
        same thing costs one small matrix multiply. That is what makes it
        affordable to ask on every query.

        A signal that is unavailable is simply absent from the result. The
        caller decides what to do with one opinion instead of two.
        """
        out: dict = {}
        if len(indices) < 2:
            return out
        dense = self._dense_similarity(indices)
        if dense is not None:
            out["dense"] = dense
        sparse = self._sparse_similarity(indices)
        if sparse is not None:
            out["milco"] = sparse
        return out

    def _dense_similarity(self, indices: list):
        if self._dense is None:
            return None
        try:
            import torch

            idx = torch.as_tensor(indices, dtype=torch.long,
                                  device=self._dense.device)
            sub = self._dense.index_select(0, idx).float()
            # Encoded normalised already; renormalising costs nothing and makes
            # this correct regardless of how the vectors got here.
            sub = sub / sub.norm(dim=1, keepdim=True).clamp_min(1e-9)
            return (sub @ sub.T).clamp(-1.0, 1.0).cpu().tolist()
        except Exception:  # noqa: BLE001
            logger.warning("[retrieval] dense similarity failed", exc_info=True)
            return None

    def _sparse_similarity(self, indices: list):
        if self._sparse is None:
            return None
        try:
            import torch

            idx = torch.as_tensor(indices, dtype=torch.long,
                                  device=self._sparse.device)
            sub = torch.index_select(self._sparse, 0, idx).coalesce().float()
            gram = torch.sparse.mm(sub, sub.t()).to_dense()
            # The diagonal of a Gram matrix IS the squared norm of each row, so
            # normalising needs no second pass over the sparse values.
            norms = gram.diagonal().clamp_min(1e-12).sqrt()
            return (gram / norms[:, None] / norms[None, :]
                    ).clamp(0.0, 1.0).cpu().tolist()
        except Exception:  # noqa: BLE001
            logger.warning("[retrieval] sparse similarity failed", exc_info=True)
            return None

    # ── per-signal scoring ────────────────────────────────────────────────

    def _dense_scores(self, query: str) -> Optional[list]:
        if self._dense is None:
            return None
        q = self._encode("dense", "query", [query], is_query=True)
        if q is None:
            return None
        try:
            # Both sides are already L2-normalised, so the dot product IS the
            # cosine similarity.
            return (self._dense.float() @ q.float().T).squeeze(1).cpu().tolist()
        except Exception:  # noqa: BLE001
            logger.warning("[retrieval] dense scoring failed", exc_info=True)
            return None

    def _sparse_scores(self, query: str) -> Optional[list]:
        if self._sparse is None:
            return None
        q = self._encode("sparse", "query", [query], is_query=True)
        if q is None:
            return None
        try:
            import torch

            return torch.sparse.mm(
                q, self._sparse.t()).to_dense().squeeze(0).cpu().tolist()
        except Exception:  # noqa: BLE001
            logger.warning("[retrieval] sparse scoring failed", exc_info=True)
            return None
