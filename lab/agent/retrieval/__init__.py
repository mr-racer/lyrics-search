"""Hybrid retrieval: three signals, reciprocal rank fusion, cross-encoder.

This subpackage is the piece meant to move into ``app/`` later, so it imports
nothing from the rest of ``lab`` and depends only on torch / transformers /
sentence-transformers / numpy — all of it inside functions.

Recall first, precision second. Dense (Octen), learned sparse (MILCO) and BM25
each contribute a ranking; RRF merges them without needing their scores to be
comparable; the cross-encoder then reads every surviving candidate together
with the query and produces the number the pipeline actually thresholds on.

Nothing here caps the result count. Selection is by ``min_prob`` alone —
callers that want a cap pass ``limit``, and the facts path deliberately does
not.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from lab.agent.retrieval.bm25 import BM25
from lab.agent.retrieval.diversity import (Duplicate, Selection,
                                           duplicate_report, is_duplicate,
                                           pick_diverse)
from lab.agent.retrieval.hub import ModelHub
from lab.agent.retrieval.types import Fact, Ranked

logger = logging.getLogger(__name__)

# The constant from the original RRF paper, and what Qdrant uses for the track
# search — keeping it identical means the two behave alike.
RRF_K = 60


def rrf(rankings: dict[str, list[int]], *, k: int = RRF_K,
        weights: Optional[dict[str, float]] = None) -> dict[int, float]:
    weights = weights or {}
    fused: dict[int, float] = defaultdict(float)
    for name, order in rankings.items():
        w = weights.get(name, 1.0)
        for rank, idx in enumerate(order, start=1):
            fused[idx] += w / (k + rank)
    return dict(fused)


class HybridRetriever:
    """Indexes ``docs`` once, then answers any number of queries against them.

    Building the index is the expensive half (two model passes over every
    document), so a caller that will ask several questions of the same corpus
    should build one retriever and reuse it — which is exactly what the web
    pipeline does across iterations.
    """

    def __init__(self, docs: list[str], *, hub: Optional[ModelHub] = None,
                 config=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or (hub.cfg if hub is not None else AgentConfig())
        self.hub = hub or ModelHub(self.cfg)
        self.docs = list(docs)
        self._dense = self.hub.encode_dense(self.docs, is_query=False)
        self._sparse = self.hub.encode_sparse(self.docs, is_query=False)
        self._bm25 = BM25(self.docs)
        logger.info("[retrieval] indexed %d docs (dense=%s sparse=%s)",
                    len(self.docs), self._dense is not None,
                    self._sparse is not None)

    def __len__(self) -> int:
        return len(self.docs)

    def extend(self, docs: list[str]) -> int:
        """Add documents, encoding ONLY the new ones. Returns how many landed.

        The web pipeline calls this once per iteration. Rebuilding the whole
        index instead would re-embed everything read so far on every round —
        correct, but three times the GPU work by the third iteration for
        vectors that cannot have changed.
        """
        docs = [d for d in docs if (d or "").strip()]
        if not docs:
            return 0

        new_dense = self.hub.encode_dense(docs, is_query=False)
        new_sparse = self.hub.encode_sparse(docs, is_query=False)
        try:
            import torch

            if self._dense is not None and new_dense is not None:
                self._dense = torch.cat([self._dense, new_dense], dim=0)
            elif new_dense is None:
                # A leg that fails mid-run must not leave the index ragged:
                # half the documents having dense vectors would silently
                # reorder everything. Drop the leg instead.
                self._dense = None
            if self._sparse is not None and new_sparse is not None:
                self._sparse = torch.cat([self._sparse, new_sparse],
                                         dim=0).coalesce()
            elif new_sparse is None:
                self._sparse = None
        except Exception:
            logger.warning("[retrieval] extend failed — rebuilding", exc_info=True)
            self.docs.extend(docs)
            self._dense = self.hub.encode_dense(self.docs, is_query=False)
            self._sparse = self.hub.encode_sparse(self.docs, is_query=False)
            self._bm25 = BM25(self.docs)
            return len(docs)

        self.docs.extend(docs)
        self._bm25 = BM25(self.docs)     # pure python, cheaper to rebuild
        return len(docs)

    @property
    def signals(self) -> list[str]:
        out = ["bm25"]
        if self._dense is not None:
            out.insert(0, "dense")
        if self._sparse is not None:
            out.insert(-1, "milco")
        return out

    # ── the one public entry point ────────────────────────────────────────

    def search(self, query: str, *, ce_query: Optional[str] = None,
               min_prob: Optional[float] = None, limit: Optional[int] = None,
               alpha: Optional[float] = None,
               weights: Optional[dict] = None) -> list[Ranked]:
        """Rank every document against ``query``, best first.

        ``ce_query`` is the text the cross-encoder sees, when it should differ
        from the retrieval query — the pipeline uses a fuller, unfiltered
        statement there, because a year or a genre helps recall but only
        confuses a relevance judgement.

        ``min_prob`` drops everything the cross-encoder scored below it. With
        no cross-encoder available the filter cannot be applied and is skipped
        rather than silently dropping everything — degrading to "unfiltered but
        ranked" beats degrading to "empty".
        """
        if not self.docs or not (query or "").strip():
            return []

        alpha = self.cfg.ce_alpha if alpha is None else alpha
        weights = weights or self.cfg.fusion_weights

        rankings: dict[str, list[int]] = {}
        scores: dict[str, list[float]] = {}

        dense_scores = self._dense_scores(query)
        if dense_scores is not None:
            scores["dense"] = dense_scores
        sparse_scores = self._sparse_scores(query)
        if sparse_scores is not None:
            scores["milco"] = sparse_scores
        scores["bm25"] = self._bm25.scores(query)

        for name, values in scores.items():
            rankings[name] = sorted(range(len(values)),
                                    key=lambda i: -values[i])

        fused = rrf(rankings, weights=weights)
        order = sorted(range(len(self.docs)), key=lambda i: -fused.get(i, 0.0))
        rank_of = {name: {idx: r for r, idx in enumerate(o, 1)}
                   for name, o in rankings.items()}

        results = [
            Ranked(index=i, rrf=fused.get(i, 0.0), ce=None, ce_prob=None,
                   final=fused.get(i, 0.0),
                   ranks={n: rank_of[n].get(i) for n in rankings},
                   scores={n: float(scores[n][i]) for n in scores})
            for i in order
        ]

        probs = self.hub.ce_probabilities(
            (ce_query or query), [self.docs[r.index] for r in results])
        if probs is None:
            logger.info("[retrieval] no cross-encoder — returning RRF order, "
                        "min_prob not applied")
            return results[:limit] if limit else results

        lo = min(r.rrf for r in results)
        hi = max(r.rrf for r in results)
        span = (hi - lo) or 1e-9
        for r, p in zip(results, probs):
            r.ce_prob = p
            r.ce = p
            r.final = alpha * p + (1 - alpha) * ((r.rrf - lo) / span)

        results.sort(key=lambda r: -r.final)
        if min_prob is not None:
            results = [r for r in results if (r.ce_prob or 0.0) >= min_prob]
        return results[:limit] if limit else results

    # ── document-to-document ──────────────────────────────────────────────

    def similarity_matrix(self, indices: list[int]) -> dict[str, list[list[float]]]:
        """Cosine between the given documents, per signal.

        No model call and no re-encoding: these are the same document vectors
        the search itself ranked with, so asking whether two passages say the
        same thing costs one small matrix multiply. That is what makes it
        affordable to ask on every query.

        A signal that is unavailable is simply absent from the result. The
        caller decides what to do with one opinion instead of two.
        """
        out: dict[str, list[list[float]]] = {}
        if len(indices) < 2:
            return out
        dense = self._dense_similarity(indices)
        if dense is not None:
            out["dense"] = dense
        sparse = self._sparse_similarity(indices)
        if sparse is not None:
            out["milco"] = sparse
        return out

    def _dense_similarity(self, indices: list[int]):
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
        except Exception:
            logger.warning("[retrieval] dense similarity failed", exc_info=True)
            return None

    def _sparse_similarity(self, indices: list[int]):
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
        except Exception:
            logger.warning("[retrieval] sparse similarity failed", exc_info=True)
            return None

    # ── per-signal scoring ────────────────────────────────────────────────

    def _dense_scores(self, query: str) -> Optional[list[float]]:
        if self._dense is None:
            return None
        q = self.hub.encode_dense([query], is_query=True)
        if q is None:
            return None
        try:
            # Both sides are already L2-normalised, so the dot product IS the
            # cosine similarity.
            return (self._dense.float() @ q.float().T).squeeze(1).cpu().tolist()
        except Exception:
            logger.warning("[retrieval] dense scoring failed", exc_info=True)
            return None

    def _sparse_scores(self, query: str) -> Optional[list[float]]:
        if self._sparse is None:
            return None
        q = self.hub.encode_sparse([query], is_query=True)
        if q is None:
            return None
        try:
            import torch

            return torch.sparse.mm(
                q, self._sparse.t()).to_dense().squeeze(0).cpu().tolist()
        except Exception:
            logger.warning("[retrieval] sparse scoring failed", exc_info=True)
            return None


__all__ = ["HybridRetriever", "ModelHub", "BM25", "Fact", "Ranked", "rrf",
           "RRF_K", "pick_diverse", "duplicate_report", "is_duplicate",
           "Duplicate", "Selection"]
