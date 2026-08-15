"""Hybrid retrieval over a small in-memory corpus.

Three signals, reciprocal rank fusion, cross-encoder. Dense (the shared text
model), learned sparse and BM25 each contribute a ranking; RRF merges them
without needing their scores to be comparable; the cross-encoder then reads
every surviving candidate together with the query and produces the number the
callers actually threshold on.

Recall first, precision second. Nothing here caps the result count on its own —
selection is by ``min_prob``, and a caller that wants a cap passes ``limit``.

This package knows nothing about music, pages or the assistant: it indexes
strings and ranks them. That is deliberate, and it is what lets the near-
duplicate policy be unit-tested with plain lists and no GPU.
"""

from app.services.retrieval.bm25 import BM25
from app.services.retrieval.diversity import (Duplicate, Selection,
                                              is_duplicate, pair_similarity,
                                              pick_diverse)
from app.services.retrieval.hub import DEFAULT_HUB, ModelHub
from app.services.retrieval.hybrid import RRF_K, HybridRetriever, Ranked, rrf

__all__ = ["BM25", "HybridRetriever", "Ranked", "ModelHub", "DEFAULT_HUB",
           "rrf", "RRF_K", "pick_diverse", "is_duplicate", "pair_similarity",
           "Duplicate", "Selection"]
