"""The three retrieval models, addressed as one object.

A seam, not a cache. Every model lives in :class:`ModelRegistry` and is loaded
exactly once per process; this class only gives the retriever one thing to hold
and gives a test one thing to replace. Swapping in a fake hub is how the ranking
logic is exercised without a GPU.

**Nothing here swallows a failure any more.** ``encode_dense`` used to catch
every exception and return ``None``, which is indistinguishable from "there was
nothing to encode" — so a leg could be dead for a whole session while the only
symptom was worse answers. The legs raise
(:mod:`app.resources.models.errors`) and this hub passes that straight through.

Degrading is still the right behaviour for the retriever — "ranked worse" beats
"no answer", especially on a box where the card is busy — but it is now a
decision :class:`~app.services.retrieval.hybrid.HybridRetriever` makes in
writing, at a named site, and counts. See ``STATS.degraded``.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelHub:
    """Dense, learned-sparse and cross-encoder, over :class:`ModelRegistry`."""

    def encode_dense(self, texts: list, *, is_query: bool = False):
        """L2-normalised dense vectors as a torch tensor.

        ``None`` only for an empty ``texts``; every failure raises.

        The model is asymmetric and ships both prompts (an instruction for the
        query, a single space for the document); ``ModelRegistry.encode_text``
        picks the right side by name. Hand-rolling a prefix here, or leaving the
        document side bare, costs real recall — see its docstring.
        """
        if not texts:
            return None
        from app.resources.model_registry import ENCODE_BATCH, ModelRegistry

        return ModelRegistry.encode_text(
            texts, is_query=is_query, batch_size=ENCODE_BATCH,
            normalize_embeddings=True, convert_to_tensor=True)

    def encode_sparse(self, texts: list, *, is_query: bool = False):
        """Learned-sparse representations as a coalesced sparse tensor.

        ``None`` only for an empty ``texts``; every failure raises.
        """
        if not texts:
            return None
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.encode_sparse(texts, is_query=is_query)

    def ce_probabilities(self, query: str, docs: list) -> Optional[list]:
        """``sigmoid(logit)`` per (query, doc) pair.

        An empty ``docs`` gives an empty list; every failure raises.
        """
        if not docs:
            return []
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.ce_probabilities(query, docs)

    def status(self) -> dict:
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.retrieval_status()


# One per process is enough — the registry behind it is a singleton anyway.
DEFAULT_HUB = ModelHub()
