"""The three retrieval models, addressed as one object.

A seam, not a cache. Every model lives in :class:`ModelRegistry` and is loaded
exactly once per process; this class only gives the retriever one thing to hold
and gives a test one thing to replace. Swapping in a fake hub is how the ranking
logic is exercised without a GPU.

Every method degrades instead of raising. A leg that is unavailable returns
``None``, the retriever drops that signal, and the run continues on the rest —
because "ranked worse" is a far better outcome than "no answer", and because
this is the only sane behaviour on a box where the card is busy.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelHub:
    """Dense, learned-sparse and cross-encoder, over :class:`ModelRegistry`."""

    def encode_dense(self, texts: list, *, is_query: bool = False):
        """L2-normalised dense vectors as a torch tensor, or None.

        The model is asymmetric and ships both prompts (an instruction for the
        query, a single space for the document); ``ModelRegistry.encode_text``
        picks the right side by name. Hand-rolling a prefix here, or leaving the
        document side bare, costs real recall — see its docstring.
        """
        if not texts:
            return None
        from app.resources.model_registry import ENCODE_BATCH, ModelRegistry

        try:
            return ModelRegistry.encode_text(
                texts, is_query=is_query, batch_size=ENCODE_BATCH,
                normalize_embeddings=True, convert_to_tensor=True)
        except Exception:  # noqa: BLE001
            logger.warning("[hub] dense encode failed", exc_info=True)
            return None

    def encode_sparse(self, texts: list, *, is_query: bool = False):
        """Learned-sparse representations as a coalesced sparse tensor, or None."""
        if not texts:
            return None
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.encode_sparse(texts, is_query=is_query)

    def ce_probabilities(self, query: str, docs: list) -> Optional[list]:
        """``sigmoid(logit)`` per (query, doc) pair, or None when unavailable."""
        if not docs:
            return None
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.ce_probabilities(query, docs)

    def status(self) -> dict:
        from app.resources.model_registry import ModelRegistry

        return ModelRegistry.retrieval_status()


# One per process is enough — the registry behind it is a singleton anyway.
DEFAULT_HUB = ModelHub()
