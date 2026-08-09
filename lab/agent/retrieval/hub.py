"""The three models, loaded once and shared.

Dense (Octen), sparse (MILCO) and the cross-encoder (bge-reranker-v2-m3). One
hub per process; the notebook builds it once and hands it to everything.

Everything heavy is imported INSIDE the methods. That is not style — it is what
lets the rest of the package be imported, and its pure logic unit-tested, on a
machine with no torch. The app enforces the same rule for the same reason.

Every method degrades instead of raising. ``available`` on each leg tells the
retriever which signals it actually has; a hub with nothing loaded still works,
it just ranks by BM25 alone.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — no torch at all is a valid lab state
        return "cpu"


class ModelHub:
    """Lazy, thread-safe singletons for the retrieval stack.

    Loading is lazy per leg, so a run that never reaches the reranker never
    pays for it, and a machine that cannot fit MILCO still gets dense + BM25.
    """

    def __init__(self, config=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.device = resolve_device(self.cfg.device)
        self._dense: Any = None
        self._sparse: Any = None
        self._ce: Optional[tuple[Any, Any]] = None
        self._lock = threading.Lock()
        # Legs that tried to load and failed are not retried: a missing model
        # is a missing model, and re-attempting it on every query turns one
        # slow run into every run being slow.
        self._failed: set[str] = set()

    # ── loading ───────────────────────────────────────────────────────────

    def dense(self):
        if self._dense is not None or "dense" in self._failed:
            return self._dense
        with self._lock:
            if self._dense is None and "dense" not in self._failed:
                try:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    kwargs: dict = {"tokenizer_kwargs": {"padding_side": "left"}}
                    if self.device == "cuda":
                        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
                    self._dense = SentenceTransformer(
                        self.cfg.dense_model, device=self.device, **kwargs)
                    logger.info("[hub] dense %s on %s (dim=%s)",
                                self.cfg.dense_model, self.device,
                                self._dense.get_sentence_embedding_dimension())
                except Exception:
                    logger.warning("[hub] dense model unavailable", exc_info=True)
                    self._failed.add("dense")
        return self._dense

    def sparse(self):
        if self._sparse is not None or "sparse" in self._failed:
            return self._sparse
        with self._lock:
            if self._sparse is None and "sparse" not in self._failed:
                try:
                    from transformers import AutoModel

                    model = AutoModel.from_pretrained(
                        self.cfg.sparse_model, trust_remote_code=True)
                    self._sparse = model.to(self.device).eval()
                    logger.info("[hub] sparse %s on %s",
                                self.cfg.sparse_model, self.device)
                except Exception:
                    logger.warning("[hub] sparse model unavailable", exc_info=True)
                    self._failed.add("sparse")
        return self._sparse

    def cross_encoder(self):
        if self._ce is not None or "ce" in self._failed:
            return self._ce
        with self._lock:
            if self._ce is None and "ce" not in self._failed:
                try:
                    import torch
                    from transformers import (AutoModelForSequenceClassification,
                                              AutoTokenizer)

                    tok = AutoTokenizer.from_pretrained(
                        self.cfg.rerank_model, trust_remote_code=True)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.cfg.rerank_model, trust_remote_code=True,
                        torch_dtype=(torch.float16 if self.device == "cuda"
                                     else torch.float32),
                    ).to(self.device).eval()
                    self._ce = (tok, model)
                    logger.info("[hub] cross-encoder %s on %s",
                                self.cfg.rerank_model, self.device)
                except Exception:
                    logger.warning("[hub] cross-encoder unavailable", exc_info=True)
                    self._failed.add("ce")
        return self._ce

    def release_reranker(self) -> None:
        """Drop the cross-encoder from memory. Only used when the config says
        not to keep it resident — the other two legs are needed every query."""
        with self._lock:
            self._ce = None
        try:
            import gc
            import torch
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ── encoding ──────────────────────────────────────────────────────────

    def encode_dense(self, texts: list[str], *, is_query: bool = False):
        """L2-normalised dense vectors, or None if the model is unavailable.

        The model is asymmetric and ships both prompts (an instruction for the
        query, a single space for the document), so each side is addressed by
        name. A model without them falls back to no prompt at all rather than
        inventing one.
        """
        model = self.dense()
        if model is None or not texts:
            return None
        name = "query" if is_query else "document"
        kwargs: dict = {}
        if name in (getattr(model, "prompts", None) or {}):
            kwargs["prompt_name"] = name
        try:
            return model.encode(
                texts, batch_size=self.cfg.encode_batch,
                normalize_embeddings=True, convert_to_tensor=True, **kwargs)
        except Exception:
            logger.warning("[hub] dense encode failed", exc_info=True)
            return None

    def encode_sparse(self, texts: list[str], *, is_query: bool = False):
        """Sparse (MILCO) representations as a coalesced sparse tensor, or None."""
        model = self.sparse()
        if model is None or not texts:
            return None
        try:
            import torch

            encode = model.encode_query if is_query else model.encode_document
            reps = []
            with torch.no_grad():
                for i in range(0, len(texts), self.cfg.encode_batch):
                    batch = texts[i:i + self.cfg.encode_batch]
                    reps.append(encode(
                        batch, source_view=self.cfg.milco_source_view).coalesce())
            return torch.cat(reps, dim=0).coalesce()
        except Exception:
            logger.warning("[hub] sparse encode failed", exc_info=True)
            return None

    def ce_probabilities(self, query: str, docs: list[str]) -> Optional[list[float]]:
        """``sigmoid(logit)`` per (query, doc) pair, or None if unavailable.

        A probability rather than a raw logit because everything downstream
        compares it against one threshold shared across the pipeline; logits
        are not comparable between model families, probabilities roughly are.
        """
        pair = self.cross_encoder()
        if pair is None or not docs:
            return None
        tok, model = pair
        try:
            import torch

            out: list[float] = []
            with torch.no_grad():
                for i in range(0, len(docs), self.cfg.rerank_batch):
                    batch = docs[i:i + self.cfg.rerank_batch]
                    enc = tok([query] * len(batch), batch, padding=True,
                              truncation=True, max_length=self.cfg.rerank_max_len,
                              return_tensors="pt").to(self.device)
                    logits = model(**enc).logits.float()
                    # 1-logit head (bge/jina/gte) vs the older 2-class one.
                    col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                    out.extend(torch.sigmoid(col).cpu().tolist())
            return out
        except Exception:
            logger.warning("[hub] cross-encoder scoring failed", exc_info=True)
            return None
        finally:
            if not self.cfg.keep_reranker_resident:
                self.release_reranker()

    # ── introspection ─────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "device": self.device,
            "dense": self._dense is not None,
            "sparse": self._sparse is not None,
            "cross_encoder": self._ce is not None,
            "failed": sorted(self._failed),
        }
