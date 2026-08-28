"""The model legs over HTTP, for services that are not MusiX.

One instance of the weights per machine. Octen (1.2 GB on disk), MILCO (2.3 GB)
and bge-reranker-v2-m3 (2.2 GB) are ~3.5 GB of VRAM on a card already shared
with a separately launched LLM; a second RAG service loading its own copy is not
affordable, and that is the whole reason this router exists.

Shapes are borrowed, not invented: OpenAI for embeddings, Cohere for reranking.
Both are what every RAG stack already speaks, so a consumer needs no MusiX
specific client. The sparse leg has no standard to borrow — the response follows
what the model produces, which happens to be Qdrant's sparse-vector format.

**The asymmetric model is addressed by NAME.** Octen ships two prompts, an
instruction for the query and a single space for the document, and mixing them
costs real recall. ``/v1/embeddings`` has no field for that, but every OpenAI
client can set ``model`` — so ``octen-query`` and ``octen-document`` are two
names for the SAME resident weights, and the asymmetry travels through a
standard field instead of an extension nobody implements.

This module deliberately holds its own request/response models rather than
putting them in ``app/domain/models.py``: those are the contracts shared BETWEEN
LAYERS, and these are a wire format shared with the outside. It also carries its
own auth and its own error mapping, so that moving it into a separate process
later (see the design spec) is a change of ``include_router``, nothing more.

**No serialisation yet.** Two concurrent callers can put two forwards on the
card at once, which on a shared GPU is how the transient peaks add up. That is a
known gap, taken deliberately: the failures are at least visible now, and the
batcher that closes it is the next step.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.resources.model_registry import (MAX_SEQ_LENGTH, RERANK_MAX_LEN,
                                          SPARSE_MAX_LEN, ModelRegistry)
from app.resources.models import (ModelEncodeFailed, ModelError, ModelOOM,
                                  ModelOverloaded, ModelUnavailable)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])

# ── the two names for one set of dense weights ───────────────────────────────
# The value is what ``is_query`` becomes, which is what picks the prompt.
DENSE_ALIASES = {"octen-query": True, "octen-document": False}
SPARSE_MODEL_ID = "milco-sparse"
RERANK_MODEL_ID = "bge-reranker-v2-m3"

# MILCO's vocabulary: 30522 (the SPLADE-v3 pivot view) + 250002 (the
# bge-m3/XLM-R source view, offset by the first). Reported with every sparse
# response because a consumer writing these into Qdrant needs it and cannot
# derive it from the non-zeros it happens to receive.
SPARSE_DIM = 280524

# ── admission limits ─────────────────────────────────────────────────────────
# Not throttling — a bound on what one request may cost. Without them a caller
# indexing a corpus sends its whole corpus, and the failure lands on the card
# rather than in the client's error handler.
MAX_INPUTS = 256
MAX_RERANK_DOCUMENTS = 256
MAX_TOTAL_CHARS = 2_000_000

TOKEN_ENV = "MUSIX_MODELS_TOKEN"


# ── auth ─────────────────────────────────────────────────────────────────────

def _require_token(request: Request) -> None:
    """A static bearer token, not the account JWT.

    The caller here is a service, not a person: there is no account to isolate
    by, nothing account-scoped is reachable through this router, and handing out
    user tokens to a background indexer would be worse in every direction.

    With no token configured the router refuses everything. An open model
    endpoint is not a sensible default even on a loopback-only deployment, and
    refusing loudly beats a door that silently turns out to have been open.
    """
    expected = (os.environ.get(TOKEN_ENV) or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"model endpoints are disabled: set {TOKEN_ENV} to enable them",
        )
    header = request.headers.get("authorization") or ""
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(presented.strip(),
                                                             expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ── error mapping ────────────────────────────────────────────────────────────

_STATUS = {
    ModelUnavailable: 503,
    ModelOverloaded: 429,
    ModelOOM: 500,
    ModelEncodeFailed: 500,
}
_TYPE = {
    ModelUnavailable: "model_unavailable",
    ModelOverloaded: "model_overloaded",
    ModelOOM: "model_out_of_memory",
    ModelEncodeFailed: "model_encode_failed",
}


def install_model_error_handler(app) -> None:
    """Map :class:`ModelError` onto HTTP for the app that mounts this router.

    An app-level handler rather than a try/except in every route: the shape of
    the error body is part of the contract, and repeating it six times is how
    one of them ends up different. ``ModelUnavailable`` and ``ModelOverloaded``
    are states a client should retry, and they say so in a header.
    """

    @app.exception_handler(ModelError)
    async def _handle(request: Request, exc: ModelError):  # noqa: ANN202
        status = _STATUS.get(type(exc), 500)
        headers = {}
        if status in (429, 503):
            headers["Retry-After"] = "5" if status == 429 else "30"
        logger.warning("[models] %s -> %d", exc, status)
        return JSONResponse(
            status_code=status,
            headers=headers,
            content={"error": {"message": str(exc),
                               "type": _TYPE.get(type(exc), "model_error"),
                               "leg": exc.leg, "op": exc.op}},
        )


# ── request/response models ──────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "octen-document"
    encoding_format: str = "float"
    # Present because the OpenAI schema has it. Rejected unless it matches the
    # model's real width — see ``_check_dimensions``.
    dimensions: Optional[int] = None


class SparseRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = SPARSE_MODEL_ID
    is_query: bool = False


class RerankRequest(BaseModel):
    query: str
    # Strings, or objects with a ``text`` field. Both are in circulation among
    # the stacks that speak this shape, and guessing wrong costs a 422 the
    # caller cannot act on.
    documents: list[Union[str, dict]]
    model: str = RERANK_MODEL_ID
    top_n: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _as_list(value: Union[str, list]) -> list:
    return [value] if isinstance(value, str) else list(value)


def _check_inputs(texts: list, *, limit: int = MAX_INPUTS) -> None:
    if not texts:
        raise HTTPException(status_code=422, detail="input is empty")
    if len(texts) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"{len(texts)} inputs exceeds the per-request limit of {limit}; "
                   "split the batch")
    total = sum(len(t or "") for t in texts)
    if total > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"{total} characters exceeds the per-request limit of "
                   f"{MAX_TOTAL_CHARS}; split the batch")
    if any(not isinstance(t, str) for t in texts):
        raise HTTPException(status_code=422, detail="every input must be a string")


def _check_dimensions(requested: Optional[int]) -> None:
    """Reject any width but the model's own.

    The OpenAI schema carries ``dimensions`` for Matryoshka models. Octen's card
    documents a fixed 1024 and says nothing about MRL, so truncating its output
    would degrade it — invisibly, which is the one failure mode this whole
    change exists to remove. Refuse in the open rather than serve a quietly
    worse vector.
    """
    if requested is not None and requested != ModelRegistry.VECTOR_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"this model has no Matryoshka support: dimensions must be "
                   f"{ModelRegistry.VECTOR_DIM}, got {requested}")


def _token_stats(texts: list, limit: int) -> tuple:
    """``(total_tokens, n_truncated)``, or ``(None, None)`` if not countable.

    Truncation is not an error — the request succeeds — but it is the same class
    of silent degradation as a dead leg, and a caller that sent 3000-token
    documents to a 2048-token model has to be able to find that out. Anything
    short enough that it CANNOT reach the ceiling skips the tokenizer, so the
    ordinary case pays nothing: one token is at least one character, so a text
    under ``limit`` characters is under ``limit`` tokens.
    """
    try:
        model, _, _ = ModelRegistry.get_text_model()
        tokenizer = model.tokenizer
        total, truncated = 0, 0
        for text in texts:
            if len(text) < limit:
                total += len(text) // 4 or 1        # cheap estimate, no encode
                continue
            n = len(tokenizer.encode(text, add_special_tokens=True))
            total += n
            if n > limit:
                truncated += 1
        return total, truncated
    except Exception:  # noqa: BLE001 — a missing tokenizer must not fail a request
        return None, None


def _b64(vector) -> str:
    """A float32 embedding as base64, the OpenAI ``encoding_format``.

    Worth offering: 1024 floats rendered as JSON text is roughly eight times the
    bytes of the same numbers packed, and a corpus indexer sends hundreds per
    request.
    """
    import numpy as np
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode()


def _sparse_rows(tensor, n: int) -> list:
    """One ``{indices, values}`` per input row, in the CALLER's order.

    ``coalesce`` sorts the COO indices by (row, column), so the row boundaries
    are a single ``searchsorted`` rather than a Python loop over every non-zero
    — which on a hundred documents is tens of thousands of iterations.
    """
    import numpy as np

    coalesced = tensor.coalesce()
    indices = coalesced.indices().cpu().numpy()
    values = coalesced.values().cpu().numpy().astype(np.float32)
    rows, cols = indices[0], indices[1]
    bounds = np.searchsorted(rows, np.arange(n + 1))
    return [{"index": i,
             "indices": cols[bounds[i]:bounds[i + 1]].astype(int).tolist(),
             "values": values[bounds[i]:bounds[i + 1]].tolist()}
            for i in range(n)]


def _document_text(doc: Union[str, dict]) -> str:
    if isinstance(doc, str):
        return doc
    for key in ("text", "content", "document"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    raise HTTPException(
        status_code=422,
        detail="a document object must carry a 'text', 'content' or "
               "'document' string")


# ── routes ───────────────────────────────────────────────────────────────────

@router.get("/v1/models", dependencies=[Depends(_require_token)])
async def list_models() -> dict:
    """What this instance serves.

    The two dense entries are the SAME weights under two names — the query and
    document sides of one asymmetric model — which is why they report the same
    dimension. A caller indexing a corpus wants ``octen-document``; the same
    caller's queries want ``octen-query``, and using one for both is a quiet
    recall loss rather than an error.
    """
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "musix",
         "backing_model": ModelRegistry.TEXT_MODEL_NAME,
         "kind": "dense", "dimensions": ModelRegistry.VECTOR_DIM,
         "side": "query" if is_query else "document",
         "max_tokens": MAX_SEQ_LENGTH}
        for name, is_query in DENSE_ALIASES.items()
    ] + [
        {"id": SPARSE_MODEL_ID, "object": "model", "owned_by": "musix",
         "backing_model": ModelRegistry.SPARSE_MODEL_NAME,
         "kind": "sparse", "dimensions": SPARSE_DIM,
         "max_tokens": SPARSE_MAX_LEN},
        {"id": RERANK_MODEL_ID, "object": "model", "owned_by": "musix",
         "backing_model": ModelRegistry.RERANK_MODEL_NAME,
         "kind": "rerank", "max_tokens": RERANK_MAX_LEN},
    ]}


@router.get("/health")
async def health() -> dict:
    """Which legs are resident, and what they have been failing at.

    Unauthenticated on purpose, and it says nothing a probe should not see: leg
    names, booleans and counters. It is what a ``depends_on`` healthcheck and a
    consumer's own circuit breaker read, and putting a token in front of that
    only means the token ends up in a compose file.
    """
    status = ModelRegistry.retrieval_status()
    return {"status": "ok" if status["dense"] else "loading",
            "auth_configured": bool((os.environ.get(TOKEN_ENV) or "").strip()),
            "retrieval": status}


@router.post("/v1/embeddings", dependencies=[Depends(_require_token)])
async def embeddings(req: EmbeddingRequest) -> dict:
    """Dense vectors, OpenAI shape.

    ``model`` picks the side of the asymmetric pair. Everything else about the
    request is standard, and the two non-standard fields in the RESPONSE
    (``truncated``, and ``side`` inside ``musix``) are additive — a strict
    OpenAI client ignores them.
    """
    if req.model not in DENSE_ALIASES:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model '{req.model}'; this instance serves "
                   f"{sorted(DENSE_ALIASES)} for dense embeddings")
    if req.encoding_format not in ("float", "base64"):
        raise HTTPException(status_code=422,
                            detail="encoding_format must be 'float' or 'base64'")
    _check_dimensions(req.dimensions)

    texts = _as_list(req.input)
    _check_inputs(texts)
    is_query = DENSE_ALIASES[req.model]

    total_tokens, truncated = await asyncio.to_thread(
        _token_stats, texts, MAX_SEQ_LENGTH)
    # encode_documents, never encode_text: the batch ceiling and the OOM shrink
    # live in there, and a caller that sends 256 texts must not turn into one
    # 256-wide forward on a card that is already holding an LLM.
    vectors = await asyncio.to_thread(
        ModelRegistry.encode_documents, texts, is_query=is_query)

    if truncated:
        logger.info("[models] %d of %d inputs exceeded %d tokens and were "
                    "truncated", truncated, len(texts), MAX_SEQ_LENGTH)
    return {
        "object": "list",
        "model": req.model,
        "data": [{"object": "embedding", "index": i,
                  "embedding": (_b64(v) if req.encoding_format == "base64"
                                else [float(x) for x in v])}
                 for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": total_tokens or 0,
                  "total_tokens": total_tokens or 0},
        "musix": {"side": "query" if is_query else "document",
                  "truncated": truncated,
                  "max_tokens": MAX_SEQ_LENGTH},
    }


@router.post("/v1/embeddings/sparse", dependencies=[Depends(_require_token)])
async def sparse_embeddings(req: SparseRequest) -> dict:
    """Learned-sparse vectors from MILCO.

    No standard to borrow, so the response is what the model produces:
    ``indices`` into a ``dim``-wide vocabulary and their ``values``. That is
    also Qdrant's sparse-vector format, so a consumer can write these straight
    into a collection.

    The term-string view MILCO can produce is deliberately not offered here: it
    rebuilds a 280k-entry vocabulary dictionary and walks every non-zero in
    Python on every call, which is a cost nobody asked for by default.
    """
    if req.model != SPARSE_MODEL_ID:
        raise HTTPException(status_code=404,
                            detail=f"unknown model '{req.model}'; the sparse leg "
                                   f"is '{SPARSE_MODEL_ID}'")
    texts = _as_list(req.input)
    _check_inputs(texts)

    tensor = await asyncio.to_thread(
        ModelRegistry.encode_sparse, texts, is_query=req.is_query)
    return {
        "object": "list",
        "model": req.model,
        "dim": SPARSE_DIM,
        "data": _sparse_rows(tensor, len(texts)),
        "musix": {"side": "query" if req.is_query else "document",
                  "max_tokens": SPARSE_MAX_LEN},
    }


@router.post("/v1/rerank", dependencies=[Depends(_require_token)])
async def rerank(req: RerankRequest) -> dict:
    """Cross-encoder scores, Cohere shape.

    ``relevance_score`` is ``sigmoid(logit)``, in (0, 1) — a probability rather
    than a raw logit because logits are not comparable between model families
    and every threshold anyone writes against this will be one number.

    Results come back sorted best-first with their ORIGINAL index, which is what
    the shape promises and what makes ``top_n`` meaningful.
    """
    if req.model != RERANK_MODEL_ID:
        raise HTTPException(status_code=404,
                            detail=f"unknown model '{req.model}'; the reranker "
                                   f"is '{RERANK_MODEL_ID}'")
    docs = [_document_text(d) for d in req.documents]
    _check_inputs(docs, limit=MAX_RERANK_DOCUMENTS)

    probs = await asyncio.to_thread(
        ModelRegistry.ce_probabilities, req.query, docs)
    ranked = sorted(range(len(docs)), key=lambda i: -probs[i])
    if req.top_n:
        ranked = ranked[:req.top_n]
    results: list[dict[str, Any]] = [
        {"index": i, "relevance_score": float(probs[i])} for i in ranked]
    if req.return_documents:
        for row in results:
            row["document"] = {"text": docs[row["index"]]}
    return {"object": "list", "model": req.model, "results": results,
            "musix": {"max_tokens": RERANK_MAX_LEN}}
