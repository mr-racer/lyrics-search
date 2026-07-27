"""Unified AI assistant — one endpoint over lyrics search, playlist building
and artist/song facts.

Thin HTTP layer: auth, the NDJSON envelope, and the ``app.state`` lookups.
Everything else is ``app/services/assistant/``.

NDJSON rather than SSE because ``EventSource`` cannot POST — the same reason
``/recommend/ai-playlist/stream`` uses it. Frame shapes::

    {"type":"route","intent":...,"confidence":...,"human":...}
    {"type":"status","stage":...,"human":..., …}
    {"type":"clarify","options":[{"intent","label"}, …]}
    {"type":"disambiguate","subject_options":[…]}
    {"type":"result","payload":<AssistantResponse>}
    {"type":"ping"}
    {"type":"error","message":...}
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.domain.models import AssistantRequest, AssistantResponse, User
from app.services.assistant import service as assistant_service
from app.services.assistant.humanize import human

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["Assistant"])


def _context(request: Request, current_user: User):
    """Resolve (search_service, qdrant, collection) or 503 like the other AI routes."""
    db_client = getattr(request.app.state, "db_client", None)
    search_service = getattr(request.app.state, "search_service", None)
    if db_client is None or db_client.qdrant is None or search_service is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return search_service, db_client.qdrant, derive_collection_for_user(current_user)


@router.post("/", response_model=AssistantResponse)
async def assistant(
    req: AssistantRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> AssistantResponse:
    """Non-streaming twin of ``/assistant/stream`` — the client's fallback when
    the stream can't open. Same routing, same payload, no progress frames."""
    search_service, qdrant, collection = _context(request, current_user)
    try:
        result = await assistant_service.run_assistant(
            req, search_service=search_service, qdrant=qdrant,
            collection_name=collection, current_user=current_user, sink=None,
        )
    except Exception as exc:
        logger.exception("[assistant] failed")
        raise HTTPException(status_code=502, detail=f"Assistant failed: {exc}")
    return AssistantResponse(**result)


@router.post("/stream")
async def assistant_stream(
    req: AssistantRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant: NDJSON progress frames while the pipeline runs, then a
    terminal ``result`` (or ``error``) line carrying the same payload the plain
    endpoint returns."""
    search_service, qdrant, collection = _context(request, current_user)

    async def gen():
        sink = assistant_service.EventSink(req.lang)
        DONE = object()

        async def run() -> None:
            try:
                result = await assistant_service.run_assistant(
                    req, search_service=search_service, qdrant=qdrant,
                    collection_name=collection, current_user=current_user, sink=sink,
                )
                sink.put({"type": "result",
                          "payload": AssistantResponse(**result).model_dump(mode="json")})
            except Exception as exc:  # surface a terminal frame, never hang
                logger.exception("[assistant/stream] failed")
                sink.put({"type": "error", "human": human("error", req.lang),
                          "message": str(exc)[:300]})
            finally:
                sink.put(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(sink.queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keepalive: an LLM call can be silent for a minute and the
                    # reverse proxy will drop a connection that quiet.
                    yield json.dumps({"type": "ping"}) + "\n"
                    continue
                if item is DONE:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            if not task.done():
                task.cancel()  # stop the pipeline when the client disconnects

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
