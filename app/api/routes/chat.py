"""Chat endpoints — thin HTTP layer over the agentic search engine.

The engine itself lives in ``app/services/chat_search_service.py`` (moved there
so the unified assistant can call it without a FastAPI ``Request``); this module
only adapts it to HTTP: auth, the SSE envelope, and the separate track-chat
routes. The public contract of both endpoints is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.domain.models import (
    ChatRequest, TrackChatRequest, TrackChatResponse, User,
)
from app.api.dependencies import get_current_user
from app.api.helpers import derive_collection_for_user
from app.api.sse_utils import sse_data
from app.services.assistant.humanize import human as _human
from app.services.chat_search_service import (  # re-exported: imported by tests
    CLASSIFICATION_SYSTEM_PROMPT,
    DEVELOPER_PROMPT,
    MAX_CTX_HITS,
    NUM_ATTEMPTS,
    SEARCH_LIMIT,
    _extract_lyrics_for_song,
    _history_preamble,
    _match_best_hit,
    _merge_hits,
    _pick_matched_line,
    _quoted_fragments,
    _rrf_merge,
    _run_searches,
    run_chat_search,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


async def _run_chat_core(req, request: Request, current_user: User, emit=None) -> dict:
    """Adapter kept for the two routes below: pulls the search service off
    ``app.state`` and delegates to the engine."""
    return await run_chat_search(
        req, getattr(request.app.state, "search_service", None), current_user, emit=emit,
    )


@router.post("/")
async def chat(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Non-streaming agentic search — runs the loop to completion and returns
    the final payload. Kept for backward compatibility and as the client's
    fallback when the SSE stream is unavailable. Contract unchanged."""
    return await _run_chat_core(req, request, current_user, emit=None)


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant of ``POST /chat/``.

    Runs the same agentic loop, emitting typed step events (classify → plan →
    search → validate → retry) over SSE as the agent works, then a final
    ``answer`` event carrying the exact payload the non-streaming endpoint
    returns. The frontend animates each step's ``human`` label in real time.
    """

    async def event_source() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        DONE = {"__done__": True}

        async def emit(event: dict) -> None:
            queue.put_nowait(event)

        async def run() -> None:
            try:
                result = await _run_chat_core(req, request, current_user, emit=emit)
                queue.put_nowait({"type": "answer", "human": _human("answer", req.lang), **result})
            except Exception as exc:  # surface a terminal error frame, never hang
                logger.error("[chat/stream] core error: %s", exc, exc_info=True)
                queue.put_nowait({
                    "type": "error",
                    "human": ("Ошибка" if (req.lang or "en").startswith("ru") else "Error"),
                    "message": str(exc),
                })
            finally:
                queue.put_nowait(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep the connection warm during long LLM waits
                    continue
                if item is DONE:
                    break
                yield sse_data(item)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Track Chat ───────────────────────────────────────────────────────────────
# Single-track conversational chat with optional web-search fallback.
# Powers the AIChatDrawer + InlineLyricExplain UIs in PlayerSection.
# Separate from the agentic search loop above — no Planner/Scorer/Validator.


@router.post("/track-chat", response_model=TrackChatResponse)
async def track_chat(
    req: TrackChatRequest,
    current_user: User = Depends(get_current_user),
) -> TrackChatResponse:
    """Single-track conversational chat (drawer or per-line explain)."""
    derived = derive_collection_for_user(current_user)
    req = req.model_copy(update={"collection_name": derived})

    if req.mode == "lyric_explain" and not req.selected_line:
        raise HTTPException(
            status_code=400,
            detail="selected_line is required for mode='lyric_explain'",
        )
    try:
        from app.services.track_chat_service import answer_track_chat
        return await answer_track_chat(req)
    except ValueError as exc:
        # answer_track_chat re-raises with same message — defensive
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("[track_chat] error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI error: {str(exc)[:200]}")


@router.post("/track-chat/stream")
async def track_chat_stream(
    req: TrackChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Streaming variant of ``POST /chat/track-chat``.

    Same orchestration, but emits humanized status frames over SSE while the
    agent works — ``{"type":"status","stage":"thinking"|"web_search"|"reading"}``
    (web_search frames carry the ``query``) — then a terminal ``answer`` frame
    with the exact TrackChatResponse payload. The drawer's activity ticker
    animates these in real time; the client falls back to the non-streaming
    endpoint if the stream can't open.
    """
    derived = derive_collection_for_user(current_user)
    req = req.model_copy(update={"collection_name": derived})

    if req.mode == "lyric_explain" and not req.selected_line:
        raise HTTPException(
            status_code=400,
            detail="selected_line is required for mode='lyric_explain'",
        )

    async def event_source() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        DONE = {"__done__": True}

        def emit(event: dict) -> None:
            queue.put_nowait(event)

        async def run() -> None:
            try:
                from app.services.track_chat_service import answer_track_chat
                emit({"type": "status", "stage": "thinking"})
                res = await answer_track_chat(req, on_event=emit)
                queue.put_nowait({
                    "type": "answer",
                    "message": res.message,
                    "web_search_used": res.web_search_used,
                })
            except Exception as exc:  # surface a terminal error frame, never hang
                logger.error("[track_chat/stream] error: %s", exc, exc_info=True)
                queue.put_nowait({"type": "error", "message": str(exc)[:200]})
            finally:
                queue.put_nowait(DONE)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep the connection warm during long LLM waits
                    continue
                if item is DONE:
                    break
                yield sse_data(item)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
