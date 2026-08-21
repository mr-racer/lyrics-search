"""Library quiz endpoints.

Two routers, for the same reason ``search.py`` has two: an ``<audio>`` element
cannot send an Authorization header, so the snippet route sits outside the
blanket ``get_current_user`` gate and authenticates itself with
``get_user_for_stream`` (Bearer **or** a short-lived ``?st=`` token). Auth is
still mandatory — only the way it arrives differs.

Everything here is a translation layer: the service raises typed errors and
this module maps them to statuses. No quiz logic lives in the router.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §12.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.api.dependencies import get_current_user, get_user_for_stream
from app.api.helpers import derive_collection_for_user
from app.resources.metadata_db import MetadataDB
from app.domain.models import (
    QuizAnswerIn,
    QuizAnswerOut,
    QuizModesResponse,
    QuizRoundOut,
    QuizRoundRequest,
    User,
)
from app.services.audio_streaming import (
    get_cached_source,
    get_streamable_path,
    put_cached_source,
)
from app.services.quiz import rounds as quiz_rounds
from app.services.quiz.errors import (
    AlreadyAnswered,
    NoRoundAvailable,
    RoundNotFound,
)

router = APIRouter(prefix="/quiz", tags=["Quiz"])

# Snippet delivery: outside the blanket auth gate, same as ``stream_router``.
stream_router = APIRouter(prefix="/quiz", tags=["Quiz"])


def _qdrant(request: Request):
    """The Qdrant client if the app has one, otherwise None.

    Deliberately not a 503. The library is read through ``light_points``, which
    serves from the SQLite ``track_metadata`` mirror first and only falls back
    to a Qdrant scroll for pre-backfill collections — so a quiz is perfectly
    playable with Qdrant unreachable. When there really is nothing to read the
    caller reports an empty pool, which is the same way the rest of the app
    degrades.
    """
    db = getattr(request.app.state, "db_client", None)
    return getattr(db, "qdrant", None) if db is not None else None


@router.get("/modes", response_model=QuizModesResponse)
def list_quiz_modes(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> QuizModesResponse:
    """Available modes with their pool sizes."""
    collection = derive_collection_for_user(current_user)
    modes = quiz_rounds.list_modes(
        qdrant_client=_qdrant(request), collection_name=collection,
    )
    return QuizModesResponse(modes=modes)


@router.post("/rounds", response_model=QuizRoundOut)
def create_quiz_round(
    req: QuizRoundRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> QuizRoundOut:
    """Build a round. 409 when the library cannot support one right now."""
    collection = derive_collection_for_user(current_user)
    try:
        built = quiz_rounds.build_round(
            qdrant_client=_qdrant(request),
            collection_name=collection,
            mode=req.mode,
            snippet_sec=req.snippet_sec,
        )
    except NoRoundAvailable as exc:
        # Not a server fault: a thin or heavily-excluded library is a normal
        # state, and the UI says so in plain words.
        raise HTTPException(status_code=409, detail=str(exc))
    return QuizRoundOut(**built)


@router.post("/rounds/{round_id}/answer", response_model=QuizAnswerOut)
def answer_quiz_round(
    round_id: str,
    req: QuizAnswerIn,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> QuizAnswerOut:
    """Score a round and reveal the truth. Rounds are single-use."""
    collection = derive_collection_for_user(current_user)
    try:
        result = quiz_rounds.submit_answer(
            qdrant_client=_qdrant(request),
            collection_name=collection,
            round_id=round_id,
            answer=req.model_dump(),
        )
    except RoundNotFound:
        raise HTTPException(status_code=404, detail="Round not found")
    except AlreadyAnswered:
        raise HTTPException(status_code=409, detail="Round already answered")
    return QuizAnswerOut(**result)


@stream_router.get("/rounds/{round_id}/audio")
async def stream_quiz_snippet(
    round_id: str,
    request: Request,
    option: str | None = Query(
        None,
        description="Play one OPTION of the round instead of the round's own "
                    "snippet. The producer round has four playable records "
                    "rather than a single thing to hear.",
    ),
    current_user: User = Depends(get_user_for_stream),
):
    """Serve the round's audio, addressed by ``round_id`` rather than track id.

    Two deliberate differences from the normal stream route:

    * the file is reached through the round, so the track id never appears in
      a URL the client can read before it answers;
    * no ``filename=`` is passed to FileResponse — the normal route sets one,
      and a Content-Disposition carrying "Kanye West - Heartless.flac" would
      give the answer away in the network panel.

    The client still seeks to ``start_sec`` and stops after ``length_sec``.
    Someone determined can play the whole file; that is accepted (spec §6) —
    this is a single-player game.
    """
    collection = derive_collection_for_user(current_user)
    try:
        track_id, _start_sec, _length_sec = quiz_rounds.resolve_round_audio(
            collection_name=collection, round_id=round_id, option_id=option,
        )
    except RoundNotFound:
        raise HTTPException(status_code=404, detail="Round not found")

    qdrant = _qdrant(request)

    async def _lookup() -> tuple[str, str | None]:
        """Resolve the file: memo, then the SQLite mirror, then Qdrant.

        Mirror before Qdrant for the same reason ``light_points`` prefers it —
        it is the authoritative local copy of the card fields and needs no
        network. Qdrant remains the fallback for pre-backfill collections, and
        is the only source of the codec hint.
        """
        row = MetadataDB.get_track_by_id(collection, track_id)
        if row and row.get("file_path"):
            return row["file_path"], None

        if qdrant is None:
            raise HTTPException(status_code=404, detail="Track not found")
        try:
            found = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: qdrant.retrieve(
                    collection_name=collection, ids=[track_id],
                    with_payload=True,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Track not found: {exc}")
        if not found:
            raise HTTPException(status_code=404, detail="Track not found")
        payload = found[0].payload or {}
        file_path = payload.get("file_path")
        if not file_path:
            raise HTTPException(status_code=404, detail="Track has no file")
        put_cached_source(collection, track_id, file_path,
                          payload.get("audio_codec"))
        return file_path, payload.get("audio_codec")

    cached = get_cached_source(collection, track_id)
    file_path, codec = cached if cached else await _lookup()
    audio_path = Path(file_path)
    if not audio_path.exists():
        file_path, codec = await _lookup()
        audio_path = Path(file_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    serve_path, content_type = await get_streamable_path(
        account_id=collection, track_id=track_id, file_path=audio_path,
        codec=codec,
    )
    return FileResponse(
        serve_path,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=600"},
    )
