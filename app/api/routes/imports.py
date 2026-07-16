"""Yandex Music import endpoints (server mode only).

Auth is a device flow: the user enters a code on Yandex, the backend polls and
stores the (encrypted) token. Then the user lists their playlists / «Мне
нравится» and starts an import job that downloads tracks and feeds them into the
existing upload-indexing pipeline.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.api.dependencies import get_current_user, require_mode
from app.domain.models import User
from app.resources.metadata_db import MetadataDB
from app.services.library_service import LibraryService
from app.services.yandex import auth as ym_auth
from app.services.yandex import playlists as ym_playlists
from app.services.yandex import token_store
from app.services.yandex.client_factory import build_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/import/yandex",
    tags=["Yandex Import"],
    dependencies=[Depends(require_mode("server"))],
)


# ── Device-flow auth ───────────────────────────────────────────────────────

@router.post("/auth/start")
async def auth_start(current_user: User = Depends(get_current_user)) -> dict:
    """Begin device-flow login; returns the code + URL the user enters on Yandex."""
    return await asyncio.to_thread(ym_auth.start_session, current_user.id)


@router.get("/auth/status")
async def auth_status(
    session_id: str = Query(..., description="session_id from /auth/start"),
    current_user: User = Depends(get_current_user),
) -> dict:
    session = ym_auth.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session


# ── Link status ────────────────────────────────────────────────────────────

@router.get("")
async def link_status(current_user: User = Depends(get_current_user)) -> dict:
    """Whether this account has a stored Yandex token (no decryption)."""
    MetadataDB.init()
    row = MetadataDB.get_yandex_account(current_user.id)
    return {
        "linked": row is not None,
        "yandex_uid": row.get("yandex_uid") if row else None,
        "yandex_login": row.get("yandex_login") if row else None,
        "expires_at": row.get("expires_at") if row else None,
    }


@router.delete("")
async def unlink(current_user: User = Depends(get_current_user)) -> dict:
    removed = token_store.delete_token(current_user.id)
    return {"unlinked": removed}


# ── Playlists ──────────────────────────────────────────────────────────────

@router.get("/playlists")
async def list_playlists(current_user: User = Depends(get_current_user)) -> dict:
    """List import sources: «Мне нравится» + the user's playlists."""
    blob = token_store.load_token(current_user.id)
    if not blob or not blob.get("access_token"):
        raise HTTPException(status_code=409, detail="yandex account not linked")

    def _load():
        client = build_client(blob["access_token"])
        return ym_playlists.list_sources(client)

    try:
        sources = await asyncio.to_thread(_load)
    except Exception as e:
        logger.warning("[import/yandex] playlists failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"yandex error: {e}")
    return {"sources": sources}


# ── Start import ───────────────────────────────────────────────────────────

class _ImportSource(BaseModel):
    # One selected source: either {"source": "likes"} or {"kind": <int>}.
    source: str | None = None   # "likes"
    kind: int | None = None     # a playlist kind


class _ImportStart(BaseModel):
    # Preferred: a list of selected sources (multi-select). Legacy single
    # source/kind at the top level is still accepted and normalized below.
    sources: list[_ImportSource] | None = None
    source: str | None = None   # "likes"  (legacy single)
    kind: int | None = None     # a playlist kind (legacy single)
    lang: str = "ru"


def _normalize_source(item: _ImportSource):
    """Turn one request item into the runner's source value, or None if invalid."""
    if item.kind is not None:
        return {"kind": item.kind}
    if item.source == "likes":
        return "likes"
    return None


@router.post("/start")
async def start_import(
    req: _ImportStart,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Start an import job (download phase → existing upload indexing).

    Accepts ``sources: [{source|kind}, ...]`` (multi-select); a legacy single
    ``source``/``kind`` at the top level is normalized into a one-item list.
    """
    if not token_store.is_linked(current_user.id):
        raise HTTPException(status_code=409, detail="yandex account not linked")

    raw_items = req.sources if req.sources else [_ImportSource(source=req.source, kind=req.kind)]
    sources = [s for s in (_normalize_source(i) for i in raw_items) if s is not None]
    if not sources:
        raise HTTPException(
            status_code=400,
            detail="provide sources=[{source:'likes'}|{kind:<int>}, ...]",
        )

    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable")

    job_id = service.enqueue_yandex_import(
        account_id=current_user.id, sources=sources, lang=(req.lang or "ru").strip().lower(),
    )
    return {"job_id": job_id}


@router.get("/status")
async def import_status(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Current import/index job status for the account (reuses the library slot)."""
    service: LibraryService = request.app.state.library_service
    if service is None:
        raise HTTPException(status_code=503, detail="Library service unavailable")
    return await service.get_status(account_id=current_user.id)
