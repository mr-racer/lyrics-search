"""Owner-only admin operations.

Gated by ``get_owner`` (verifies the JWT user has role='owner'). These routes
bypass the per-account collection derivation used everywhere else, since the
owner explicitly targets ANOTHER account by ``user_id`` in the path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import get_owner
from app.api.helpers import derive_collection_for_user
from app.domain.models import User
from app.resources.metadata_db import MetadataDB

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


@router.post("/accounts/{user_id}/wipe")
def wipe_account_library(
    user_id: str,
    request: Request,
    owner: User = Depends(get_owner),  # 403 if the requester is not an owner
) -> dict:
    """Delete the target user's Qdrant collection and associated caches.

    Replaces the self-serve ``DELETE /library/collection/{name}`` removed in
    Phase D. Owner-only — designed for offboarding a member.
    """
    db_client = request.app.state.db_client
    if db_client is None:
        raise HTTPException(status_code=503, detail="Qdrant unavailable")

    # Verify the target user exists BEFORE deriving / deleting — avoids phantom
    # delete logs and typo-wipes, and guarantees user_id is a real (UUID) id.
    if MetadataDB.get_user_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown user_id: {user_id}")

    # Derive the target's collection through the same single source of truth as
    # every route (any future change to the naming flows through one place).
    target_collection = derive_collection_for_user(SimpleNamespace(id=user_id))

    # Collect track ids first (for the transcoded-cache purge below), mirroring
    # the cleanup parity the old self-serve delete_collection used to provide.
    qdrant = db_client.qdrant
    track_ids: list[str] = []
    try:
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=target_collection, limit=512, offset=offset,
                with_payload=False, with_vectors=False,
            )
            track_ids.extend(str(p.id) for p in points)
            if offset is None:
                break
    except Exception:
        track_ids = []

    try:
        qdrant.delete_collection(target_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

    # ── Best-effort cache cleanups — never fail the wipe over a stale file ──
    # 1. top-pairs JSON cache (keyed by collection name).
    try:
        from app.services.similarity_service import CACHE_DIR as _TOP_PAIRS_DIR
        (_TOP_PAIRS_DIR / f"{target_collection}.json").unlink(missing_ok=True)
    except Exception:
        pass

    # 2. transcoded audio cache (Phase B §6.6 — keyed by account/collection).
    if track_ids:
        try:
            from app.services.audio_streaming import drop_transcoded_for_tracks
            drop_transcoded_for_tracks(account_id=target_collection, track_ids=track_ids)
        except Exception:
            pass

    logger.warning(
        "[admin] owner=%s wiped collection=%s (target user_id=%s, tracks=%d)",
        owner.email, target_collection, user_id, len(track_ids),
    )

    return {
        "deleted": True,
        "user_id": user_id,
        "collection_name": target_collection,
        "tracks_purged": len(track_ids),
    }
