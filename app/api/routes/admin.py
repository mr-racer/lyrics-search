"""Owner-only admin operations.

Gated by ``get_owner`` (verifies the JWT user has role='owner'). These routes
bypass the per-account collection derivation used everywhere else, since the
owner explicitly targets ANOTHER account by ``user_id`` in the path.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import get_owner, require_mode
from app.api.helpers import derive_collection_for_user
from app.domain.models import DeleteAccountRequest, MemberResponse, User
from app.resources.metadata_db import MetadataDB

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


@router.get(
    "/members", response_model=List[MemberResponse],
    dependencies=[Depends(require_mode("server"))],
)
def list_members(_owner: User = Depends(get_owner)) -> List[MemberResponse]:
    """Owner-only roster for the 'Members' admin view: every account with the
    invite code it registered through, plus a per-account library summary
    (tracks / seconds listened / likes). Server mode only (sharing has one user).

    The collection is derived through ``derive_collection_for_user`` (the single
    source of truth) rather than reconstructed inline."""
    out: List[MemberResponse] = []
    for r in MetadataDB.list_users_with_invite():
        collection = derive_collection_for_user(SimpleNamespace(id=r["id"]))
        stats = MetadataDB.account_stats(collection)
        out.append(MemberResponse(**r, **stats))
    return out


@router.post("/accounts/{user_id}/wipe")
def wipe_account_library(
    user_id: str,
    request: Request,
    owner: User = Depends(get_owner),  # 403 if the requester is not an owner
) -> dict:
    """Delete the target user's Qdrant collection and associated caches, KEEPING
    the account. For removing the account entirely, see DELETE /admin/accounts/{id}.

    Replaces the self-serve ``DELETE /library/collection/{name}`` removed in
    Phase D. Owner-only — designed for offboarding a member's library."""
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

    # The account's fact vectors live in their own collection alongside it;
    # leaving them behind would hand them to whoever takes this id next.
    from app.services import facts_index
    facts_index.drop_collection(qdrant, target_collection)

    # ── Best-effort cache cleanups — never fail the wipe over a stale file ──
    # 1. top-pairs JSON cache (keyed by collection name).
    try:
        from app.services.similarity_service import CACHE_DIR as _TOP_PAIRS_DIR
        (_TOP_PAIRS_DIR / f"{target_collection}.json").unlink(missing_ok=True)
    except Exception:
        pass

    # 2. transcoded audio cache (Phase B §6.6 — keyed by account/collection)
    # + in-memory track-source cache for the stream hot path.
    if track_ids:
        try:
            from app.services.audio_streaming import drop_transcoded_for_tracks, drop_source_for_tracks
            drop_transcoded_for_tracks(account_id=target_collection, track_ids=track_ids)
            drop_source_for_tracks(target_collection, track_ids)
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


@router.delete(
    "/accounts/{user_id}",
    dependencies=[Depends(require_mode("server"))],
)
def delete_account(
    user_id: str,
    req: DeleteAccountRequest,
    request: Request,
    owner: User = Depends(get_owner),  # 403 if the requester is not an owner
) -> dict:
    """Fully delete a member account: SQL rows, the Qdrant collection, on-disk
    audio, transcoded cache, and reference-counted cover art. Owner-only.

    Guards (so we never brick the instance or wipe the wrong account):
      - target must exist (404);
      - target must NOT be an owner (403) — that would orphan the instance;
      - target must not be the requester themselves (403);
      - ``confirm_email`` must match the target's email (400).

    Ordering matters: cover paths are captured before any delete; SQL is wiped
    atomically; only then are now-orphaned cover files removed (a cover still
    referenced by another account is kept). Qdrant + file cleanups are
    best-effort so a missing collection / stale file never blocks the wipe.
    """
    target = MetadataDB.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown user_id: {user_id}")
    if target.get("role") == "owner":
        raise HTTPException(status_code=403, detail="cannot delete an owner account")
    if user_id == owner.id:
        raise HTTPException(status_code=403, detail="cannot delete your own account")
    if (req.confirm_email or "").strip().lower() != (target.get("email") or "").strip().lower():
        raise HTTPException(status_code=400, detail="confirm_email does not match the target account")

    collection = derive_collection_for_user(SimpleNamespace(id=user_id))

    # 1. Capture cover paths BEFORE deleting rows — the orphan check below needs
    #    the list, and the rows are about to disappear.
    try:
        cover_paths = MetadataDB.list_account_cover_paths(collection)
    except Exception:
        cover_paths = []

    # 2. Qdrant collection (best-effort; a down/absent Qdrant must not block the
    #    SQL + file wipe — otherwise the account is left half-deleted).
    db_client = request.app.state.db_client
    if db_client is not None:
        try:
            db_client.qdrant.delete_collection(collection)
        except Exception as e:
            logger.warning("[admin] delete: qdrant drop failed for %s: %s", collection, e)
        from app.services import facts_index
        facts_index.drop_collection(db_client.qdrant, collection)

    # 3. SQL — atomic. Surfaces as 500 on failure rather than orphaning files.
    MetadataDB.delete_account_data(user_id, collection)

    # 4. File cleanups — all best-effort.
    # 4a. covers: now that this account's track rows are gone, a cover with zero
    #     remaining references is a true orphan and safe to unlink. Filenames are
    #     taken as basename only and joined under COVERS_DIR — never a raw DB path.
    covers_removed = 0
    try:
        from app.indexing.cover_art import COVERS_DIR
        for cp in cover_paths:
            try:
                if MetadataDB.cover_reference_count(cp) == 0:
                    fname = Path(str(cp)).name
                    if fname:
                        (COVERS_DIR / fname).unlink(missing_ok=True)
                        covers_removed += 1
            except Exception:
                continue
    except Exception as e:
        logger.warning("[admin] delete: cover cleanup failed: %s", e)

    # 4b. on-disk audio (media/<user_id>/, keyed by user.id — isolated per account).
    try:
        from app.services.uploads_service import media_root_default
        acct_media = media_root_default() / user_id
        if acct_media.exists():
            shutil.rmtree(acct_media, ignore_errors=True)
    except Exception as e:
        logger.warning("[admin] delete: media cleanup failed: %s", e)

    # 4c. transcoded cache (keyed by collection) + top-pairs cache.
    try:
        from app.services.audio_streaming import drop_account_transcodes, drop_account_sources
        drop_account_transcodes(collection)
        drop_account_sources(collection)
    except Exception:
        pass
    try:
        from app.services.similarity_service import CACHE_DIR as _TOP_PAIRS_DIR
        (_TOP_PAIRS_DIR / f"{collection}.json").unlink(missing_ok=True)
    except Exception:
        pass

    logger.warning(
        "[admin] owner=%s DELETED account=%s (user_id=%s, collection=%s, covers_removed=%d)",
        owner.email, target.get("email"), user_id, collection, covers_removed,
    )

    return {
        "deleted": True,
        "user_id": user_id,
        "email": target.get("email"),
        "collection_name": collection,
        "covers_removed": covers_removed,
    }
