"""HTTP-layer helpers shared across routes.

`derive_collection_for_user` is the SINGLE source of truth that maps a
JWT-identified `User` to the Qdrant collection name. Every route MUST use this —
never derive the collection name inline, never accept it from the client.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Conservative: alnum, dash, underscore — anything else hints at injection or a
# broken upstream (e.g. user.id ended up being an email). UUID4 / UUID strings
# fit this comfortably.
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def derive_collection_for_user(user) -> str:
    """Return the Qdrant collection name owned by ``user``.

    Trust model: ``user`` comes from ``get_current_user`` which validates JWT.
    The collection name is **always** ``f"acct_{user.id}"``. Routes MUST NOT
    accept a client-supplied collection name in D-hard; during D-soft they
    accept it for backward compat but call this helper to override.
    """
    user_id = user.id  # raises AttributeError if shape is wrong → bug, not 400
    if not user_id:
        raise ValueError("user.id is empty")
    if not _VALID_USER_ID.match(user_id):
        raise ValueError(f"user.id contains invalid characters: {user_id!r}")
    return f"acct_{user_id}"


def index_root_ceiling() -> str:
    """Upper bound on the folder-index grants the owner may hand out.

    Still read from ``MEMBER_INDEX_ROOT`` so existing deployments need no config
    change, but the meaning is now strictly narrower. It used to GRANT: setting
    it let every member index that root, which meant any member could clone the
    whole by-reference library into their own collection, and the path leaked
    through the unauthenticated ``GET /instance/config``.

    It now only CONFINES: the grant itself lives per-account in
    ``users.index_root``, and this caps what may be granted. Empty/unset means
    no ceiling — grants are unrestricted, since the owner runs the host anyway.
    """
    return os.getenv("MEMBER_INDEX_ROOT", "").strip()


def path_within_root(candidate: str, root: str) -> bool:
    """True iff ``candidate`` resolves to ``root`` or a path beneath it.

    Both sides are ``resolve()``-d (symlinks + ``..`` collapsed) BEFORE comparison
    so a member can't escape via ``/music/../etc`` or a crafted relative path. An
    empty ``root`` means the feature is disabled → always False.
    """
    if not root:
        return False
    try:
        root_real = Path(root).resolve()
        cand_real = Path(candidate).resolve()
    except (OSError, ValueError):
        return False
    return cand_real == root_real or root_real in cand_real.parents


def index_grant_allows(*, role: str, index_root: str | None, candidate: str) -> bool:
    """True iff this account may point the host indexer at ``candidate``.

    The grant is PER-ACCOUNT (``users.index_root``), not an instance-wide flag:
    the streamer serves whatever ``file_path`` lands in a collection, so the
    indexer is the boundary, and it has to be drawn per caller. An owner runs
    the host machine and is unrestricted; everyone else needs an explicit grant
    and stays inside it.
    """
    if role == "owner":
        return True
    if not index_root:
        return False
    return path_within_root(candidate, index_root)


def grant_within_ceiling(grant_root: str, *, ceiling: str) -> bool:
    """True iff the owner is allowed to hand out ``grant_root``.

    Defence in depth against a fat-fingered grant: with ``MEMBER_INDEX_ROOT``
    set, a mistyped ``/`` in the admin panel would otherwise pour the whole
    host filesystem into a member's collection. An unset ceiling means the
    operator has not asked for one, so any grant is allowed. An empty grant is
    a REVOCATION, not a grant, and callers must handle it as such — it never
    passes this check.
    """
    if not grant_root:
        return False
    if not ceiling:
        return True
    return path_within_root(grant_root, ceiling)


def deprecated_collection_warning(supplied: str | None, derived: str, endpoint: str) -> None:
    """Log a structured WARNING when a client still sends ``collection_name``.

    Used during D-soft phase. The supplied value is IGNORED — we use ``derived``
    regardless. Logged at WARNING level so operators see migration progress.

    Special-case: if ``supplied == derived``, the client is effectively a no-op;
    still log at INFO (low signal) for full visibility during rollout.
    """
    if supplied is None:
        return
    if supplied == derived:
        logger.info(
            "[phase-d] %s: client supplied collection_name=%r matches derived — "
            "drop in next release",
            endpoint, supplied,
        )
        return
    logger.warning(
        "[phase-d] %s: client supplied collection_name=%r but server derived %r — "
        "client-supplied value IGNORED; update client to drop the parameter",
        endpoint, supplied, derived,
    )
