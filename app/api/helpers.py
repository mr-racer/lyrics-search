"""HTTP-layer helpers shared across routes.

`derive_collection_for_user` is the SINGLE source of truth that maps a
JWT-identified `User` to the Qdrant collection name. Every route MUST use this —
never derive the collection name inline, never accept it from the client.
"""

from __future__ import annotations

import logging
import re

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
