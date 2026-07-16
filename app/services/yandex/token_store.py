"""Fernet-encrypted persistence of the per-account Yandex OAuth token.

The token blob (access/refresh/expiry/uid) is encrypted before it touches the
SQLite ``yandex_accounts`` table — plaintext OAuth tokens never hit disk. The
encryption key is resolved once, in this order:

1. ``MUSIX_YM_TOKEN_KEY`` — a urlsafe-base64 32-byte Fernet key (operator-set).
2. Derived from ``MUSIX_JWT_SECRET`` — ``urlsafe_b64encode(sha256(secret + b"ym-token"))``.

Rotating either secret invalidates every stored token (decrypt fails → treated
as "not linked"); that's an acceptable kill-switch — the user re-logs into Yandex.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

from app.resources.metadata_db import MetadataDB

logger = logging.getLogger(__name__)


class TokenStoreError(RuntimeError):
    """No usable encryption key is configured (no MUSIX_YM_TOKEN_KEY / MUSIX_JWT_SECRET)."""


def _resolve_key() -> bytes:
    """Return a valid Fernet key (urlsafe-base64 of 32 bytes)."""
    explicit = os.environ.get("MUSIX_YM_TOKEN_KEY")
    if explicit:
        # Validate eagerly so a malformed operator key fails loudly here, not
        # at the first encrypt deep inside an import job.
        try:
            Fernet(explicit.encode() if isinstance(explicit, str) else explicit)
        except Exception as e:  # noqa: BLE001 - re-raise as domain error
            raise TokenStoreError(f"MUSIX_YM_TOKEN_KEY is not a valid Fernet key: {e}") from e
        return explicit.encode() if isinstance(explicit, str) else explicit

    secret = os.environ.get("MUSIX_JWT_SECRET")
    if secret:
        digest = hashlib.sha256(secret.encode("utf-8") + b"ym-token").digest()
        return base64.urlsafe_b64encode(digest)

    raise TokenStoreError(
        "no encryption key: set MUSIX_YM_TOKEN_KEY or MUSIX_JWT_SECRET",
    )


def _fernet() -> Fernet:
    return Fernet(_resolve_key())


def save_token(
    account_id: str,
    *,
    access_token: str,
    refresh_token: str | None = None,
    expires_at: float | None = None,
    yandex_uid: str | None = None,
    yandex_login: str | None = None,
) -> None:
    """Encrypt and persist the token for ``account_id`` (re-link overwrites).

    ``yandex_login`` (display login/name, not a secret) is stored alongside
    the encrypted blob in a plaintext column so the UI can show which account
    is linked without decrypting anything.
    """
    blob = json.dumps(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "yandex_uid": yandex_uid,
        },
        ensure_ascii=False,
    )
    enc = _fernet().encrypt(blob.encode("utf-8")).decode("ascii")
    MetadataDB.init()
    MetadataDB.save_yandex_account(
        account_id=account_id, enc_token=enc,
        yandex_uid=yandex_uid, expires_at=expires_at,
        yandex_login=yandex_login,
    )


def load_token(account_id: str) -> dict | None:
    """Return the decrypted token blob, or None if not linked / undecryptable.

    A decrypt failure (key rotated, corrupted row) is logged and treated as
    "not linked" so the caller offers a re-login instead of crashing.
    """
    MetadataDB.init()
    row = MetadataDB.get_yandex_account(account_id)
    if not row:
        return None
    try:
        raw = _fernet().decrypt(row["enc_token"].encode("ascii"))
    except InvalidToken:
        logger.warning(
            "[yandex/token_store] token for account=%s failed to decrypt "
            "(key rotated?) — treating as unlinked", account_id,
        )
        return None
    blob = json.loads(raw.decode("utf-8"))
    # Surface the stored metadata alongside the decrypted secrets.
    blob.setdefault("yandex_uid", row.get("yandex_uid"))
    blob.setdefault("expires_at", row.get("expires_at"))
    blob.setdefault("yandex_login", row.get("yandex_login"))
    return blob


def delete_token(account_id: str) -> bool:
    """Unlink the account (drop the stored token). True if a row was removed."""
    MetadataDB.init()
    return MetadataDB.delete_yandex_account(account_id)


def is_linked(account_id: str) -> bool:
    """Cheap link check that does not attempt decryption."""
    MetadataDB.init()
    return MetadataDB.get_yandex_account(account_id) is not None
