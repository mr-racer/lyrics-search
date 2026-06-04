"""Phase A: AuthService — email/password identity, JWT issuance, invite-only
multi-account onboarding.

This service is the *only* place where:
  - passwords are hashed (argon2id) and verified
  - JWT tokens are signed (HS256) and verified
  - invite codes are minted, validated, and stamped consumed

Email normalization
-------------------
Emails are lower-cased on EVERY write and read so that "Alice@x.y" and
"alice@x.y" map to the same user. We do NOT slugify emails — see repo memory
note about the two divergent `_slugify` implementations; for auth we keep the
raw RFC-5322 string verbatim (lower-cased), no transformation.

Token shape
-----------
HS256 JWT with claims:
  - sub: user.id
  - email: user.email
  - role: 'owner' | 'member'
  - iat: issued-at epoch seconds
  - exp: iat + 30 days

Secret comes from MUSIX_JWT_SECRET env var (32+ chars recommended). Rotating
the secret invalidates every outstanding token — intentional kill-switch.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import time
import uuid
from typing import Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

from app.domain.models import Invite, User
from app.resources.metadata_db import MetadataDB

logger = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 30 * 24 * 3600        # 30 days
INVITE_TTL_SECONDS = 7 * 24 * 3600         # 7 days
INVITE_CODE_LENGTH = 12
MIN_PASSWORD_LENGTH = 10


class AuthError(Exception):
    """Base for auth-domain errors. Routes map these to HTTP statuses."""


class InvalidCredentialsError(AuthError):
    """Login email/password mismatch (or unknown email — same response, no enum)."""


class InvalidInviteError(AuthError):
    """Invite code unknown, already consumed, or expired."""


class EmailAlreadyTakenError(AuthError):
    """Registration: email already maps to an existing user."""


class WeakPasswordError(AuthError):
    """Registration: password fails the minimum-length policy."""


class OwnerOnlyError(AuthError):
    """Action restricted to role='owner'."""


class TokenExpiredError(AuthError):
    """JWT past its exp claim. Frontend should redirect to login."""


class InvalidTokenError(AuthError):
    """JWT malformed, bad signature, or claims missing — including unknown sub."""


class OwnerAlreadyExistsError(AuthError):
    """create_owner refused because an owner row is already present."""


def _now() -> float:
    return time.time()


def _gen_invite_code() -> str:
    """12-char URL-safe code, ~72 bits of entropy."""
    return secrets.token_urlsafe(9)[:INVITE_CODE_LENGTH]


def _row_to_user(row: dict) -> User:
    return User(
        id=row["id"], email=row["email"], role=row["role"],
        created_at=row["created_at"], last_login_at=row["last_login_at"],
    )


class AuthService:
    """Pure business logic — instantiated once in app lifespan with the shared
    MetadataDB module and the JWT secret from env. Methods are sync (MetadataDB
    is sync sqlite3); routes call them directly without await."""

    def __init__(self, db=MetadataDB, jwt_secret: str = ""):
        if not jwt_secret or len(jwt_secret) < 16:
            raise ValueError("jwt_secret must be a non-empty string of >=16 chars")
        self.db = db
        self.jwt_secret = jwt_secret
        self._hasher = PasswordHasher()

    # ── Password ────────────────────────────────────────────────────────
    def _hash_password(self, password: str) -> str:
        return self._hasher.hash(password)

    def _verify_password(self, password_hash: str, password: str) -> bool:
        try:
            self._hasher.verify(password_hash, password)
            return True
        except (VerifyMismatchError, InvalidHashError):
            return False

    @staticmethod
    def _normalize_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def _validate_password_strength(password: str) -> None:
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise WeakPasswordError(
                f"password must be at least {MIN_PASSWORD_LENGTH} characters"
            )

    # ── Registration ─────────────────────────────────────────────────────
    def create_owner(self, *, email: str, password: str) -> str:
        """Bootstrap entry point — used by scripts.create_owner CLI ONLY.

        Returns the newly-created user.id. Raises OwnerAlreadyExistsError if
        ANY owner row already exists (idempotency: re-running create_owner is
        a noop-fail, never a clobber)."""
        email = self._normalize_email(email)
        self._validate_password_strength(password)
        if self.db.has_owner():
            raise OwnerAlreadyExistsError("owner already exists")
        if self.db.get_user_by_email(email) is not None:
            raise EmailAlreadyTakenError(email)
        uid = uuid.uuid4().hex
        try:
            self.db.create_user(
                user_id=uid, email=email, password_hash=self._hash_password(password),
                role="owner", created_at=_now(),
            )
        except sqlite3.IntegrityError as exc:
            raise EmailAlreadyTakenError(email) from exc
        logger.info("[AuthService] owner created: id=%s", uid)
        return uid

    def register_with_invite(
        self, *, email: str, password: str, invite_code: str,
    ) -> User:
        """Server-mode-only path. Mode enforcement happens in the route via
        require_mode('server') — this method does NOT re-check mode."""
        email = self._normalize_email(email)
        self._validate_password_strength(password)
        invite = self.db.get_invite(invite_code)
        if invite is None:
            raise InvalidInviteError("unknown invite code")
        if invite["consumed_at"] is not None:
            raise InvalidInviteError("invite already used")
        if invite["expires_at"] < _now():
            raise InvalidInviteError("invite expired")
        if self.db.get_user_by_email(email) is not None:
            raise EmailAlreadyTakenError(email)
        uid = uuid.uuid4().hex
        now = _now()
        try:
            self.db.create_user(
                user_id=uid, email=email, password_hash=self._hash_password(password),
                role="member", created_at=now,
            )
        except sqlite3.IntegrityError as exc:
            raise EmailAlreadyTakenError(email) from exc
        self.db.consume_invite(invite_code, consumed_by=uid, consumed_at=now)
        logger.info(
            "[AuthService] member registered: id=%s via invite=%s",
            uid, invite_code[:4] + "…",
        )
        return _row_to_user(self.db.get_user_by_id(uid))

    # ── Invite admin (owner-only) ────────────────────────────────────────
    def _assert_owner(self, owner_id: str) -> None:
        row = self.db.get_user_by_id(owner_id)
        if row is None or row["role"] != "owner":
            raise OwnerOnlyError("only owner may manage invites")

    def create_invite(self, *, owner_id: str) -> Invite:
        self._assert_owner(owner_id)
        code = _gen_invite_code()
        now = _now()
        self.db.create_invite(
            code=code, created_by=owner_id,
            created_at=now, expires_at=now + INVITE_TTL_SECONDS,
        )
        return Invite(
            code=code, created_by=owner_id,
            created_at=now, expires_at=now + INVITE_TTL_SECONDS,
            consumed_by=None, consumed_at=None,
        )

    def list_invites(
        self, *, owner_id: str, include_consumed: bool = False,
    ) -> list[Invite]:
        self._assert_owner(owner_id)
        rows = self.db.list_invites(include_consumed=include_consumed)
        return [Invite(**r) for r in rows]

    def revoke_invite(self, *, code: str, owner_id: str) -> bool:
        """Delete an invite. Returns True if a row was removed, False if the
        code did not exist (lets the route surface 404 on a typo)."""
        self._assert_owner(owner_id)
        return self.db.delete_invite(code)

    # ── Token issuance / verification ───────────────────────────────────
    def issue_token(self, user: User) -> str:
        """Sign an HS256 JWT for an already-authenticated User. Public because
        register() issues a token for a freshly-created user without a second
        password round-trip."""
        now_int = int(_now())
        payload = {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "iat": now_int,
            "exp": now_int + TOKEN_TTL_SECONDS,
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> User:
        """Decode + signature-check + lookup. Raises TokenExpiredError /
        InvalidTokenError. Returns the canonical User from DB (so role/email
        changes since issuance are reflected; user-row deletion → 401)."""
        try:
            # PyJWT's built-in exp check uses its own clock; we disable it
            # and re-check manually below using `_now` so monkeypatching
            # `_now` in tests behaves the same as real wall-clock expiry.
            claims = jwt.decode(
                token, self.jwt_secret, algorithms=["HS256"],
                options={"verify_exp": False},
            )
        except jwt.PyJWTError as e:
            # Catches InvalidSignature, Decode, InvalidAlgorithm,
            # ImmatureSignature, etc. — every failure becomes a 401.
            raise InvalidTokenError(f"invalid token: {e}") from e
        exp = claims.get("exp")
        if exp is None or _now() >= float(exp):
            raise TokenExpiredError("token expired")
        uid = claims.get("sub")
        if not uid:
            raise InvalidTokenError("missing sub claim")
        row = self.db.get_user_by_id(uid)
        if row is None:
            raise InvalidTokenError(f"unknown user id {uid}")
        return _row_to_user(row)

    # ── Login ───────────────────────────────────────────────────────────
    def login(self, *, email: str, password: str) -> tuple[User, str]:
        """Verify email+password, stamp last_login_at, return (User, token).
        Raises InvalidCredentialsError on miss — same error for unknown email
        and wrong password (no user-enumeration leak)."""
        email = self._normalize_email(email)
        row = self.db.get_user_by_email(email)
        if row is None:
            raise InvalidCredentialsError("invalid credentials")
        if not self._verify_password(row["password_hash"], password):
            raise InvalidCredentialsError("invalid credentials")
        now = _now()
        self.db.update_last_login(row["id"], now)
        # Re-read after update so the returned User carries the fresh last_login_at.
        fresh = self.db.get_user_by_id(row["id"])
        if fresh is None:
            # User row vanished between password check and re-read (admin race).
            raise InvalidCredentialsError("invalid credentials")
        user = _row_to_user(fresh)
        return user, self.issue_token(user)
