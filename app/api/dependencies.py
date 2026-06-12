"""Phase A FastAPI dependencies — single source of truth for identity + mode.

All existing routers are wired to depend on get_current_user via the
dependencies=[...] kwarg on app.include_router() in app/api/main.py. After
Phase A, EVERY non-/auth and non-/instance endpoint requires a valid JWT.
"""
from __future__ import annotations

from typing import Callable

from fastapi import Depends, HTTPException, Request

from app.domain.models import User
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, InvalidTokenError, TokenExpiredError,
)


def get_auth_service(request: Request) -> AuthService:
    """Return the AuthService instance attached to app.state in lifespan.
    Raises 500 (via plain RuntimeError) if lifespan didn't run — that means
    the deployment is broken; no point pretending otherwise."""
    svc = getattr(request.app.state, "auth_service", None)
    if svc is None:
        raise RuntimeError(
            "auth_service is not initialized — check app lifespan + MUSIX_JWT_SECRET env"
        )
    return svc


def get_current_user(
    request: Request,
    auth: AuthService = Depends(get_auth_service),
) -> User:
    """Parse `Authorization: Bearer <token>` and return the verified User.

    401 on:
      - missing header
      - non-Bearer scheme
      - bad signature / malformed token
      - expired token
      - sub claim points at deleted user

    The response body's `detail` field carries a short reason that the frontend
    uses to distinguish 'expired' (re-auth flow) from 'malformed' (clear all
    and redirect)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header[len("Bearer "):].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return auth.verify_token(token)
    except TokenExpiredError:
        raise HTTPException(
            status_code=401, detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        # Do NOT echo the exception text: verify_token's messages may carry an
        # internal user id ("unknown user id <uid>"). A fixed string keeps the
        # boundary leak-free; the detailed reason stays in server-side logs.
        raise HTTPException(
            status_code=401, detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_for_stream(
    request: Request,
    st: str | None = None,
    auth: AuthService = Depends(get_auth_service),
) -> User:
    """Auth for the audio stream endpoint ONLY.

    <audio src=...> cannot send an Authorization header, so this dependency
    accepts EITHER the normal Bearer header (curl, tests, future native
    clients) OR a short-lived scope-limited stream token in the ?st= query
    parameter (issued by POST /auth/stream-token). Full login tokens in ?st=
    are rejected — see AuthService.verify_stream_token."""
    if request.headers.get("Authorization", "").startswith("Bearer "):
        return get_current_user(request, auth)
    if not st:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return auth.verify_stream_token(st)
    except TokenExpiredError:
        raise HTTPException(
            status_code=401, detail="stream token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=401, detail="invalid stream token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_owner(user: User = Depends(get_current_user)) -> User:
    """Same as get_current_user but adds a role gate. Returns 403 (not 401) so
    the frontend doesn't redirect to login — the user IS logged in, just not
    authorized for this surface."""
    if user.role != "owner":
        raise HTTPException(status_code=403, detail="owner only")
    return user


def require_mode(mode: str) -> Callable:
    """Factory returning a FastAPI dependency that 404s when the current
    instance_mode does not equal the requested mode. Used to surface mode-
    exclusive endpoints (e.g. /auth/register only exists in server mode)."""
    if mode not in ("sharing", "server"):
        raise ValueError(f"require_mode: invalid mode {mode!r}")

    def _dep() -> None:
        cfg = MetadataDB.get_instance_config()
        if cfg is None or cfg.get("mode") != mode:
            raise HTTPException(status_code=404, detail="endpoint not available in this mode")

    return _dep
