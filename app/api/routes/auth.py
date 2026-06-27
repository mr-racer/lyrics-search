"""Phase A auth endpoints — login, register, me, logout, invites.

Mode-aware: register + invite admin are server-only (404 in sharing mode).
Login + me + logout work in BOTH modes (sharing has exactly one user — the
owner — but still needs to log in to obtain a JWT).
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import (
    get_auth_service, get_current_user, get_owner, require_mode,
)
from app.domain.models import (
    AuthResponse, InviteResponse, LoginRequest, RegisterRequest, User,
)
from app.services.auth_service import (
    STREAM_TOKEN_TTL_SECONDS,
    AuthError, AuthService, EmailAlreadyTakenError, InvalidCredentialsError,
    InvalidInviteError, WeakPasswordError,
)


router = APIRouter(prefix="/auth", tags=["Auth"])


def _to_invite_response(inv) -> InviteResponse:
    """Map domain Invite → public InviteResponse (drops created_by). ``link`` is
    the full registration URL, built from the server-resolved public base URL so
    it works behind a reverse proxy."""
    from app.services.public_url import invite_url
    return InviteResponse(
        code=inv.code,
        created_at=inv.created_at,
        expires_at=inv.expires_at,
        consumed=inv.consumed_at is not None,
        consumed_at=inv.consumed_at,
        link=invite_url(inv.code),
    )


# ── /auth/login ────────────────────────────────────────────────────────────
@router.post("/login", response_model=AuthResponse)
def login(
    req: LoginRequest,
    auth: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    try:
        user, token = auth.login(email=req.email, password=req.password)
    except InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return AuthResponse(token=token, user=user)


# ── /auth/register (server mode only) ──────────────────────────────────────
@router.post(
    "/register", response_model=AuthResponse,
    dependencies=[Depends(require_mode("server"))],
)
def register(
    req: RegisterRequest,
    auth: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    try:
        user = auth.register_with_invite(
            email=req.email, password=req.password, invite_code=req.invite_code,
        )
    except InvalidInviteError as e:
        raise HTTPException(status_code=400, detail=f"invite: {e}")
    except EmailAlreadyTakenError:
        raise HTTPException(status_code=409, detail="email already registered")
    except WeakPasswordError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Issue a token directly from the freshly-created user — avoids a second
    # argon2 verify (login re-hashes the password) and the limbo state where
    # the account exists but a failing second call leaves the client tokenless.
    token = auth.issue_token(user)
    return AuthResponse(token=token, user=user)


# ── /auth/me ───────────────────────────────────────────────────────────────
@router.get("/me", response_model=User)
def me(user: User = Depends(get_current_user)) -> User:
    return user


# ── /auth/stream-token ──────────────────────────────────────────────────────
@router.post("/stream-token")
def stream_token(
    user: User = Depends(get_current_user),
    auth: AuthService = Depends(get_auth_service),
) -> dict:
    """Mint a short-lived scope='stream' JWT for <audio> URLs (?st=...).
    Media elements can't send Authorization headers, so the frontend appends
    this token to /tracks/{id}/stream URLs and refreshes it periodically."""
    return {
        "token": auth.issue_stream_token(user),
        "expires_in": STREAM_TOKEN_TTL_SECONDS,
    }


# ── /auth/logout ───────────────────────────────────────────────────────────
@router.post("/logout")
def logout(_: User = Depends(get_current_user)) -> dict:
    """Server-side no-op — tokens are stateless. Client should drop the
    token from localStorage. Endpoint exists for symmetry + future
    revocation list if we add one."""
    return {"ok": True}


# ── /auth/invites (server mode only, owner-gated) ──────────────────────────
@router.post(
    "/invites", response_model=InviteResponse,
    dependencies=[Depends(require_mode("server"))],
)
def create_invite(
    owner: User = Depends(get_owner),
    auth: AuthService = Depends(get_auth_service),
) -> InviteResponse:
    try:
        inv = auth.create_invite(owner_id=owner.id)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _to_invite_response(inv)


@router.get(
    "/invites", response_model=List[InviteResponse],
    dependencies=[Depends(require_mode("server"))],
)
def list_invites(
    include_consumed: bool = Query(False),
    owner: User = Depends(get_owner),
    auth: AuthService = Depends(get_auth_service),
) -> List[InviteResponse]:
    try:
        rows = auth.list_invites(
            owner_id=owner.id, include_consumed=include_consumed,
        )
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [_to_invite_response(r) for r in rows]


@router.delete(
    "/invites/{code}",
    dependencies=[Depends(require_mode("server"))],
)
def revoke_invite(
    code: str,
    owner: User = Depends(get_owner),
    auth: AuthService = Depends(get_auth_service),
) -> dict:
    try:
        removed = auth.revoke_invite(code=code, owner_id=owner.id)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail="invite not found")
    return {"ok": True}
