"""Public endpoints: GET /instance/config + POST /instance/setup. No auth —
the frontend reads /config pre-login to pick the right UX (sharing vs server);
/setup is the first-run bootstrap and self-closes after the first success."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_auth_service
from app.domain.models import AuthResponse, InstanceConfigResponse, SetupRequest
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, EmailAlreadyTakenError, InstanceAlreadyInitializedError,
    WeakPasswordError,
)


router = APIRouter(prefix="/instance", tags=["Instance"])


@router.get("/config", response_model=InstanceConfigResponse)
def get_instance_config() -> InstanceConfigResponse:
    cfg = MetadataDB.get_instance_config()
    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail="instance not initialized",
        )
    return InstanceConfigResponse(mode=cfg["mode"])


@router.post("/setup", response_model=AuthResponse)
def setup_instance(
    req: SetupRequest,
    auth: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """First-run bootstrap: create the owner, lock the mode, return a JWT.

    First-come-first-served (Grafana/Jellyfin pattern): works only while the
    instance is uninitialized; every later call gets 409. Concurrent-setup
    races resolve inside bootstrap_instance (DB constraint + rollback)."""
    try:
        user = auth.bootstrap_instance(
            email=req.email, password=req.password, mode=req.mode,
        )
    except InstanceAlreadyInitializedError:
        raise HTTPException(status_code=409, detail="instance already initialized")
    except EmailAlreadyTakenError:
        # A stray pre-owner user row holds this email — recovery is the same
        # as "already initialized": go log in / contact the admin.
        raise HTTPException(status_code=409, detail="instance already initialized")
    except WeakPasswordError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Token issued directly from the fresh user — no second login round-trip
    # (same rationale as /auth/register).
    return AuthResponse(token=auth.issue_token(user), user=user)
