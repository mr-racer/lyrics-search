"""Public endpoints: GET /instance/config + POST /instance/setup. No auth —
the frontend reads /config pre-login to pick the right UX (sharing vs server);
/setup is the first-run bootstrap and self-closes after the first success."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_auth_service, get_owner, get_settings_service,
)
from app.domain.models import (
    AuthResponse, InstanceConfigResponse, InstanceSettingsPatch,
    InstanceSettingsResponse, SetupRequest, User,
)
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import (
    AuthService, EmailAlreadyTakenError, InstanceAlreadyInitializedError,
    WeakPasswordError,
)
from app.services.settings_service import SettingsService


router = APIRouter(prefix="/instance", tags=["Instance"])

# PATCH field (snake) → instance_settings key (mirrors env var name).
# ``embed_model`` is deliberately absent: the field is still accepted on the
# request model so cached clients don't 422, but there is one embedding model
# app-wide and nothing to store.
_FIELD_TO_KEY = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_model":    "LLM_MODEL",
    "llm_api_key":  "LLM_API_KEY",
    "clap_enabled": "CLAP_ENABLED",
    "ai_enabled":   "AI_ENABLED",
}


@router.get("/config", response_model=InstanceConfigResponse)
def get_instance_config(
    settings: SettingsService = Depends(get_settings_service),
) -> InstanceConfigResponse:
    cfg = MetadataDB.get_instance_config()
    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail="instance not initialized",
        )
    from app.api.helpers import member_index_root
    return InstanceConfigResponse(
        mode=cfg["mode"], ai_available=settings.ai_available(),
        member_index_root=(member_index_root() or None) if cfg["mode"] == "server" else None,
    )


# ── Instance settings (owner-only) ───────────────────────────────────────────
@router.get("/settings", response_model=InstanceSettingsResponse)
def get_instance_settings(
    _owner: User = Depends(get_owner),
    settings: SettingsService = Depends(get_settings_service),
) -> InstanceSettingsResponse:
    """Resolved instance settings for the owner UI. The LLM API key is masked
    (value=null, has_value bool) — the raw secret never crosses this boundary."""
    return InstanceSettingsResponse(settings=settings.public_view())


@router.patch("/settings", response_model=InstanceSettingsResponse)
def patch_instance_settings(
    req: InstanceSettingsPatch,
    _owner: User = Depends(get_owner),
    settings: SettingsService = Depends(get_settings_service),
) -> InstanceSettingsResponse:
    """Write only the fields present in the request (exclude_unset). A field set
    to null clears the override; an omitted field is left untouched. Returns the
    re-resolved (masked) view so the UI updates without a second round-trip."""
    provided = req.model_dump(exclude_unset=True)
    patch = {_FIELD_TO_KEY[f]: v for f, v in provided.items() if f in _FIELD_TO_KEY}
    settings.set_many(patch, updated_at=time.time())
    return InstanceSettingsResponse(settings=settings.public_view())


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
