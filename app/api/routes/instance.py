"""Public endpoint: GET /instance/config. No auth — the frontend reads this
pre-login to pick the right UX (sharing vs server)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.domain.models import InstanceConfigResponse
from app.resources.metadata_db import MetadataDB


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
