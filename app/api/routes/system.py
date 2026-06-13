"""System endpoints — LLM availability probe, etc."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

from app.domain.models import LLMStatusRequest, LLMStatusResponse
from app.services.llm_client import resolve_base_url, resolve_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["System"])

PROBE_TIMEOUT_SEC = 3.0


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


@router.post("/llm-status", response_model=LLMStatusResponse)
async def llm_status(req: LLMStatusRequest) -> LLMStatusResponse:
    """Probe the configured LLM endpoint with a lightweight GET /models.

    The body wins so the owner can test a CANDIDATE endpoint (typed in the
    wizard) before saving it. When the body is empty we fall back to the
    instance-settings resolver (DB > env > default), so a later "test current
    config" probe works too.
    """
    raw = (req.base_url or "").strip()
    base_url = _normalize_base_url(raw) if raw else resolve_base_url()
    model = (req.model or "").strip() or resolve_model()

    if not base_url:
        return LLMStatusResponse(
            available=False,
            base_url=None,
            model=model,
            error="no base_url configured (set llm_base_url or LLM_BASE_URL env)",
        )

    base_url = _normalize_base_url(base_url)
    probe_url = f"{base_url}/models"

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SEC) as client:
            resp = await client.get(probe_url)
            resp.raise_for_status()
        data = resp.json()
        if model in [m["id"] for m in data["data"]]:
            return LLMStatusResponse(available=True, base_url=base_url, model=model, error=None)
        else:
            return LLMStatusResponse(available=False, base_url=base_url, model=model, error="model is not found in v1/models")
    except Exception as e:
        logger.debug("[llm-status] probe failed for %s: %s", probe_url, e)
        return LLMStatusResponse(
            available=False,
            base_url=base_url,
            model=model,
            error=str(e)[:200],
        )