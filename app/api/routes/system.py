"""System endpoints — LLM availability probe, etc."""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter

from app.domain.models import LLMStatusRequest, LLMStatusResponse

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

    The frontend passes its localStorage-configured base_url+model in the body;
    server falls back to env LLM_BASE_URL / LLM_MODEL when fields are omitted.
    Returns available=True if the upstream responds 2xx within 3s.
    """
    base_url = (req.base_url or os.getenv("LLM_BASE_URL", "")).strip()
    model = (req.model or os.getenv("LLM_MODEL", "")).strip() or None

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
    except Exception as e:
        logger.debug("[llm-status] probe failed for %s: %s", probe_url, e)
        return LLMStatusResponse(
            available=False,
            base_url=base_url,
            model=model,
            error=str(e)[:200],
        )

    return LLMStatusResponse(available=True, base_url=base_url, model=model, error=None)
