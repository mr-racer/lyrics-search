"""LLM client — OpenAI-compatible endpoint (LM Studio / Ollama / OpenAI).

Supports any server that speaks the OpenAI Chat Completions API, including
LM Studio, Ollama (with --openai-compat), and the real OpenAI API.

Configuration (lowest → highest priority):
  1. Built-in defaults (model 'openai/gpt-oss-20b', api key 'lm-studio')
  2. Environment variables  LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY
  3. Per-call keyword arguments (base_url, model, api_key) — i.e. request body
  4. Admin instance settings (instance_settings DB row) — authoritative

The top rung means a self-hosted admin's choice overrides whatever a member's
browser sends. When no instance setting is stored the chain collapses to the
historical "arg > env > default", so existing behavior is unchanged.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from openai import AsyncOpenAI
try:
    from openai import DefaultAsyncHttpxClient as _AsyncHttpxClient
except ImportError:  # very old openai SDK
    from httpx import AsyncClient as _AsyncHttpxClient

from app.services.settings_service import settings_service

logger = logging.getLogger(__name__)

# Cached clients keyed by (resolved base_url, resolved api key) — including the
# key so that rotating it via instance settings yields a fresh client rather
# than a stale cached one.
_clients: dict[tuple, AsyncOpenAI] = {}

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_API_KEY = "lm-studio"


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def resolve_base_url(arg: str | None = None) -> str | None:
    """DB instance setting > arg (request body) > env > None (unconfigured)."""
    raw = (settings_service.db_value("LLM_BASE_URL")
           or arg or os.getenv("LLM_BASE_URL", "")).strip()
    return _normalize_base_url(raw) if raw else None


def resolve_api_key(arg: str | None = None) -> str:
    """DB instance setting > arg > env > built-in 'lm-studio' default."""
    return (settings_service.db_value("LLM_API_KEY")
            or arg or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY).strip()


async def is_llm_available(base_url: str | None = None, model: str | None = None) -> bool:
    """Return True if the resolved LLM endpoint is reachable and serves the model.

    Resolves the endpoint/model from instance settings (DB > arg > env), then
    probes ``GET <base>/v1/models`` and checks the resolved model is listed.
    Mirrors the /system/llm-status check, but returns a plain bool so the
    server-mode upload flow can gate auto AI-indexing on it. ``trust_env=False``
    keeps a shell-exported HTTP_PROXY from tunnelling the (localhost) LLM probe.
    """
    resolved_url = resolve_base_url(base_url)
    if not resolved_url:
        return False
    resolved_model = (model or "").strip() or resolve_model()
    try:
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            resp = await client.get(f"{resolved_url}/models")
            resp.raise_for_status()
            data = resp.json()
        return resolved_model in [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        logger.debug("[llm_client] availability probe failed for %s: %s", resolved_url, e)
        return False


def resolve_model(arg: str | None = None) -> str:
    """DB instance setting > arg > env > built-in default model."""
    return (settings_service.db_value("LLM_MODEL")
            or arg or os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip()


def _get_client(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    """Return (and cache) an AsyncOpenAI client for the resolved base_url+key."""
    resolved_url = resolve_base_url(base_url)
    resolved_key = resolve_api_key(api_key)

    cache_key = (resolved_url or "__default__", resolved_key)
    if cache_key not in _clients:
        kwargs: dict = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        # trust_env=False: never auto-route the (usually localhost) LLM endpoint
        # through a shell-exported HTTP_PROXY. Our proxy is sourced only from .env
        # and applied explicitly where needed (see proxy_config).
        kwargs["http_client"] = _AsyncHttpxClient(trust_env=False)
        _clients[cache_key] = AsyncOpenAI(**kwargs)

    return _clients[cache_key]


async def ask_llm(
    user_message: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    extra_body: dict | None = None,
    parse_json: bool = False,
) -> str | dict:
    """Send a message to an OpenAI-compatible LLM and return the response.

    Parameters
    ----------
    user_message  : Content of the user turn.
    system_prompt : Optional system / developer prompt prepended to messages.
    model         : Model name. Falls back to $LLM_MODEL, then 'openai/gpt-oss-20b'.
    base_url      : API base URL (e.g. http://localhost:8000/v1).
                    Falls back to $LLM_BASE_URL.
    api_key       : API key. Falls back to $OPENAI_API_KEY ('lm-studio' default).
    temperature   : Sampling temperature.
    extra_body    : Extra fields forwarded in the request body.
                    Example: {"enable_thinking": False} for LM Studio.
    parse_json    : If True, strip markdown fences and return parsed dict.
                    Raises json.JSONDecodeError if the response is not valid JSON.
    """
    resolved_model = resolve_model(model)
    client = _get_client(base_url, api_key)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    call_kwargs: dict = {
        "model":       resolved_model,
        "messages":    messages,
        "temperature": temperature,
    }
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    logger.debug("LLM call: %s", call_kwargs)

    response = await client.chat.completions.create(**call_kwargs)
    content: str = (response.choices[0].message.content or "").strip()

    if not parse_json:
        return content

    # Strip markdown code fences the model may add despite instructions
    clean = content
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):].lstrip()
    if clean.endswith("```"):
        clean = clean[:-3].rstrip()

    return json.loads(clean.strip())
