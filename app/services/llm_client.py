"""LLM client — OpenAI-compatible endpoint (LM Studio / Ollama / OpenAI).

Supports any server that speaks the OpenAI Chat Completions API, including
LM Studio, Ollama (with --openai-compat), and the real OpenAI API.

Configuration (lowest → highest priority):
  1. Environment variables  LLM_BASE_URL, LLM_MODEL, OPENAI_API_KEY
  2. Per-call keyword arguments (base_url, model, api_key)
"""

from __future__ import annotations

import json
import os
from openai import AsyncOpenAI

# Cached clients keyed by resolved base_url (avoids creating a new httpx
# session on every request).
_clients: dict[str, AsyncOpenAI] = {}

def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url

def _get_client(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    """Return (and cache) an AsyncOpenAI client for the given base_url."""
    resolved_url = _normalize_base_url((base_url or os.getenv("LLM_BASE_URL", "")).strip()) or None
    resolved_key = (api_key or os.getenv("OPENAI_API_KEY", "lm-studio")).strip()

    cache_key = resolved_url or "__default__"
    if cache_key not in _clients:
        kwargs: dict = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
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
    resolved_model = (model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")).strip()
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

    print(call_kwargs)

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
