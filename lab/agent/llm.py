"""The LLM, spoken to in strict JSON and nothing else.

No tool calling anywhere in this package. The repository has been burned twice
by small models mangling tool-call syntax, and the two branches that survive on
a 12b model (``recsys_ai_service``, ``facts_executor``) both work the same way
this does: a fixed pipeline calls the model at known points, the model returns
one JSON object, and code decides what any of it means.

Two robustness measures, both cheap:

* **Extraction, not parsing.** Local models wrap JSON in prose, in ``` fences,
  or emit a reasoning preamble. The object is located in the text rather than
  the whole response being handed to ``json.loads``.
* **One repair round.** If extraction fails, the model is shown its own output
  and asked for the object alone. One, not a loop: a model that cannot produce
  JSON twice will not produce it on the fifth try, and the caller has a
  deterministic fallback for exactly this.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S | re.I)


def extract_json_object(text: str) -> Optional[dict]:
    """The first JSON object in ``text``, or None.

    Tries the whole string, then any fenced block, then a brace-matched scan —
    which is what actually catches "Here is the plan: {...} Hope this helps!".
    """
    if not text:
        return None

    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _candidates(text: str):
    stripped = text.strip()
    yield stripped
    for match in _FENCE_RE.findall(stripped):
        yield match.strip()
    depth, start, in_string, escaped = 0, None, False, False
    for i, ch in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield stripped[start:i + 1]
                start = None


class LLMClient:
    """OpenAI-compatible chat completions over httpx."""

    def __init__(self, config=None):
        from lab.agent.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.calls = 0
        self.last_raw: str = ""

    async def chat(self, messages: list[dict], *,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> str:
        """Raw completion text. Returns "" on any transport or server error."""
        import httpx

        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "temperature": (self.cfg.llm_temperature if temperature is None
                            else temperature),
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
        }
        self.calls += 1
        try:
            async with httpx.AsyncClient(timeout=self.cfg.llm_timeout) as client:
                resp = await client.post(
                    f"{self.cfg.llm_base_url}/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self.cfg.llm_api_key}"})
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 — an unreachable LLM is a normal state
            logger.warning("[llm] call failed: %s: %s", type(exc).__name__, exc)
            return ""

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            logger.warning("[llm] unexpected response shape: %.200s", data)
            return ""
        self.last_raw = text
        return text

    async def ask_json(self, messages: list[dict], *,
                       required: tuple[str, ...] = (),
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Optional[dict]:
        """One JSON object, or None.

        ``required`` names keys that must be present; a response missing one is
        treated as a failed parse and gets the single repair round, because a
        half-filled object is the shape that causes silent nonsense downstream.
        """
        text = await self.chat(messages, temperature=temperature,
                               max_tokens=max_tokens)
        obj = extract_json_object(text)
        if obj is not None and all(k in obj for k in required):
            return obj

        if not text:
            return None

        logger.info("[llm] JSON repair round (missing=%s)",
                    [k for k in required if not obj or k not in obj])
        demand = ("Reply with the JSON object only. No prose, no markdown "
                  "fence, no explanation.")
        if required:
            demand += " It must contain these keys: " + ", ".join(required) + "."
        repair = messages + [
            {"role": "assistant", "content": text[:2000]},
            {"role": "user", "content": demand},
        ]
        text = await self.chat(repair, temperature=0.0, max_tokens=max_tokens)
        obj = extract_json_object(text)
        if obj is not None and all(k in obj for k in required):
            return obj
        logger.warning("[llm] no usable JSON after repair")
        return None


def as_str(value: Any, limit: int = 300) -> str:
    """A model field coerced to a trimmed string. Lists collapse to their head."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("text") or value.get("value") or ""
    return str(value).strip()[:limit]


def as_str_list(value: Any, *, limit: int = 10, item_limit: int = 300) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value[:limit]:
        text = as_str(item, item_limit)
        if text:
            out.append(text)
    return out


def as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group()) if match else None
