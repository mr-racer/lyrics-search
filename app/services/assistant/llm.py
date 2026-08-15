"""The LLM, spoken to in strict JSON and nothing else.

No tool calling anywhere in this package. This repository has been burned twice
by small models mangling tool-call syntax, and the branches that survive on a 12b
model all work the way this does: a fixed pipeline calls the model at known
points, the model returns one JSON object, and code decides what any of it means.

Two robustness measures on top of ``llm_client``, both cheap and both earned:

* **Extraction, not parsing.** Local models wrap JSON in prose, in ``` fences, or
  emit a reasoning preamble. ``llm_client.ask_llm(parse_json=True)`` strips a
  leading fence and then calls ``json.loads``, which raises on "Here is the plan:
  {...} Hope this helps!". The object is located in the text instead.
* **One repair round.** If extraction fails, the model is shown its own output and
  asked for the object alone. One, not a loop: a model that cannot produce JSON
  twice will not produce it on the fifth try, and every caller here has a
  deterministic fallback for exactly this.

The transport is ``llm_client``'s own cached ``AsyncOpenAI``, so instance
settings (DB > request > env) resolve exactly as they do everywhere else. What
this adds is a MESSAGE LIST — the repair round needs one, and ``ask_llm`` only
accepts a single user turn.
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


def extract_json_array(text: str) -> Optional[list]:
    """The first JSON array in ``text``, or None. Used by the CLAP rephrasing."""
    if not text:
        return None
    stripped = text.strip()
    for candidate in [stripped, *(m.strip() for m in _FENCE_RE.findall(stripped))]:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, list):
            return value
    start, end = stripped.find("["), stripped.rfind("]")
    if 0 <= start < end:
        try:
            value = json.loads(stripped[start:end + 1])
        except (ValueError, TypeError):
            return None
        if isinstance(value, list):
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
    """OpenAI-compatible chat completions, resolved through ``llm_client``."""

    def __init__(self, config=None):
        from app.services.assistant.config import AgentConfig

        self.cfg = config or AgentConfig()
        self.calls = 0
        self.last_raw: str = ""
        # Why the last call produced nothing. "Unreachable endpoint" and
        # "answered with prose" both end as an empty result upstream, and they
        # have completely different fixes — so the difference is kept.
        self.last_error: Optional[str] = None

    async def chat(self, messages: list, *, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> str:
        """Raw completion text. Returns "" on any transport or server error."""
        # Private on purpose in llm_client, but it is the seam that carries the
        # resolved base URL, key and cached client. Rebuilding a client here
        # would mean a second place that decides where the LLM lives.
        from app.services.llm_client import _get_client, resolve_model

        self.calls += 1
        self.last_error = None
        model = resolve_model(self.cfg.llm_model)
        try:
            client = _get_client(self.cfg.llm_base_url, None)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(self.cfg.llm_temperature if temperature is None
                             else temperature),
                max_tokens=max_tokens or self.cfg.llm_max_tokens,
                timeout=self.cfg.llm_timeout,
            )
        except Exception as exc:  # noqa: BLE001 — an unreachable LLM is a state
            self.last_error = f"{type(exc).__name__}: {exc} (model={model!r})"
            logger.warning("[llm] call failed — %s", self.last_error)
            return ""

        try:
            choice = response.choices[0]
            text = choice.message.content or ""
        except (AttributeError, IndexError, TypeError):
            self.last_error = f"unexpected response shape: {str(response)[:200]}"
            logger.warning("[llm] %s", self.last_error)
            return ""

        # Truncation and "the model cannot write JSON" both arrive here as an
        # unparseable reply, and they have opposite fixes — raise the ceiling
        # versus change the prompt. Only the server knows which, so ask it.
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning("[llm] output hit max_tokens=%d and was cut off — "
                           "this reply is a fragment",
                           max_tokens or self.cfg.llm_max_tokens)

        self.last_raw = text
        if not text.strip():
            self.last_error = "the model returned an empty message"
        return text

    async def ask_json(self, messages: list, *, required: tuple = (),
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

        # "No object at all" and "an object without the key" read the same in a
        # bare missing= list, and the tail is what tells a truncated reply (cut
        # mid-string) from a model that answered in prose.
        why = ("no JSON object in the reply" if obj is None else
               "keys missing: " + ", ".join(k for k in required if k not in obj))
        logger.info("[llm] JSON repair round — %s (%d chars, tail=%r)",
                    why, len(text), text[-120:])
        demand = ("Reply with the JSON object only. No prose, no markdown fence, "
                  "no explanation.")
        if required:
            demand += " It must contain these keys: " + ", ".join(required) + "."
        repair = list(messages) + [
            {"role": "assistant", "content": text[:2000]},
            {"role": "user", "content": demand},
        ]
        text = await self.chat(repair, temperature=0.0, max_tokens=max_tokens)
        obj = extract_json_object(text)
        if obj is not None and all(k in obj for k in required):
            return obj
        logger.warning("[llm] no usable JSON after repair")
        return None

    async def ask_list(self, messages: list, *,
                       max_tokens: Optional[int] = None) -> Optional[list]:
        """One JSON array, or None. No repair round — every caller of this has a
        one-line deterministic fallback, which is cheaper than a second call."""
        text = await self.chat(messages, max_tokens=max_tokens)
        return extract_json_array(text)


def as_str(value: Any, limit: int = 300) -> str:
    """A model field coerced to a trimmed string. Lists collapse to their head."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("text") or value.get("value") or ""
    return str(value).strip()[:limit]


def as_str_list(value: Any, *, limit: int = 10, item_limit: int = 300) -> list:
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
