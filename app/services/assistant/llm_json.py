"""Tolerant JSON-object extraction from a small model's reply.

Every assistant prompt asks for one minified object and nothing else, and a 12b
model obeys that most of the time — but not deterministically. The same question
comes back as a bare object one run, wrapped in a code fence the next, and
preceded by one polite sentence the run after that. ``ask_llm(parse_json=True)``
raises on all but the first shape, which is why the assistant branches take the
raw text and dig the object out here instead.

Returning ``None`` is a normal outcome: the caller's verification gate then
rejects the answer exactly as it rejects a malformed one, which is the safe
path in both cases.
"""

from __future__ import annotations

import json


def parse_json_object(text: object) -> object:
    """Pull the first complete JSON object out of whatever the model wrapped it in.

    Handles a dict passed straight through, a code fence, and a leading or
    trailing sentence. Returns ``None`` when there is no object to find.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    body = text.strip()
    for fence in ("```json", "```"):
        if body.startswith(fence):
            body = body[len(fence):].lstrip()
    if body.endswith("```"):
        body = body[:-3].rstrip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except ValueError:
        return None
