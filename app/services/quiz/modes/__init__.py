"""Mode registry — built from an explicit import list, on purpose.

``app.services.ai_tasks`` registers its members as a side effect of being
imported, and that has broken silently once already: drop the seemingly unused
``import ai_tasks`` from ``main.py`` and the endpoint starts answering HTTP 400
with nothing in the logs and every unit test still green (CLAUDE.md, invariant
3). Registering here by explicit assignment means a missing mode is an
ImportError at startup, not a mystery at runtime — and an integration test
asserts ``GET /quiz/modes`` returns every key, because that is the failure
class unit tests cannot see.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §4.
"""
from __future__ import annotations

from typing import Dict, Optional

from app.services.quiz.modes import blind_year, producer, track_snippet

MODES: Dict[str, object] = {
    track_snippet.KEY: track_snippet,
    producer.KEY: producer,
    blind_year.KEY: blind_year,
}


def get_mode(key: str) -> Optional[object]:
    """The mode module for ``key``, or None when the key is unknown."""
    return MODES.get(key)
