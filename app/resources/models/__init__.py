"""Shared vocabulary for the model legs: failures, counters, load state.

Deliberately free of ``torch`` and of ``model_registry`` itself — this package
is imported at module level by the registry, by the retrieval hub and by every
caller that has to catch something, and unit tests run with the whole ML stack
stubbed out. Nothing here may make that untrue.
"""

from .breaker import CircuitBreaker
from .errors import (
    LEGS,
    ModelEncodeFailed,
    ModelError,
    ModelOOM,
    ModelOverloaded,
    ModelUnavailable,
)
from .stats import STATS, ModelStats

__all__ = [
    "CircuitBreaker",
    "LEGS",
    "ModelEncodeFailed",
    "ModelError",
    "ModelOOM",
    "ModelOverloaded",
    "ModelUnavailable",
    "ModelStats",
    "STATS",
]
