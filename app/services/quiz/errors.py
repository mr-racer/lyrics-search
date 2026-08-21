"""Quiz failure modes, as types rather than as None-with-a-reason.

Each maps to one HTTP status at the router boundary, so the router stays a
translation layer and the service layer never imports FastAPI.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §6.
"""
from __future__ import annotations


class QuizError(Exception):
    """Base for everything this package raises."""


class NoRoundAvailable(QuizError):
    """The library cannot support a round of this mode right now.

    Not an error in the usual sense — a thin or heavily-excluded library is a
    normal state. The router answers 409, and the UI says so in plain words
    rather than showing a broken screen.
    """


class RoundNotFound(QuizError):
    """No such round, or it belongs to another account.

    Deliberately one error for both cases: telling a caller that a round exists
    but is not theirs is itself a leak.
    """


class AlreadyAnswered(QuizError):
    """This round already has a verdict; rounds are single-use."""
