"""Typed failures for the three model legs.

Every leg used to answer failure with ``None``, and the four call sites that
consumed one of those ``None``s each invented their own recovery:
``bio_v2/article.py`` scored every candidate 1.0 and admitted the entire pool,
``sources._gate_chunks`` returned every chunk ungated, ``_gate_hits`` kept the
top result, and ``retrieval.facet_sentences`` returned nothing at all. One
cause, four behaviours, three of them unlogged — and the first has already made
a measurement report four artists as having a Wikipedia article that does not
exist.

The rule this module enforces: **a failure raises.** ``None`` and empty results
survive only for "you passed no input". What to do about a failure then becomes
a decision each caller makes in writing, which is the part that was missing.

Degrading is still right in the retriever — ranking on two signals beats not
answering — but it has to be a written decision at a named site, not the
default that falls out of a bare ``None``.
"""

from __future__ import annotations

from typing import Optional

#: The three legs, as they are named in logs, counters and ``retrieval_status``.
LEGS = ("dense", "sparse", "cross_encoder")


class ModelError(RuntimeError):
    """A model leg could not do what was asked.

    ``leg`` and ``op`` are attributes rather than prose inside the message
    because handlers dispatch on them: the retriever drops the signal named by
    ``leg``, and the HTTP layer maps the subclass to a status code. A caller
    that only wants to log still gets a readable ``str()``.
    """

    def __init__(self, leg: str, op: str, message: str, *,
                 cause: Optional[BaseException] = None) -> None:
        self.leg = leg
        self.op = op
        self.cause = cause
        super().__init__(f"[{leg}/{op}] {message}")


class ModelUnavailable(ModelError):
    """The weights are not there — never loaded, or the breaker is still open.

    A state rather than an accident: answered with 503, retried later, and
    deliberately not worth a stack trace.
    """


class ModelOverloaded(ModelError):
    """The work could not be admitted — the queue is full or the wait expired.

    Also a state, and the one failure a client can act on by simply trying
    again, which is why it is answered with 429 and a ``Retry-After`` rather
    than folded into a generic 500.
    """


class ModelOOM(ModelError):
    """The allocator refused even at a single item.

    Kept apart from :class:`ModelEncodeFailed` because it says something about
    the MACHINE rather than the request: the card is shared with a separately
    launched LLM whose footprint moves underneath us, so this one can clear on
    its own and the same call may well succeed a minute later.
    """


class ModelEncodeFailed(ModelError):
    """Anything else on the encode path — the one subclass worth a traceback."""
