"""Per-listener adaptive signal weights (design §2.3).

A skip means different things for different people. Someone who finishes almost
every track and then abandons one is telling you far more than a chronic
skipper doing the same. The fixed global constants (``W_SKIP = −0.6`` for
everyone) throw that information away.

This module derives the listener's own baseline — how often they skip, how
often they finish, how often they touch the fire/water buttons — from the event
history already loaded by ``next_chunk``, and turns it into three multipliers.
No new state: the baseline is recomputed per request from the same 2000 events.

Multipliers are capped at ×2 / ÷2 and ramp in over the first 200 events, so a
new account is never whipsawed by a handful of data points.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.stream.signals import (
    FULL_RATIO,
    FireSignal,
    PlaybackSignal,
    is_skip,
    listen_ratio,
)

# Reference rates a "typical" listener is assumed to have. A user matching
# these exactly gets multipliers of 1.0 — i.e. today's behaviour.
P_SKIP_REF = 0.25
P_FULL_REF = 0.50
P_REACT_REF = 0.08

ADAPT_CAP = 2.0     # multipliers live in [1/ADAPT_CAP, ADAPT_CAP]
EPS = 1e-3          # floor under the observed rate (no division by zero)

BASELINE_MIN = 50    # below this many events the multipliers stay at 1.0
BASELINE_FULL = 200  # at/above this they act at full strength


@dataclass(frozen=True)
class Baseline:
    """Observed rates + the three multipliers derived from them."""
    n_events: int
    p_skip: float
    p_full: float
    p_react: float
    m_skip: float
    m_full: float
    m_react: float

    def as_diagnostics(self) -> dict:
        return {
            "n_events": self.n_events,
            "p_skip": round(self.p_skip, 3),
            "p_full": round(self.p_full, 3),
            "p_react": round(self.p_react, 3),
            "m_skip": round(self.m_skip, 3),
            "m_full": round(self.m_full, 3),
            "m_react": round(self.m_react, 3),
        }


NEUTRAL = Baseline(
    n_events=0, p_skip=P_SKIP_REF, p_full=P_FULL_REF, p_react=P_REACT_REF,
    m_skip=1.0, m_full=1.0, m_react=1.0,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _multiplier(reference: float, observed: float) -> float:
    """``(ref / observed) ** 0.5``, clamped. Rarer behaviour ⇒ stronger signal.

    The square root softens the response: halving your skip rate raises the
    weight of each skip by ×1.41, not ×2.
    """
    ratio = reference / max(observed, EPS)
    return _clamp(ratio ** 0.5, 1.0 / ADAPT_CAP, ADAPT_CAP)


def _ramp(n_events: int) -> float:
    """Cold-start ramp ∈ [0, 1] over BASELINE_MIN → BASELINE_FULL events."""
    span = BASELINE_FULL - BASELINE_MIN
    if span <= 0:
        return 1.0
    return _clamp((n_events - BASELINE_MIN) / span, 0.0, 1.0)


def compute(
    signals: list[PlaybackSignal],
    taste_signals: list[FireSignal],
) -> Baseline:
    """Derive the listener baseline from the full (all-session) history.

    Only ``influence=True`` events with a known duration count: hand-queued
    tracks aren't taste, and an unknown duration makes «finished» unanswerable.
    Below BASELINE_MIN usable events the neutral baseline is returned unchanged.
    """
    usable = [
        s for s in signals
        if getattr(s, "influence", True) and listen_ratio(s.played_sec, s.total_dur) is not None
    ]
    n = len(usable)
    if n < BASELINE_MIN:
        return Baseline(
            n_events=n, p_skip=P_SKIP_REF, p_full=P_FULL_REF, p_react=P_REACT_REF,
            m_skip=1.0, m_full=1.0, m_react=1.0,
        )

    n_skip = sum(1 for s in usable if is_skip(s.played_sec, s.total_dur))
    n_full = sum(
        1 for s in usable
        if (r := listen_ratio(s.played_sec, s.total_dur)) is not None and r >= FULL_RATIO
    )
    n_react = sum(1 for t in taste_signals if t.kind in ("fire", "water"))

    p_skip = n_skip / n
    p_full = n_full / n
    p_react = n_react / n

    ramp = _ramp(n)
    return Baseline(
        n_events=n,
        p_skip=p_skip, p_full=p_full, p_react=p_react,
        m_skip=1.0 + (_multiplier(P_SKIP_REF, p_skip) - 1.0) * ramp,
        m_full=1.0 + (_multiplier(P_FULL_REF, p_full) - 1.0) * ramp,
        m_react=1.0 + (_multiplier(P_REACT_REF, p_react) - 1.0) * ramp,
    )
