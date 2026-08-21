"""Library quiz — a game about the listener's own collection.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md

The one thing to know before changing anything here: **nothing in this package
may reach the recommender.** The quiz writes no ``taste_signals`` and no
``playback_events``, so a score can never move the taste profile (invariant
I-1), and a round's snippet is never reported as listening (I-2). The
integration suite asserts both by counting rows either side of a full round.
"""
