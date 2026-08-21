"""The data a mode needs to build one round, and the round it produces.

A mode receives a ``RoundContext`` and never reaches for Qdrant, SQLite or the
clock itself. That is what lets the whole difficulty and slate design be tested
as plain functions over plain dicts, instead of only being judged by playing
the game and squinting.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §4.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RoundContext:
    """Everything one round may look at, gathered once by the facade."""

    collection_name: str
    tracks: List[Dict]                       # light payloads, no lyrics
    plays: Dict[str, int]                    # non-skipped play counts
    last_played: Dict[str, Optional[float]]  # epoch seconds, None if never
    percentiles: Dict[str, float]            # familiarity, 0..100
    skill: Dict                              # MetadataDB.get_quiz_skill row
    exclude: Set[str] = field(default_factory=set)   # anti-repeat
    axis_stats: Optional[Dict] = None
    # {producer_key: {"name": display, "tracks": [track_id, ...]}} over every
    # effectively-credited track. Built once per snapshot; M2 reads it.
    producers: Dict[str, Dict] = field(default_factory=dict)
    rng: object = _random
    now: float = 0.0

    def __post_init__(self) -> None:
        self._index = {t.get("track_id"): t for t in self.tracks}

    def by_id(self, track_id: str) -> Optional[Dict]:
        return self._index.get(track_id)


@dataclass
class RoundSpec:
    """A built round. ``correct_option_id`` never leaves the server."""

    mode: str
    track_id: str
    options: List[Dict]          # {option_id, title, artist, cover_art_path}
    correct_option_id: str
    start_sec: float = 0.0
    length_sec: float = 0.0
    # Facts the round can only show AFTER it is answered — the producer whose
    # three tracks those were, the year that was being guessed. Kept out of the
    # question payload entirely: anything here would give the answer away.
    reveal: Dict = field(default_factory=dict)

    def to_stored(self) -> Dict:
        """The shape persisted in ``quiz_rounds.spec_json``."""
        return {
            "mode": self.mode,
            "options": self.options,
            "correct_option_id": self.correct_option_id,
            "start_sec": self.start_sec,
            "length_sec": self.length_sec,
            "reveal": self.reveal,
        }
