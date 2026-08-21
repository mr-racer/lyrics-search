"""M2 «Почерк продюсера» — three tracks share a producer, one does not.

The constraints are the mode, not decoration around it. Three tracks by one
producer off one album is not a discovery, it is the definition of an album;
three by one artist asks about the artist rather than about the production.
Strip either constraint and the round can be won without hearing a thing.

The audio works differently from the recognition modes. There is no single
snippet, because the round is four records rather than one: each option plays
its own five seconds, taken a third of the way in where the production shows
rather than at a point chosen to make the track recognisable. Covers stay
visible here too — this mode asks who shaped a sound, not what the track is,
and hiding art buys nothing.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M2.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from app.services.quiz.context import RoundContext, RoundSpec
from app.services.quiz.distractors import shares_artist_or_album
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.snippet import BODY_HI, BODY_LO, start_point

KEY = "producer"

# There is no single thing to hear: the round is four records, not one. So no
# round-level snippet well...
HAS_AUDIO = False

# ...but every option is playable on its own. You cannot hear a producer's hand
# without hearing the records, and reading four titles teaches nothing.
OPTION_AUDIO = True

# Longer than the recognition modes: five seconds is about the least you need
# to hear how something was built rather than merely what it is.
OPTION_SNIPPET_SEC = 5.0

# Tracks by the shared producer in one round.
GROUP_SIZE = 3

# Half-width of the "same era" window used to pick a fair odd one out.
ERA_YEARS = 5


def pool_size(ctx: RoundContext) -> int:
    """How many producers could host a round right now."""
    return sum(1 for key in ctx.producers if _group_for(ctx, key) is not None)


def build(ctx: RoundContext, *, snippet_sec: int = 0) -> RoundSpec:
    """Assemble a round, or refuse when no producer qualifies."""
    keys = sorted(ctx.producers)
    ctx.rng.shuffle(keys)

    for key in keys:
        group = _group_for(ctx, key)
        if group is None:
            continue
        odd = _pick_odd_one_out(ctx, group, key)
        if odd is None:
            continue

        options = [_option(track, ctx.rng) for track in group + [odd]]
        correct_option_id = options[-1]["option_id"]
        ctx.rng.shuffle(options)
        return RoundSpec(
            mode=KEY,
            track_id=odd["track_id"],
            options=options,
            correct_option_id=correct_option_id,
            reveal={"producer": ctx.producers[key].get("name") or key},
        )

    raise NoRoundAvailable("no producer has three usable tracks right now")


def score(spec: Dict, answer: Dict) -> Tuple[bool, float]:
    chosen = (answer or {}).get("option_id")
    correct = bool(chosen) and chosen == spec.get("correct_option_id")
    return correct, 100.0 if correct else 0.0


# ── internals ────────────────────────────────────────────────────────────────

def _group_for(ctx: RoundContext, key: str) -> Optional[List[Dict]]:
    """Three tracks by this producer, from three artists and three albums.

    Returns None the moment that is impossible — a producer who only ever
    appears on one record, or only with one artist, cannot host a fair round.
    """
    entry = ctx.producers.get(key) or {}
    candidates = [t for t in (ctx.by_id(tid) for tid in entry.get("tracks") or [])
                  if t is not None]
    if len(candidates) < GROUP_SIZE:
        return None

    chosen: List[Dict] = []
    for track in candidates:
        # shares_artist_or_album is the same helper the M1 slate uses, so
        # "different artist" means the same thing in both modes — features
        # included.
        if any(shares_artist_or_album(track, picked) for picked in chosen):
            continue
        chosen.append(track)
        if len(chosen) == GROUP_SIZE:
            return chosen
    return None


def _pick_odd_one_out(
    ctx: RoundContext, group: List[Dict], producer_key: str,
) -> Optional[Dict]:
    """A track this producer did not make, close enough to be a fair choice.

    Preference goes to the same era and genre: an outsider from another decade
    is spotted without knowing anything about production, which is the failure
    mode this mode exists to avoid.
    """
    own = set((ctx.producers.get(producer_key) or {}).get("tracks") or [])
    group_ids = {t["track_id"] for t in group}
    years = [t.get("year") for t in group if t.get("year") is not None]
    era = sum(years) / len(years) if years else None
    genres = {(t.get("genre") or "").lower() for t in group if t.get("genre")}

    pool = [
        t for t in ctx.tracks
        if t.get("track_id") not in own
        and t.get("track_id") not in group_ids
        and t.get("track_id") not in ctx.exclude
        # Sharing an artist or album with one of the three makes the round
        # ambiguous: the outsider then looks like part of the set.
        and not any(shares_artist_or_album(t, picked) for picked in group)
    ]
    if not pool:
        return None

    def close(track: Dict) -> bool:
        if era is not None and track.get("year") is not None:
            if abs(float(track["year"]) - era) > ERA_YEARS:
                return False
        if genres and (track.get("genre") or "").lower() not in genres:
            return False
        return True

    preferred = sorted((t for t in pool if close(t)), key=lambda t: t["track_id"])
    fallback = sorted(pool, key=lambda t: t["track_id"])
    return ctx.rng.choice(preferred or fallback)


def _option(track: Dict, rng) -> Dict:
    """One rendered option, with its own playable window.

    ``track_id`` is stripped by :func:`quiz.context.public_options` before the
    round is sent — it exists here so the per-option audio route can resolve a
    file without the client ever holding an id.
    """
    return {
        "option_id": uuid.uuid4().hex[:12],
        "track_id": track.get("track_id"),
        "title": track.get("title_display") or track.get("title") or "—",
        "artist": track.get("artist") or "—",
        "cover_art_path": track.get("cover_art_path"),
        "year": track.get("year"),
        "start_sec": start_point(track, OPTION_SNIPPET_SEC, rng,
                                 lo=BODY_LO, hi=BODY_HI),
        "length_sec": OPTION_SNIPPET_SEC,
    }
