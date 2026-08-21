"""M4 «Родословная» — which record was this track built on?

This mode teaches more than it tests. Most listeners cannot answer it cold, and
that is fine: the reveal is the payload, and a wrong guess still leaves you
knowing where a song you love came from.

Its one hard constraint: a distractor must not itself be a source of the same
track. A round with two true answers is broken, not hard — and tracks with
several credited samples are exactly the ones this mode likes to pick.

Sample coverage runs around a quarter of a library, so this mode leans hardest
on the I-5 pool gate: on a small collection it simply never appears.

Spec: docs/superpowers/specs/2026-08-21-music-quiz-design.md §7 M4.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Tuple

from app.services.quiz.context import RoundContext, RoundSpec
from app.services.quiz.errors import NoRoundAvailable
from app.services.quiz.snippet import start_point

KEY = "lineage"

HAS_AUDIO = True
INPUT_KIND = "options"

OPTION_COUNT = 4


def pool_size(ctx: RoundContext) -> int:
    """Links that could host a round: named source, track present."""
    return len(_usable(ctx, respect_exclude=False))


def build(ctx: RoundContext, *, snippet_sec: int) -> RoundSpec:
    usable = _usable(ctx) or _usable(ctx, respect_exclude=False)
    if not usable:
        raise NoRoundAvailable("no sample link has both ends")

    chosen = ctx.rng.choice(sorted(usable, key=_link_sort_key))
    track = ctx.by_id(chosen["src_track_id"])
    if track is None:
        raise NoRoundAvailable("the chosen track vanished from the snapshot")

    # Every source credited to THIS track is off the distractor list: naming a
    # second real sample as a wrong answer would make the round unanswerable.
    own = {
        _source_key(link) for link in ctx.sample_links
        if link.get("src_track_id") == chosen["src_track_id"]
    }
    others: Dict[str, Dict] = {}
    for link in _usable(ctx, respect_exclude=False):
        key = _source_key(link)
        if key in own or key in others:
            continue
        others[key] = link
    if len(others) < OPTION_COUNT - 1:
        raise NoRoundAvailable("not enough distinct sources to choose between")

    picks = ctx.rng.sample(sorted(others), OPTION_COUNT - 1)
    options = [_option(chosen)] + [_option(others[key]) for key in picks]
    correct_option_id = options[0]["option_id"]
    ctx.rng.shuffle(options)

    return RoundSpec(
        mode=KEY,
        track_id=chosen["src_track_id"],
        options=options,
        correct_option_id=correct_option_id,
        start_sec=start_point(track, snippet_sec, ctx.rng),
        length_sec=float(snippet_sec),
        reveal={"relation": chosen.get("relation") or "sample"},
        # The question is "what did THIS take from?", so the track has to be
        # named. That is not a leak: it is the subject of the sentence.
        meta={"prompt": {
            "title": track.get("title_display") or track.get("title") or "—",
            "artist": track.get("artist") or "—",
            "cover_art_path": track.get("cover_art_path"),
        }},
    )


def score(spec: Dict, answer: Dict) -> Tuple[bool, float]:
    chosen = (answer or {}).get("option_id")
    correct = bool(chosen) and chosen == spec.get("correct_option_id")
    return correct, 100.0 if correct else 0.0


# ── internals ────────────────────────────────────────────────────────────────

def _named(link: Dict) -> bool:
    return bool((link.get("dst_title") or "").strip()
                and (link.get("dst_artist") or "").strip())


def _usable(ctx: RoundContext, *, respect_exclude: bool = True) -> List[Dict]:
    out = []
    for link in ctx.sample_links:
        src = link.get("src_track_id")
        if not src or ctx.by_id(src) is None or not _named(link):
            continue
        if respect_exclude and src in ctx.exclude:
            continue
        out.append(link)
    return out


def _source_key(link: Dict) -> str:
    artist = (link.get("dst_artist") or "").strip().lower()
    title = (link.get("dst_title") or "").strip().lower()
    return f"{artist}|{title}"


def _link_sort_key(link: Dict) -> Tuple[str, str]:
    return (str(link.get("src_track_id") or ""), _source_key(link))


def _option(link: Dict) -> Dict:
    """A source, rendered like a track card but sourced from the link row.

    The source often is not in the library at all — that is the point of the
    mode — so there is no cover to show and no id to leak.
    """
    return {
        "option_id": uuid.uuid4().hex[:12],
        "title": (link.get("dst_title") or "—").strip(),
        "artist": (link.get("dst_artist") or "—").strip(),
        "cover_art_path": None,
    }
