"""Code gates over the LLM's classified links: the model decides, code carries.

The 2026-07-26 production audit found 2977 "samples" and 634 "sampled_by"
rows, the great majority of them wrong. Four failure shapes accounted for
almost all of it, and every one of them is decidable without a model:

- an artist "sampling" themselves — 170 of 2977 and 91 of 634 rows
  (``kanye-west/Bound 2`` (2013) was "sampled by" Kanye's own *Illest
  Motherfucker Alive* (2011));
- a link to a track on the very same album (``the-weeknd/Hardest to Love``
  "sampled by" *After Hours*);
- a source that came out later than the song that supposedly used it;
- the same pair claimed in both directions at once.

So the LLM is asked only what it is good at — *what kind* of link is this —
and everything checkable is checked here. Every rejection carries a reason so
the dry-run report can show why a link disappeared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from .library_index import LibraryIndex, artists_match, norm, title_key

# Only these two mean sound was actually taken. Everything else the classifier
# can return — lyrical_reference, cover, remix, inspiration, other — is a real
# relationship between songs, just not this one.
ACCEPTED_RELATIONS = frozenset({"sample", "interpolation"})

# A song shows at most this many links per direction; Bound 2 had eleven.
MAX_LINKS_PER_DIRECTION = 6

# Wording that claims sound was taken. A fact without any of it is not read for
# samples at all — that alone removes the Adele/Hello ("matches the melody of
# Lionel Richie's…") and Piggy Bank/Kelis ("Referring to Kelis's chorus…")
# families before a model ever sees them.
SAMPLE_LEXICON = re.compile(
    r"\bsampl(?:e|es|ed|ing)\b"
    r"|\bre-?sampl\w*"
    r"|\binterpolat\w+"
    r"|\bflip(?:s|ped|ping)?\s+(?:the|a|that)\b"
    r"|\bloop(?:s|ed|ing)?\s+(?:from|of|taken)\b"
    r"|\bbuilt\s+(?:a)?round\b"
    r"|\blifted\s+from\b"
    r"|\bborrow(?:s|ed)\s+(?:from|the)\b"
    r"|\bchopped\s+up\b",
    re.IGNORECASE,
)


def fact_mentions_sampling(fact: str) -> bool:
    """True when the fact claims, in words, that sound was taken."""
    return bool(SAMPLE_LEXICON.search(fact or ""))


def dst_key(artist: Optional[str], title: Optional[str]) -> str:
    """Stable identity of the other side — folds "Sweet Nothin's"/"Sweet
    Nothings" and "Aeroplane (Reprise)"/"Aeroplane" onto one row."""
    return f"{norm((artist or '').replace('-', ' '))}|{title_key(title or '')}"


@dataclass(frozen=True)
class SongContext:
    """Everything the gates know about the song being processed."""

    slug: str = ""
    title: str = ""
    artist: str = ""                      # slug or display name
    album: str = ""
    year: Optional[int] = None
    collection_name: str = ""
    # Normalized names the performer answers to: display name, slug, aliases,
    # and — for a group — its members. "Ye" must lose against "kanye-west".
    artist_keys: FrozenSet[str] = field(default_factory=frozenset)
    library: Optional[LibraryIndex] = None

    def own_keys(self) -> FrozenSet[str]:
        keys = set(self.artist_keys)
        primary = norm((self.artist or "").replace("-", " "))
        if primary:
            keys.add(primary)
        return frozenset(k for k in keys if k)


def _is_self(claim_artist: Optional[str], ctx: SongContext) -> bool:
    if not claim_artist:
        return False
    return any(artists_match(claim_artist, key) for key in ctx.own_keys())


def _year_conflict(direction: str, src_year, dst_year) -> bool:
    """True when the claimed direction contradicts the release order.

    Equal years pass: a song can sample another from the same year, and the
    library's year granularity cannot tell them apart anyway.
    """
    if not isinstance(src_year, int) or not isinstance(dst_year, int):
        return False
    return dst_year > src_year if direction == "source" else dst_year < src_year


def gate_claim(claim: dict, ctx: SongContext) -> Tuple[Optional[dict], Optional[str]]:
    """``(link, None)`` when the claim survives, ``(None, reason)`` otherwise."""
    relation = (claim.get("relation") or "").strip().lower()
    if relation not in ACCEPTED_RELATIONS:
        return None, f"relation:{relation or 'missing'}"

    direction = (claim.get("direction") or "source").strip().lower()
    if direction not in ("source", "usage"):
        return None, "direction:invalid"

    song = (claim.get("song") or "").strip() or None
    artist = (claim.get("artist") or "").strip() or None
    if not artist:
        # A bare title is what let "Don't Tread on Me" (1991) "sample" Imagine
        # Dragons: the library matched on the title alone.
        return None, "no-artist"
    if not song and direction == "usage":
        # "was sampled by <person>" with no track named is unshowable.
        return None, "no-song"
    if _is_self(artist, ctx):
        return None, "self-artist"
    if song and title_key(song) == title_key(ctx.title):
        return None, "self-title"

    hit = ctx.library.resolve(song, artist) if ctx.library else None
    if hit is not None:
        if ctx.album and ctx.album in (hit.get("albums") or set()):
            return None, "same-album"
        if _year_conflict(direction, ctx.year, hit.get("year")):
            return None, "year"

    return {
        "direction": direction,
        "dst_key": dst_key(artist, song),
        "dst_title": song,
        "dst_artist": artist,
        "dst_slug": ctx.library.song_slug(song, artist) if ctx.library else None,
        "relation": relation,
        "src_year": ctx.year,
        "dst_year": (hit or {}).get("year"),
        "evidence": (claim.get("evidence") or "")[:400] or None,
        "confidence": claim.get("confidence"),
    }, None


def apply_gates(
    claims: List[dict], ctx: SongContext,
) -> Tuple[List[dict], List[Tuple[dict, str]]]:
    """Filter a song's claims down to the links worth storing.

    Returns ``(links, rejections)`` where each rejection is
    ``(claim, reason)`` — the dry-run report prints them verbatim.
    """
    kept: List[dict] = []
    rejected: List[Tuple[dict, str]] = []
    by_key: dict[Tuple[str, str], dict] = {}

    for claim in claims or []:
        link, reason = gate_claim(claim, ctx)
        if link is None:
            rejected.append((claim, reason or "rejected"))
            continue
        ident = (link["direction"], link["dst_key"])
        prev = by_key.get(ident)
        if prev is None:
            by_key[ident] = link
        elif (link.get("confidence") or 0) > (prev.get("confidence") or 0):
            by_key[ident] = link

    # The same pair claimed both ways cannot both be true, and there is no
    # signal here for choosing — drop both rather than pick a coin flip.
    keys_source = {k for (d, k) in by_key if d == "source"}
    keys_usage = {k for (d, k) in by_key if d == "usage"}
    both = keys_source & keys_usage
    for ident, link in list(by_key.items()):
        if ident[1] in both:
            rejected.append((link, "direction-echo"))
            del by_key[ident]

    for direction in ("source", "usage"):
        group = [ln for (d, _k), ln in by_key.items() if d == direction]
        group.sort(key=lambda ln: (-(ln.get("confidence") or 0), ln["dst_key"]))
        kept.extend(group[:MAX_LINKS_PER_DIRECTION])
        for extra in group[MAX_LINKS_PER_DIRECTION:]:
            rejected.append((extra, "cap"))

    return kept, rejected


def links_to_relations(links: List[dict]) -> dict:
    """The ``songs.samples_json`` shape, built from gated links."""
    samples, sampled_by = [], []
    for ln in links:
        entry = {"song": ln.get("dst_title"), "artist": ln.get("dst_artist")}
        (samples if ln["direction"] == "source" else sampled_by).append(entry)
    return {"samples": samples, "sampled_by": sampled_by}
