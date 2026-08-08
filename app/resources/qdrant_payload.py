"""Qdrant payload + text-embedding helpers + metadata preparation.

Extracted from legacy search_engine/utils.py during Refactor 2.
"""

from __future__ import annotations

import re

import numpy as np


# Filtering threshold reused by prepare_metadata + downstream callers.
MAX_DURATION = 420  # seconds — songs longer than this are dropped from the index

# A blank line separates paragraphs. Tolerant on purpose: 82 of 758 prod tracks
# carry CRLF line endings, and a plain ``split("\n\n")`` misses those plus every
# "blank" line that holds a stray space — 99 of 758 tracks collapsed into a
# single paragraph that way, silently skipping the dedup below.
_PARA_SPLIT = re.compile(r"\n\s*\n")


def unique_paragraphs(lyrics: str) -> list[str]:
    """Paragraphs of ``lyrics`` with duplicates dropped, IN SOURCE ORDER.

    A chorus repeated six times should not carry six times the weight in the
    track's embedding — but the verses around it must stay in the order the song
    puts them in, so the vector still describes the song and not a bag of lines.
    ``dict`` preserves insertion order; the previous ``tuple(set(...))`` did not,
    and measured on prod it reordered 711 of 758 tracks — which also meant the
    same track embedded differently on every re-index.

    Duplicates are matched on a folded key (collapsed whitespace, casefolded) so
    that "Oh-oh-oh" and "oh-oh-oh " count as one paragraph, while the text kept
    is the first occurrence exactly as written.
    """
    text = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n")
    seen: dict[str, str] = {}
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        seen.setdefault(" ".join(para.split()).casefold(), para)
    return list(seen.values())


def build_text_for_embedding(track: dict) -> str:
    """Concatenate metadata + deduplicated lyrics into one blob for encoding."""
    parts = []
    if track.get("title"):
        parts.append(f"title: {track['title']}")
    if track.get("artist"):
        parts.append(f"artist: {track['artist']}")
    if track.get("album"):
        parts.append(f"album: {track['album']}")
    if track.get("genre"):
        parts.append(f"genre: {track['genre']}")
    lyrics = "\n\n".join(unique_paragraphs(track.get("lyrics") or ""))
    if len(lyrics) > 20:
        parts.append(lyrics)
    return " | ".join(parts)


def prepare_metadata(data: dict):
    """Filter + add duration_range + year_range to each track.

    Input: dict keyed by ``artist — title`` whose values are full metadata dicts.
    Output: list of metadata dicts with original numeric ``duration`` preserved
    (Pydantic models downstream expect float seconds), plus added:
        - ``duration_range`` — IQR-based bucket label (mirrors year/year_range pair)
        - ``year_range`` — decade label, e.g. ``1990-1999``

    Note: this used to also emit ``lyrics_chunked``. Nothing ever read it, it
    shipped a near-copy of the lyrics into every Qdrant payload, and it was built
    with ``set()`` so its order changed between runs. Paragraph dedup now lives in
    :func:`unique_paragraphs`, applied where it is actually used — at encode time.

    Historical note: pre-2026-05-22 this function CLOBBERED ``duration`` with the
    bucket string, which broke Pydantic validation for the Spotify-MVP responses
    (LikedSongTrack/RecentTrack/PlaylistTrack). Fixed by 3c77220 — kept here.
    """
    data_prep = list(data.values())

    # No lyrics gate: index every track (lyric-less songs rely on the CLAP audio
    # vector + a title/artist text vector). Keep the duration cap, and require a
    # numeric duration so newly-included tracks whose duration couldn't be read
    # don't crash the comparison / percentile math below.
    filtered = [
        d for d in data_prep
        if isinstance(d.get('duration'), (int, float)) and d['duration'] <= MAX_DURATION
    ]
    durations = np.array([d['duration'] for d in filtered])

    p25 = np.percentile(durations, 25)
    p50 = np.percentile(durations, 50)
    p75 = np.percentile(durations, 75)
    iqr_custom = p50 - p25

    lower = p25 - 1.5 * iqr_custom
    upper = p75 + 1.5 * iqr_custom
    max_dur = durations.max()

    conditions = [
        durations < lower,
        (durations >= lower) & (durations < p25),
        (durations >= p25) & (durations < p50),
        (durations >= p50) & (durations <= p75),
        (durations > p75) & (durations <= upper),
        durations > upper,
    ]
    labels = [
        f'0-{int(round(lower))}',
        f'{int(round(lower))}-{int(round(p25))}',
        f'{int(round(p25))}-{int(round(p50))}',
        f'{int(round(p50))}-{int(round(p75))}',
        f'{int(round(p75))}-{int(round(upper))}',
        f'{int(round(upper))}-{int(max_dur)}',
    ]
    buckets = np.select(conditions, labels, default='')
    buckets = list(map(lambda x: str(x), list(buckets)))

    result = [
        {**d, 'duration_range': bucket}
        for d, bucket in zip(filtered, buckets)
    ]

    for track in result:
        if track.get('year'):
            decade_start = (track['year'] // 10) * 10
            track['year_range'] = f"{decade_start}-{decade_start + 9}"

    return result
