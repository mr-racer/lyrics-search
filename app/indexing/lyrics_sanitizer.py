"""Strip provider artifacts from lyrics at ingestion. Pure stdlib.

Lyrics arrive from several places — the file's own tags (often written by the
Yandex import downloader), lyrics.ovh, and the syncedlyrics provider chain —
and every one of them mixes non-lyric text into the payload. Cleaning it once,
here, is worth far more than cleaning it in each consumer: the stored text is
what the dense vectors encode, what the player renders, what the track chat
reads and what the gems pipeline mines.

Measured on the 2026-07-26 production snapshot (5474 tracks with lyrics in the
main library):

  1115  bare ``\\r`` line separators   (20%)
   112  CJK credit headers            ("作词 : Billie Joe Armstrong/Mike Dirnt")
    91  ``[mm:ss.xx]`` timestamps

The newline case is the one that hurts most and is least visible: the player
does ``lyrics.split('\\n')``, so a fifth of the library rendered as a single
unbroken paragraph. It is also a prerequisite for the other two rules — you
cannot drop a credit *line* or a leading timestamp in text that has no lines.
"""
from __future__ import annotations

import re

# "[00:12.34]", "[1:05]", "[00:01:02.500]" — LRC timing, never lyrics. The
# pattern is deliberately narrow: "[Verse 1: Eminem]" is structure, not noise,
# and the consumers that care about it strip it themselves.
_TIMESTAMP = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\s*\]")

# Credit headers from Chinese lyric providers: 1-8 CJK characters naming a role
# followed by a colon. Twenty distinct labels appear in production (作词, 作曲,
# 编曲, 制作人, 混音师, 吉他, 贝斯, 母带工程师, 人声, 鼓, 和声, 钢琴 …), so a
# whitelist would need permanent upkeep; the shape is what identifies them.
# A Chinese lyric line of that exact shape would be lost, which is the accepted
# trade — these blocks are pure metadata and appear in Latin-script songs.
_CJK_CREDIT_LINE = re.compile("^\\s*[一-鿿]{1,8}\\s*[:：]")

_BLANK_RUN = re.compile(r"\n{3,}")


def sanitize_lyrics(text: str | None) -> str | None:
    """Provider artifacts out, lyrics untouched. ``None``/empty passes through.

    Returns ``None`` when nothing is left, so callers keep their existing
    "no lyrics" fallbacks.
    """
    if not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    kept = []
    for line in normalized.split("\n"):
        if _CJK_CREDIT_LINE.match(line):
            continue
        kept.append(_TIMESTAMP.sub("", line).rstrip())

    cleaned = _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()
    return cleaned or None
