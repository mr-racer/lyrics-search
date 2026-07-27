"""Code gate over the per-track "why this fits" line of a built playlist.

The reason is written by the curate call of ``recsys_ai_service`` and only on
the library path (``clap_search`` / ``library_search`` / ``similar_tracks``) —
``web_hits`` hardcodes it to ``None`` because no curation happened there. The
prompt already forbids filler, but a 12b model obeys that unevenly, so the UI
never shows a reason this module has not cleared.

Dropping a reason is cheap: the card renders the track without a second line
and nothing looks broken. Showing "отличный трек" is what makes the whole
feature read as noise — hence code decides, not the model.

``recsys_ai_service`` itself stays untouched (the unified-assistant spec keeps
it frozen); the gate runs in the assistant layer on the way out.
"""

from __future__ import annotations

from app.services.text_normalize import fold, tokenize

# Words that carry no information once the track's own name is subtracted, so
# they must not rescue a reason that is otherwise pure restatement.
_STOPWORDS = frozenset({
    "трек", "песня", "песню", "композиция", "это", "тут", "здесь", "и", "а",
    "the", "a", "an", "track", "song", "this", "is", "it", "of", "by",
})

# Kill switch. If a prod run shows almost nothing survives the gate, flip this
# and the reason stops reaching the frontend — no UI change needed.
SHOW_REASONS = True

MIN_LEN = 12
MAX_LEN = 90

# Filler the curate prompt bans and the model produces anyway. Matched against
# the folded reason as substrings, so declensions of a phrase are not covered —
# only the shapes actually observed. Keep it short: over-blocking silently eats
# good reasons, and a weak-but-specific line still beats none.
_FILLER = tuple(fold(p) for p in (
    "отличный трек",
    "отличная песня",
    "хороший трек",
    "хорошая песня",
    "подходит под настроение",
    "подходит к настроению",
    "создаёт настроение",
    "хороший выбор",
    "просто хит",
    "great track",
    "great song",
    "good track",
    "fits the mood",
    "fits the vibe",
    "matches the mood",
    "perfect for this",
))


def _is_filler(folded: str) -> bool:
    return any(f in folded for f in _FILLER)


def _restates_track(text: str, title: str | None, artist: str | None) -> bool:
    """True when the reason says nothing the row above it does not already say.

    Measured by subtraction rather than containment: the model both quotes the
    title verbatim and pads it ("трек «Bohemian Rhapsody»"), so what matters is
    what is left once the words already printed on the row are removed. One
    leftover word carries no information; two can ("Runaway задаёт темп").
    """
    known = set(tokenize(title or "")) | set(tokenize(artist or ""))
    rest = [t for t in tokenize(text) if t not in known and t not in _STOPWORDS]
    return len(rest) <= 1


def clean_reason(reason: object, *, title: str | None = None,
                 artist: str | None = None) -> str | None:
    """Return the reason to show, or None to show no second line at all."""
    if not SHOW_REASONS or not isinstance(reason, str):
        return None
    text = reason.strip()
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return None
    folded = fold(text)
    if not folded:
        return None
    if _is_filler(folded):
        return None
    if _restates_track(text, title, artist):
        return None
    return text


def gate_playlist(payload: dict) -> dict:
    """Apply :func:`clean_reason` to every track of a playlist payload in place."""
    for track in payload.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        track["reason"] = clean_reason(
            track.get("reason"),
            title=track.get("title"),
            artist=track.get("artist"),
        )
    return payload
