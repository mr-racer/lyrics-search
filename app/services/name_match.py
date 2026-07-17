"""Cross-script fuzzy scoring of a user-typed name against catalog values.

Shared by the playlist agent's ``fetch_filters`` and the chat planner's
``resolve_filters``: difflib alone scores «канье» vs "Kanye West" at ~0
(different scripts), so the query is expanded through
``text_normalize.translit_variants`` and compared against both the whole
folded value and its individual tokens.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from app.services.text_normalize import fold, translit_variants


def score_names(query: str, values) -> list[tuple[float, str]]:
    """Score ``values`` against ``query``, best first.

    The query's transliteration variants («канье» also matches as "kane") are
    compared with the folded value and each of its tokens, so a first-name-only
    or cross-script spelling still finds the full catalog name.
    """
    q_variants = translit_variants(query)
    scored = []
    for value in values:
        v_fold = fold(value)
        targets = [v_fold] + v_fold.split()
        best = max(
            (SequenceMatcher(None, q, t).ratio() for q in q_variants for t in targets),
            default=0.0,
        )
        scored.append((best, value))
    scored.sort(reverse=True)
    return scored
