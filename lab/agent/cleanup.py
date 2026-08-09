"""Cutting the parts of a page that carry no information.

A MediaWiki article ends in an appendix — References, External links, Further
reading — that is a third of its length and none of its meaning. Left in, it
costs three ways:

* chunks made of nothing but citation lines get embedded and reranked like any
  other, so they take slots in the top-k;
* those lines are dense with proper nouns and years, which is exactly what BM25
  scores highly, so they out-rank prose on a name query;
* it is all paid for twice, once per model.

Cut BEFORE chunking, so nothing downstream ever sees it.

The match is deliberately narrow: a heading whose entire text is one of the
known appendix titles. Not a prefix, not a substring. "References" as a whole
heading is the appendix on every article; "References to earlier work" is a
section someone wrote on purpose, and a substring rule would eat it.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# English plus the Russian equivalents, because ru.wikipedia turns up whenever
# a query slips through in Cyrillic.
APPENDIX_HEADINGS = frozenset({
    "references", "reference", "notes", "notes and references",
    "citations", "footnotes", "sources", "bibliography",
    "further reading", "external links", "see also", "works cited",
    "примечания", "ссылки", "литература", "источники", "см. также",
    "внешние ссылки", "комментарии",
})


def _normalise_heading(text: str) -> str:
    """Heading text without numbering, markup or trailing punctuation."""
    text = re.sub(r"^\s*\d+(\.\d+)*[.)]?\s+", "", text or "")
    text = re.sub(r"[*_`\[\]]", "", text)
    return " ".join(text.lower().split()).rstrip(":").strip()


def strip_appendix(markdown: str) -> tuple[str, int]:
    """Drop everything from the first appendix heading on.

    Returns ``(markdown, characters_removed)``. The heading itself goes too —
    a lone "References" at the end of the last real chunk is noise in the
    embedded text.
    """
    if not markdown:
        return markdown, 0

    lines = markdown.split("\n")
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match and _normalise_heading(match.group(2)) in APPENDIX_HEADINGS:
            kept = "\n".join(lines[:i]).rstrip()
            removed = len(markdown) - len(kept)
            # A page whose appendix starts in the first fifth is a page we
            # misread — a stub, or a heading structure we do not understand.
            # Keeping it whole is the safer error.
            if len(kept) < len(markdown) * 0.2:
                logger.info("[cleanup] appendix at %d%% of the page — left alone",
                            round(100 * len(kept) / max(len(markdown), 1)))
                return markdown, 0
            return kept, removed
    return markdown, 0
