"""AudioDB (theaudiodb.com) artist enrichment.

Pulls bio + mood + country + label + 2 PNGs per artist during the FACTS stage
of indexing. Mandatory: runs always, fails gracefully on network errors.

Sibling of artist_facts_service.py — same sequential-fetch-with-progress pattern.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


_FEAT_RE = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with)\s+.*$",
    re.IGNORECASE,
)


def _canonical_artist_name(artist: str) -> str:
    """Strip trailing 'feat. X' / 'ft. X' / 'featuring X' / 'with X' from artist names.

    'Dua Lipa feat. Angele' -> 'Dua Lipa'
    'Tyler, the Creator' -> 'Tyler, the Creator'  (no change — no trailing feat)
    """
    return _FEAT_RE.sub("", artist).strip()


def _audiodb_slug(canonical_artist: str) -> str:
    """Lowercase + drop URL-unsafe punctuation + join words with '+'.

    Caller is expected to pass the canonical name (apply _canonical_artist_name first).
    """
    s = canonical_artist.lower()
    # Drop punctuation. `+` is dropped because we use it as the separator below.
    s = re.sub(r"[,.'`\"!?\\/&()+]", "", s)
    # Collapse whitespace + dashes + underscores into single space, then join with +.
    s = re.sub(r"[\s\-_]+", " ", s).strip()
    return "+".join(s.split())
