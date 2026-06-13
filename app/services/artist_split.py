"""Split a raw ``artist`` tag into participating artists + canonical slugs.

Pure module. The raw tag mixes collaborators with varied separators
("Kanye West, Sia", "Drake feat. Future", "Calvin Harris & Dua Lipa").
We split into ordered participants (primary first) and canonical slugs.

Matching elsewhere is by WHOLE-SLUG equality, never substring — so "ye"
can never match inside "kanye-west". Curated ``artist_split_rules.json``
holds (a) ``known_groups`` whose names legitimately contain separators and
must NOT be split, and (b) ``aliases`` mapping an alias slug to a canonical
slug.

IMPORTANT: slugs are produced by ``artist_facts_service._slugify`` — the SAME
function the artist page uses to build its query slug. Do not substitute
``metadata_db._slugify`` (different normalization).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.services.artist_facts_service import _slugify

_RULES_PATH = Path(__file__).with_name("artist_split_rules.json")

# Word separators are only honored when surrounded by whitespace, so names
# like "SBTRKT" / "Charli XCX" / "MGMT" are never torn apart.
_WORD_SEP = r"(?:featuring|feat\.?|ft\.?|vs\.?|with|x)"
_SEP_RE = re.compile(rf"\s+{_WORD_SEP}\s+|\s*[,;/&+]\s*", re.IGNORECASE)


def _load_rules() -> tuple[set[str], dict[str, str]]:
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set(), {}
    groups = {_slugify(g) for g in data.get("known_groups", []) if g}
    # Slug-normalize keys so an alias fires regardless of how it's written in JSON
    # ("Ye" / "ye" both map). Values are already canonical slugs.
    aliases = {_slugify(k): str(v) for k, v in (data.get("aliases") or {}).items()}
    return groups, aliases


_KNOWN_GROUPS, _ALIASES = _load_rules()


def split_artists(raw: str | None) -> list[str]:
    """['Kanye West', 'Sia'] from 'Kanye West, Sia'. Primary first, deduped."""
    norm = " ".join((raw or "").split())
    if not norm:
        return []
    # Known-group protection only fires when the WHOLE string is a known group.
    # A known group appearing as a collaborator (e.g. "Earth, Wind & Fire feat. X")
    # can still be split — accepted as a rare edge case (no multi-pass matching).
    if _slugify(norm) in _KNOWN_GROUPS:
        return [norm]
    out: list[str] = []
    seen: set[str] = set()
    for part in _SEP_RE.split(norm):
        part = part.strip()
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    # `out` is empty only for separator-only input (e.g. "&", ","), where every
    # split token is blank — fall back to the normalized raw string.
    return out or [norm]


def artist_slugs(raw: str | None) -> list[str]:
    """Canonical slugs for each participant, alias-resolved, deduped in order."""
    out: list[str] = []
    seen: set[str] = set()
    for name in split_artists(raw):
        slug = _slugify(name)
        slug = _ALIASES.get(slug, slug)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
    return out


def primary_artist(raw: str | None) -> str:
    """First (primary) participant, or the trimmed raw string if unsplittable."""
    parts = split_artists(raw)
    return parts[0] if parts else " ".join((raw or "").split())


def name_for_slug(raw: str | None, slug: str) -> str | None:
    """The participant display name whose canonical slug == ``slug``.

    Labels a collaborator correctly on the artist page: "Dua Lipa x Angele" must
    show "Dua Lipa" on the dua-lipa page (and "Angele" on the angele page), never
    the whole raw tag. Slug computation mirrors ``artist_slugs`` exactly
    (``_slugify`` + alias resolution). Returns None when no participant maps.
    """
    for name in split_artists(raw):
        if _ALIASES.get(_slugify(name), _slugify(name)) == slug:
            return name
    return None
