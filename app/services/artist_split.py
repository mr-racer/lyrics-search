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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.services.artist_facts_service import _slugify

_RULES_PATH = Path(__file__).with_name("artist_split_rules.json")


def normalize_artist_name(name: str) -> str:
    """Normalize artist name: Unicode equivalents, collapse whitespace.

    Handles the common ID3/MP3 tag issues:
    - NFKC normalization (™→TM, ﬁ→fi, etc.)
    - Unicode dash variants → ASCII hyphen
    - Unicode quotes → ASCII equivalents (preserved, not deleted)
    - Collapse whitespace (NBSP, zero-width, etc.)

    This is applied BEFORE slug generation and external API calls so that
    "Guns N' Roses" (with curly quote) and "Guns N' Roses" (straight)
    produce the same slug and search query.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name)
    # Dash variants → ASCII hyphen
    s = re.sub(r"[‐‑‒–—―−]", "-", s)
    # Unicode quotes → ASCII equivalents (preserved for display)
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201B", "'").replace("\u201A", ",")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u201E", '"')
    # Collapse all whitespace variants (NBSP, zero-width, etc.) into single space
    s = " ".join(s.split())
    return s.strip()

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
    """['Kanye West', 'Sia'] from 'Kanye West, Sia'. Primary first, deduped.

    Applies ``normalize_artist_name`` before splitting so that Unicode dashes
    and quotes in source metadata do not produce garbled participant names.
    """
    norm = normalize_artist_name(raw or "")
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


# ─── Feat-in-title parsing ───────────────────────────────────────────────────
#
# A large share of real-world files carry the collaboration in the TITLE tag
# («Bangarang (ft. Sirah)», «Ride ft. MUTEMATH») with the featured artist
# absent from the artist tag entirely (381 of 388 such tracks in the reference
# 5.6k library). These parsers extract that credit so the featured artist gets
# a clickable ref + artist-page binding, and the title can render clean.

_FEAT_KW = r"(?:featuring|feat\.?|ft\.?)"
# Bracketed chunk, optional prefix before the keyword: «(Bad Meets Evil Feat.
# Bruno Mars)» keeps «(Bad Meets Evil)» in the title. \b rejects keywords glued
# to a word («Swift», «Shift») and fires on «(feat…» / «… feat…» alike.
_TITLE_PAREN_FEAT_RE = re.compile(
    rf"[(\[]\s*(?P<prefix>[^()\[\]]*?)\b{_FEAT_KW}\s+(?P<names>[^()\[\]]+?)\s*[)\]]",
    re.IGNORECASE,
)
# «(with X)» / «[with X]» — a co-lead billing, NOT a feat (industry convention:
# Spotify shows «Adele, Erroll Garner» for «(with Erroll Garner)»). Bare
# «with» is deliberately not honored — too many legit titles contain the word.
_TITLE_PAREN_WITH_RE = re.compile(
    r"[(\[]\s*with\s+(?P<names>[^()\[\]]+?)\s*[)\]]",
    re.IGNORECASE,
)
# Bare «… ft. X» to end of string (applied after bracketed chunks are gone).
# The keyword must be followed by a name, so «10,000 Ft.» never matches.
_TITLE_BARE_FEAT_RE = re.compile(
    rf"\s+{_FEAT_KW}\s+(?P<names>\S.*)$",
    re.IGNORECASE,
)
# Separators between names INSIDE a credit chunk. Here «&» is a list separator
# between guests (unlike the artist tag, where «A & B» is a co-primary duo).
# Handles the Oxford comma («Freeway, Beanie Sigel, & Murphy Lee») because
# empty parts are dropped.
_CREDIT_NAME_SEP_RE = re.compile(
    r"\s*(?:,|&|\+|/)\s*|\s+(?:and|и|x)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TitleFeatParse:
    """Result of ``parse_title_feat``: cleaned title + extracted credits."""
    clean_title: str
    feat_names: list[str] = field(default_factory=list)
    with_names: list[str] = field(default_factory=list)


def _split_credit_names(chunk: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in _CREDIT_NAME_SEP_RE.split(chunk or ""):
        part = normalize_artist_name(part)
        if not part:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def parse_title_feat(title: str | None) -> TitleFeatParse:
    """Extract feat/with credits out of a track TITLE.

    Returns the cleaned title (surrounding text preserved: «Starboy (ft. Daft
    Punk) [Kygo Remix]» → «Starboy [Kygo Remix]») plus the extracted names.
    Idempotent: a clean title round-trips unchanged with empty credit lists.
    """
    s = " ".join((title or "").split())
    if not s:
        return TitleFeatParse(clean_title="")
    feat_names: list[str] = []
    with_names: list[str] = []

    def _take_feat(m: re.Match) -> str:
        feat_names.extend(_split_credit_names(m.group("names")))
        prefix = m.group("prefix").strip(" -–—")
        return f"({prefix})" if prefix else ""

    def _take_with(m: re.Match) -> str:
        with_names.extend(_split_credit_names(m.group("names")))
        return ""

    s = _TITLE_PAREN_FEAT_RE.sub(_take_feat, s)
    s = _TITLE_PAREN_WITH_RE.sub(_take_with, s)
    m = _TITLE_BARE_FEAT_RE.search(s)
    if m:
        feat_names.extend(_split_credit_names(m.group("names")))
        s = s[: m.start()]
    clean = " ".join(s.split()).strip(" -–—")
    if not clean:
        # Title was ONLY a credit chunk — keep the original rather than
        # rendering an empty string.
        clean = " ".join((title or "").split())
    return TitleFeatParse(
        clean_title=clean, feat_names=feat_names, with_names=with_names
    )


def _resolved_slug(name: str) -> str:
    slug = _slugify(name)
    return _ALIASES.get(slug, slug)


def tag_feat_slugs(raw: str | None) -> set[str]:
    """Slugs of participants credited after a feat keyword in the ARTIST tag.

    «Drake feat. Future» → {"future"}. Only feat/ft/featuring mark guests —
    «with»/«vs»/«x»/«&» collaborators are co-leads, not feats. Historically the
    frontend faked this by position («everyone after the first ref is a
    feat»), which mislabeled duos; this keeps the real tag signal instead.
    """
    norm = normalize_artist_name(raw or "")
    if not norm or _slugify(norm) in _KNOWN_GROUPS:
        return set()
    m = _TITLE_BARE_FEAT_RE.search(norm)
    if not m:
        return set()
    return {
        s for s in (_resolved_slug(n) for n in split_artists(m.group("names"))) if s
    }


def artist_refs_for_track(track: dict | None) -> list[dict]:
    """Role-aware ``[{"name", "slug", "role"}]`` for a track payload/row dict.

    Preferred source: payload arrays written at index/backfill time —
    ``artists``/``artist_slugs`` (aligned) + ``feat_artist_slugs`` marking
    which of them are title-feats. Legacy payloads (no ``feat_artist_slugs``)
    fall back to parsing the raw ``artist`` tag plus the title on the fly, so
    responses are enriched even before a backfill ran.

    Roles: co-leads (tag «&»/«,»/«with» participants, title «(with X)») are
    ``main``; artists credited after a feat keyword — in the title OR in the
    tag («A feat. B») — are ``feat``.
    """
    track = track or {}
    names = track.get("artists") or []
    slugs = track.get("artist_slugs") or []
    feat_slugs = track.get("feat_artist_slugs")
    aligned = bool(names and slugs and len(names) == len(slugs))
    if isinstance(feat_slugs, list) and aligned:
        feat_set = set(feat_slugs)
        return [
            {"name": n, "slug": s, "role": "feat" if s in feat_set else "main"}
            for n, s in zip(names, slugs)
        ]
    if aligned:
        refs = [{"name": n, "slug": s, "role": "main"} for n, s in zip(names, slugs)]
    else:
        refs = [{**r, "role": "main"} for r in artist_refs(track.get("artist"))]
    by_slug = {r["slug"]: r for r in refs}
    for slug in tag_feat_slugs(track.get("artist")):
        if slug in by_slug:
            by_slug[slug]["role"] = "feat"
    parsed = parse_title_feat(track.get("title"))
    extracted = [(n, "main") for n in parsed.with_names]
    extracted += [(n, "feat") for n in parsed.feat_names]
    for name, role in extracted:
        slug = _resolved_slug(name)
        if not slug:
            continue
        existing = by_slug.get(slug)
        if existing is not None:
            # Already credited (tag or SQLite-mirrored arrays, which don't
            # store roles) — the title says they're a feat, so upgrade the
            # role instead of dropping the signal.
            if role == "feat":
                existing["role"] = "feat"
            continue
        ref = {"name": name, "slug": slug, "role": role}
        by_slug[slug] = ref
        refs.append(ref)
    return refs


def display_title_for_track(track: dict | None) -> str | None:
    """Cleaned display title for a payload/row dict, or None when unchanged.

    Prefers the ``title_display`` payload field (index/backfill time). On a
    backfilled payload (``feat_artist_slugs`` present) an absent
    ``title_display`` means the title is already clean — skip re-parsing.
    """
    track = track or {}
    td = track.get("title_display")
    if td:
        return str(td)
    if isinstance(track.get("feat_artist_slugs"), list):
        return None
    title = track.get("title") or ""
    parsed = parse_title_feat(title)
    normalized = " ".join(title.split())
    if parsed.clean_title and parsed.clean_title != normalized:
        return parsed.clean_title
    return None


def artist_refs(raw: str | None) -> list[dict]:
    """Aligned ``[{"name", "slug"}]`` participants for a raw artist tag.

    Pairs each canonical slug (``artist_slugs``: deduped, alias-resolved) with
    its display name via ``name_for_slug``, so name and slug always describe the
    SAME participant. Zipping ``split_artists`` with ``artist_slugs`` would
    misalign when aliases collapse (e.g. "Ye, Kanye West" → 2 names but 1 slug).

    Returns a list of plain dicts so it coerces directly into a Pydantic
    ``list[ArtistRef]`` field on track models — letting the frontend render each
    collaborator as its own clickable link to the right artist page.
    """
    out: list[dict] = []
    for slug in artist_slugs(raw):
        name = name_for_slug(raw, slug) or slug.replace("-", " ").title()
        out.append({"name": name, "slug": slug})
    return out
