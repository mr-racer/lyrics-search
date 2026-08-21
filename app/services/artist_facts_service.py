"""Fetch and cache interesting artist facts from SongFacts.

Facts are stored in SQLite (``MetadataDB``) for structured querying.
A fallback to the legacy ``.txt`` cache is kept for backward compatibility.
"""

import asyncio
import logging
import re
import unicodedata
from html import unescape
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from ..resources.metadata_db import MetadataDB
from app.services.proxy_config import get_proxy

logger = logging.getLogger(__name__)

# Legacy cache directory — kept for backward-compat fallback
_FACTS_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "facts"
REQUEST_TIMEOUT = 10  # seconds


def _slugify(artist: str) -> str:
    """'The Weeknd' -> 'the-weeknd'

    Normalizes Unicode equivalents (NFKC, dash variants, curly quotes) before
    slug generation, so that "Guns N' Roses" and "Guns N' Roses" produce the
    same slug regardless of source metadata. Mirrors the normalization in
    artist_split.normalize_artist_name to avoid a circular import.
    """
    s = unicodedata.normalize("NFKC", artist)
    s = re.sub(r"[‐‑‒–—―−]", "-", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201B", "'").replace("\u201A", ",")
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u201E", '"')
    cleaned_artist = " ".join(s.split())
    # Strip noisy punctuation (apostrophes, quotes, +, &, .)
    cleaned_artist = re.sub(
        "[+&.'`''‚‛""„‟′″ʼ«»]",
        '', cleaned_artist,
    )
    return "-".join(cleaned_artist.lower().split())


def _fetch_facts_html(artist: str) -> Tuple[Optional[str], bool]:
    """GET songfacts.com/facts/{slug}.

    Returns ``(html, definitive)``. ``definitive`` means the server actually
    answered: a 404 is an answer — there is no page for this artist — and is
    worth remembering. A timeout, a refused connection or a 5xx is not, and has
    to be retried on the next run rather than cached as "no facts".
    """
    slug = _slugify(artist)
    url = f"https://www.songfacts.com/facts/{slug}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, proxies=get_proxy())
        resp.raise_for_status()
        return resp.text, True
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        logger.debug("[ArtistFacts] Failed to fetch facts for %s: %s", artist, e)
        return None, bool(status and 400 <= status < 500)
    except requests.RequestException as e:
        logger.debug("[ArtistFacts] Failed to fetch facts for %s: %s", artist, e)
        return None, False


def _parse_facts(html_string: str) -> List[str]:
    """Extract fact list items from SongFacts HTML."""
    soup = BeautifulSoup(html_string, "html.parser")
    container = soup.find("ul", class_="artistfacts-results")
    if not container:
        return []

    facts: List[str] = []
    for li in container.find_all("li"):
        inner_div = li.find("div", class_="inner")
        if inner_div:
            raw = inner_div.get_text(separator=" ", strip=True)
            cleaned = re.sub(r"\s+", " ", raw).strip()
            cleaned = unescape(cleaned)
            if cleaned:
                facts.append(cleaned)
    return facts


def _legacy_facts_path(collection_name: str, artist: str) -> Path:
    """Return path to legacy cached facts file for a collection + artist."""
    coll_dir = _FACTS_CACHE_DIR / collection_name
    return coll_dir / f"{_slugify(artist)}.txt"


def get_cached_facts(collection_name: str, artist: str) -> Optional[str]:
    """Read cached facts for an artist in a collection, or None.

    Prefers SQLite; falls back to legacy ``.txt`` files.
    """
    slug = _slugify(artist)
    try:
        facts = MetadataDB.get_artist_facts(slug, collection_name)
        if facts:
            return "\n\n".join(facts)
    except Exception as e:
        logger.debug("[ArtistFacts] SQLite read failed for %s: %s", slug, e)

    # Fallback to legacy .txt
    p = _legacy_facts_path(collection_name, artist)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _save_facts_to_sqlite(collection_name: str, artist: str, facts: List[str]) -> None:
    """Save parsed facts to SQLite (with the real display name, so the
    artists row never degrades to a slug-derived placeholder)."""
    slug = _slugify(artist)
    MetadataDB.add_artist_facts_batch(
        slug, collection_name, facts, source="songfacts.com", name=artist,
    )


def _remember_miss(kind: str, slug: str) -> None:
    """Record a definitive "nothing here" so the next run skips the request."""
    try:
        MetadataDB.mark_fact_miss(kind, slug)
    except Exception:  # noqa: BLE001 — enrichment degrades, never raises
        logger.debug("[ArtistFacts] could not record miss for %s/%s", kind, slug)


async def fetch_artist_facts(
    artist: str,
    collection_name: str,
) -> Tuple[Optional[str], bool]:
    """Fetch facts for a single artist.

    Returns ``(facts text or None, hit_network)``. The second element is what
    lets the caller pay the politeness delay only when a request was actually
    made — sleeping half a second after a SQLite read adds up to minutes across
    a library.
    """
    cached = get_cached_facts(collection_name, artist)
    if cached:
        return cached, False

    # Shared-pool short-circuit: facts are keyed by slug and shared across
    # accounts — if another account already fetched this artist, don't hit
    # songfacts.com again; just make the shared facts visible to us.
    slug = _slugify(artist)
    try:
        shared = MetadataDB.get_artist_facts_any(slug)
    except Exception:
        shared = []
    if shared:
        MetadataDB.mark_visible("artist", slug, collection_name)
        return "\n\n".join(shared), False

    # The same short-circuit for the answer "there is no such page". Without it
    # every rescan re-asked for every artist songfacts.com does not know, got
    # the same 404 back, and did so forever.
    try:
        if MetadataDB.has_fact_miss("artist", slug):
            return None, False
    except Exception:  # noqa: BLE001 — a lost cache costs a request, not a run
        logger.debug("[ArtistFacts] miss lookup failed for %s", slug)

    html, definitive = await asyncio.to_thread(_fetch_facts_html, artist)
    facts = _parse_facts(html) if html else []
    if not facts:
        # Remember only a real answer. A network failure says nothing about
        # whether the page exists, and writing it down would bury the artist
        # for good after one bad night upstream.
        if definitive:
            _remember_miss("artist", slug)
        return None, True

    _save_facts_to_sqlite(collection_name, artist, facts)
    return "\n\n".join(facts), True


async def fetch_facts_for_artists(
    artists: List[str],
    collection_name: str,
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str, bool], None]] = None,
) -> Dict[str, str]:
    """Fetch facts for multiple artists sequentially with delay between requests.

    Returns dict of artist -> facts text (only for artists that had facts).
    """
    results: Dict[str, str] = {}
    total = len(artists)
    for idx, artist in enumerate(artists, 1):
        text, hit_network = await fetch_artist_facts(artist, collection_name)
        found = bool(text)
        if found:
            results[artist] = text
        logger.info(
            "[enrich] artist facts | %-8s | %s",
            "FOUND" if found else "MISSING", artist,
        )
        if progress_callback:
            progress_callback(idx, total, artist, found)
        # The delay is owed to songfacts.com, not to SQLite: an artist answered
        # from cache — a hit or a remembered miss — costs one indexed SELECT.
        if hit_network:
            await asyncio.sleep(delay)
    return results


def load_all_facts_for_collection(
    collection_name: str,
    ai_enabled: bool = True,
) -> Dict[str, str]:
    """Load all cached facts for a collection.

    When ``ai_enabled`` is True, refined facts (from AI Indexing) take
    precedence over originals for any artist that has a refined row.
    An explicit empty refined list (AI ran but kept nothing) returns
    no entry for that artist — the original facts are suppressed.

    When ``ai_enabled`` is False, only original facts are returned.

    Prefers SQLite; falls back to legacy ``.txt`` files if SQLite is empty.
    """
    # Layer 1: originals from SQLite or legacy .txt
    originals: Dict[str, str] = {}
    try:
        originals = MetadataDB.get_all_artist_facts_by_collection(collection_name)
    except Exception as e:
        logger.debug("[ArtistFacts] SQLite collection read failed: %s", e)

    if not originals:
        coll_dir = _FACTS_CACHE_DIR / collection_name
        if coll_dir.is_dir():
            for f in coll_dir.iterdir():
                if f.suffix == ".txt":
                    originals[f.stem] = f.read_text(encoding="utf-8").strip()

    # Layer 2: if AI enabled, overlay refined facts on top
    if ai_enabled:
        try:
            refined = MetadataDB.get_all_refined_artist_facts(collection_name)
            if refined:
                # Start with originals, overlay refined where available
                merged = dict(originals)
                merged.update(refined)
                return merged
        except Exception as e:
            logger.debug("[ArtistFacts] refined facts read failed: %s", e)

    return originals
