"""Fetch song facts + producer/label credits from Genius.com.

Mirrors the shape of ``song_facts_service.py`` / ``artist_facts_service.py``:
build a URL, fetch + parse, return structured data ready for
``MetadataDB.add_song_facts_batch`` / ``MetadataDB.update_track_producer_label``.

Unlike songfacts.com (a flat bullet list), Genius carries a song description
plus per-line annotations ("referents") explaining specific lyric fragments.
Line-annotation facts are pre-formatted as
``"Lyrics string: {line}. Fact: {explanation}"`` so that
``track_chat_service.resolve_song_facts_list`` (a plain
``SELECT fact FROM song_facts``) feeds them into both the "chat about song"
and "explain this line" prompts unchanged — no prompt-building code needs to
know these facts came from Genius.
"""

from __future__ import annotations

import logging
import re
import time
from html import unescape
from typing import List, Optional, Tuple, TypedDict

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from app.services.artist_split import normalize_artist_name, primary_artist
from app.services.proxy_config import get_proxy

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15


class GeniusParseError(Exception):
    """Raised when a Genius page can't be fetched or doesn't parse."""


# ---------------------------------------------------------------------------
# Genius URL slug
# ---------------------------------------------------------------------------

def _title_case_slug(text: str) -> str:
    """'Still D.R.E.' -> 'Still-dre'-style Genius title fragment.

    Genius slugs are Title-Case-With-Dashes: strip apostrophes/quotes/commas/
    '?'/'!' (same punctuation set as ``song_facts_service._slugify``), drop
    parenthetical suffixes, normalize dash variants, then join words with '-'.
    Only the FIRST word of the fragment is capitalized in practice on Genius
    (e.g. "Dr-dre", "Still-dre"), so we title-case just the leading word and
    lowercase the rest — matching real Genius URLs like
    ``genius.com/Dr-dre-still-dre-lyrics``.
    """
    cleaned = re.sub(r"[‐‑‒–—―−]", "-", text)
    cleaned = re.sub("['`,?!''‚‛""„‟′″ʼ«».]", "", cleaned)
    cleaned = re.sub(r"\s*\(.+?\)\s*", " ", cleaned)
    words = cleaned.split()
    return "-".join(w.lower() for w in words)


def build_genius_url(artist: str, title: str) -> str:
    """Build the Genius lyrics-page URL for a (primary artist, title) pair.

    Feat./collab artists are dropped (Genius pages are keyed by primary
    performer); ``&`` is replaced with ``and`` only in the artist portion.
    """
    artist_primary = primary_artist(normalize_artist_name(artist))
    artist_primary = artist_primary.replace("&", "and")
    artist_slug = _title_case_slug(artist_primary)
    title_slug = _title_case_slug(normalize_artist_name(title))
    # Genius capitalizes only the very first word of the whole slug.
    combined = f"{artist_slug}-{title_slug}"
    if combined:
        combined = combined[0].upper() + combined[1:]
    return f"https://genius.com/{combined}-lyrics"


# ---------------------------------------------------------------------------
# Low-level fetch (adapted from parse_genius.py, routed through get_proxy())
# ---------------------------------------------------------------------------

def _unescape_js_string(s: str) -> str:
    result = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n:
            nc = s[i + 1]
            if nc == '"':
                result.append('"'); i += 2; continue
            elif nc == "'":
                result.append("'"); i += 2; continue
            elif nc == '\\':
                result.append('\\'); i += 2; continue
            elif nc == 'n':
                result.append('\n'); i += 2; continue
            elif nc == 'r':
                result.append('\r'); i += 2; continue
            elif nc == 't':
                result.append('\t'); i += 2; continue
            elif nc == 'b':
                result.append('\b'); i += 2; continue
            elif nc == 'f':
                result.append('\f'); i += 2; continue
            elif nc == '/':
                result.append('/'); i += 2; continue
            elif nc == 'x' and i + 3 < n:
                try:
                    result.append(chr(int(s[i+2:i+4], 16)))
                    i += 4; continue
                except ValueError:
                    pass
            elif nc == 'u' and i + 5 < n:
                try:
                    result.append(chr(int(s[i+2:i+6], 16)))
                    i += 6; continue
                except ValueError:
                    pass
            result.append(nc); i += 2; continue
        else:
            result.append(c); i += 1
    return ''.join(result)


def _parse_preloaded_state(html: str) -> dict:
    import json as _json
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        if script.string and "__PRELOADED_STATE__" in script.string:
            match = re.search(
                r"window\.__PRELOADED_STATE__\s*=\s*JSON\.parse\('(.+?)'\);\s*\n",
                script.string,
                re.DOTALL,
            )
            if match:
                raw = _unescape_js_string(match.group(1))
                return _json.loads(raw)
    raise GeniusParseError("no __PRELOADED_STATE__ on page")


def _fetch_page(url: str) -> Tuple[str, dict]:
    proxies = get_proxy()
    kwargs = {"impersonate": "chrome99_android", "timeout": REQUEST_TIMEOUT}
    if proxies:
        kwargs["proxies"] = proxies
    resp = curl_requests.get(url, **kwargs)
    if resp.status_code == 404:
        raise GeniusParseError(f"404 at {url}")
    resp.raise_for_status()
    html = resp.text
    data = _parse_preloaded_state(html)
    return html, data


def _resolve_song_id(data: dict) -> str:
    song_page = data.get("songPage", {})
    song_ref = song_page.get("song")
    if isinstance(song_ref, dict):
        return str(song_ref.get("id", ""))
    if song_ref is not None:
        return str(song_ref)
    songs = data.get("entities", {}).get("songs", {})
    if songs:
        return next(iter(songs))
    return ""


def _extract_description(data: dict, song_id: str) -> Optional[str]:
    songs = data.get("entities", {}).get("songs", {})
    song = songs.get(song_id, {}) if song_id else {}
    desc = song.get("descriptionPreview") or data.get("songPage", {}).get("description")
    if not desc:
        return None
    return re.sub(r"\s+", " ", desc).strip() or None


def _extract_producers(data: dict, song_id: str) -> List[str]:
    entities = data.get("entities", {})
    artists = entities.get("artists", {})
    songs = entities.get("songs", {})
    song = songs.get(song_id, {}) if song_id else {}

    names: List[str] = []
    for aid in song.get("producerArtists", []):
        artist = artists.get(str(aid), {})
        name = artist.get("name")
        if name:
            names.append(name)

    if not names:
        producer_keywords = {"producer", "production"}
        for perf in song.get("customPerformances", []):
            label = perf.get("label", "")
            if any(k in label.lower() for k in producer_keywords):
                for aid in perf.get("artists", []):
                    name = artists.get(str(aid), {}).get("name")
                    if name:
                        names.append(name)

    seen = set()
    deduped = []
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def _extract_label(data: dict, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for label_el in soup.find_all(True, class_=lambda c: c and "Credit__Label" in str(c)):
        text = label_el.get_text(strip=True)
        if text == "Label":
            sibling = label_el.find_next_sibling()
            if not sibling:
                return None
            # Multiple labels render as adjacent <span> links joined by bare
            # "," / "&" text nodes with no surrounding whitespace — get_text
            # would otherwise glue them together (e.g. "Labelа,Labelb&Labelc").
            val = sibling.get_text(strip=True)
            val = re.sub(r"\s*,\s*", ", ", val)
            val = re.sub(r"\s*&\s*", " & ", val)
            return val.strip() or None
    return None


def _get_referent_links(data: dict) -> dict:
    html = data["songPage"]["lyricsData"]["body"]["html"]
    soup = BeautifulSoup(html, "html.parser")
    links = {}
    for a in soup.find_all("a", attrs={"data-id": True}):
        rid = a["data-id"]
        href = a["href"]
        full_url = "https://genius.com" + href if href.startswith("/") else href
        links[rid] = full_url
    return links


def _clean_annotation_text(text: str) -> str:
    """Strip markdown/HTML down to readable plain text for an LLM prompt."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_single_referent(url: str, referent_id: str) -> Optional[Tuple[str, str]]:
    """Returns (fragment_text, annotation_text) or None if unavailable."""
    import json as _json

    proxies = get_proxy()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    }
    try:
        resp = curl_requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT,
            proxies=proxies if proxies else None,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.debug("[Genius] referent fetch failed for %s: %s", url, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    def _meta(prop: str):
        tag = soup.find("meta", attrs={"property": prop})
        return tag.get("content") if tag else None

    fragment = _meta("rap_genius:referent")
    body = _meta("rap_genius:body")
    if fragment and body:
        return fragment, _clean_annotation_text(body)

    target_script = None
    for script in soup.find_all("script"):
        if script.string and "__PRELOADED_STATE__" in script.string:
            target_script = script.string
            break
    if not target_script:
        return None

    match = re.search(
        r"window\.__PRELOADED_STATE__\s*=\s*JSON\.parse\('(.+?)'\);\s*\n",
        target_script, re.DOTALL,
    )
    if not match:
        return None
    try:
        raw = _unescape_js_string(match.group(1))
        page_data = _json.loads(raw)
        entities = page_data.get("entities", {})
        referents_map = entities.get("referents", {})
        annotations_map = entities.get("annotations", {})
        referent_info = referents_map.get(referent_id)
        frag = fragment or (referent_info.get("fragment") if referent_info else None)
        ann_ids = referent_info.get("annotations", []) if referent_info else []
        if not frag or not ann_ids:
            return None
        ann = annotations_map.get(str(ann_ids[0]))
        if not ann:
            return None
        ann_text = ann.get("body", {}).get("markdown") or ann.get("body", {}).get("html")
        if not ann_text:
            return None
        return frag, _clean_annotation_text(ann_text)
    except (ValueError, KeyError):
        return None


def _fetch_all_annotations(data: dict, delay: float) -> List[Tuple[str, str]]:
    """Returns list of (fragment_text, annotation_text) pairs."""
    links = _get_referent_links(data)
    referent_ids = data["songPage"]["lyricsData"]["referents"]
    unique_ids = list(dict.fromkeys(str(r) for r in referent_ids))

    out: List[Tuple[str, str]] = []
    for rid in unique_ids:
        url = links.get(rid, f"https://genius.com/{rid}")
        result = _fetch_single_referent(url, rid)
        if result:
            out.append(result)
        if delay:
            time.sleep(delay)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GeniusSongData(TypedDict):
    url: str
    description: Optional[str]
    annotations: List[Tuple[str, str]]
    producers: List[str]
    label: Optional[str]


def fetch_song_data(url: str, delay: float = 0.3) -> GeniusSongData:
    """Fetch + parse one Genius song page. Raises GeniusParseError on failure."""
    html, data = _fetch_page(url)
    song_id = _resolve_song_id(data)
    if not song_id:
        raise GeniusParseError(f"could not resolve song id at {url}")

    description = _extract_description(data, song_id)
    producers = _extract_producers(data, song_id)
    label = _extract_label(data, html)
    try:
        annotations = _fetch_all_annotations(data, delay=delay)
    except (KeyError, GeniusParseError):
        annotations = []

    return {
        "url": url,
        "description": description,
        "annotations": annotations,
        "producers": producers,
        "label": label,
    }


def build_song_facts(parsed: GeniusSongData) -> List[Tuple[str, str]]:
    """Returns list of (fact_text, category) ready for add_song_facts_batch."""
    facts: List[Tuple[str, str]] = []
    if parsed["description"]:
        facts.append((parsed["description"], "genius_description"))
    for fragment, annotation in parsed["annotations"]:
        fact = f"Lyrics string: {fragment.strip()}. Fact: {annotation.strip()}"
        facts.append((fact, "genius_annotation"))
    return facts


def build_producer_label(parsed: GeniusSongData) -> Tuple[Optional[str], Optional[str]]:
    producer = ", ".join(parsed["producers"]) if parsed["producers"] else None
    return producer, parsed["label"]
