"""Parser for Apple Music playlist / catalogue pages (music.apple.com).

A page carries THREE independent sources of data and not one of them is
sufficient on its own:

  1. JSON-LD  <script id="schema:music-playlist" type="application/ld+json">
     Always present, including in a raw HTTP response.
     Gives: name, author, url, numTracks, datePublished, and per track
     name/url/duration.
     Does NOT give: the performing artist, the album, or a real description
     (what is there is an SEO stub).

  2. serialized-server-data  <script id="serialized-server-data" type="application/json">
     SvelteKit's payload. Present in a raw response too, and the main source
     whenever the page has not been rendered by a browser.
     Gives: title, artistName/subtitleLinks, tertiaryLinks (the album),
     duration in ms, composer, showExplicitBadge, artwork, adam-id.

  3. DOM (the songs-list markup with data-testid)
     Only after hydration — a page saved from a browser, or headless. Gives the
     same as (2).

Merge priority: DOM > serialized > JSON-LD. The join key is the track's numeric
adam-id, so neither ordering nor disagreement between the sources matters.

Ported from the lab verbatim; only the prose is translated. Depends on lxml.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

from lxml import html as lhtml

__all__ = ["Playlist", "Track", "parse_apple_playlist"]

_SONG_ID = re.compile(r"/song/(?:[^/]*/)?(\d+)")
_SONG_ID_Q = re.compile(r"[?&]i=(\d+)")
_ARTIST_ID = re.compile(r"/artist/(?:[^/]*/)?(\d+)")
_ALBUM_ID = re.compile(r"/album/(?:[^/]*/)?(\d+)")
_PLAYLIST_ID = re.compile(r"/playlist/(?:[^/]*/)?(pl\.[0-9a-zA-Z\-]+)")
_ISO_DUR = re.compile(r"^P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")
# "Listen to the X playlist on Apple Music. 38 Songs. Duration: 2 hours 45 minutes."
_SEO_DESC = re.compile(r"^Listen to .{0,200}? on Apple Music\.\s*\d+\s+Songs?\.", re.I)


# --------------------------------------------------------------------------- #
#  Models
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    position: int | None = None
    song_id: str | None = None
    title: str | None = None
    artists: list[str] = field(default_factory=list)
    artist_ids: list[str] = field(default_factory=list)
    album: str | None = None
    album_id: str | None = None
    composer: str | None = None
    duration_iso: str | None = None
    duration_sec: int | None = None
    explicit: bool = False
    url: str | None = None
    artwork: str | None = None

    @property
    def artist(self) -> str | None:
        """Every performer on one line, the way Apple shows them."""
        return " & ".join(self.artists) or None

    def fill_from(self, other: "Track") -> None:
        """Fill empty fields from another source; ``self`` wins on conflicts."""
        for f in ("song_id", "title", "album", "album_id", "composer",
                  "duration_iso", "duration_sec", "url", "artwork"):
            if getattr(self, f) in (None, "") and getattr(other, f) not in (None, ""):
                setattr(self, f, getattr(other, f))
        if not self.artists and other.artists:
            self.artists = other.artists
        if not self.artist_ids and other.artist_ids:
            self.artist_ids = other.artist_ids
        self.explicit = self.explicit or other.explicit


@dataclass
class Playlist:
    playlist_id: str | None = None
    title: str | None = None
    author: str | None = None           # the curator: "Apple Music Alternative"
    author_url: str | None = None
    description: str | None = None      # the real editorial description
    seo_description: str | None = None  # what sits in JSON-LD / og:description
    url: str | None = None
    artwork: str | None = None
    date_published: str | None = None
    num_tracks: int | None = None
    sources: list[str] = field(default_factory=list)  # which sources produced data
    tracks: list[Track] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _txt(node) -> str | None:
    if node is None:
        return None
    s = " ".join(node.text_content().split())
    return s or None


def _first(seq, default=None):
    return seq[0] if seq else default


def _iso_to_sec(iso: str | None) -> int | None:
    if not iso:
        return None
    m = _ISO_DUR.match(iso.strip())
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def _sec_to_iso(sec: int | None) -> str | None:
    if sec is None:
        return None
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return "PT" + (f"{h}H" if h else "") + (f"{m}M" if m else "") + f"{s}S"


def _id_from(href: str | None, pattern: re.Pattern) -> str | None:
    if not href:
        return None
    m = pattern.search(href)
    return m.group(1) if m else None


def _song_id_from_url(url: str | None) -> str | None:
    """Apple uses two shapes: /song/<slug>/<id> and /album/<slug>/<aid>?i=<id>."""
    return _id_from(url, _SONG_ID) or _id_from(url, _SONG_ID_Q)


def _walk(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _dig(obj: Any, *keys, default=None):
    """Walk nested dicts without raising on a missing key."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _types(obj: dict) -> set[str]:
    t = obj.get("@type")
    if isinstance(t, list):
        return set(t)
    return {t} if t else set()


# --------------------------------------------------------------------------- #
#  Source 1: JSON-LD
# --------------------------------------------------------------------------- #

def _extract_jsonld(doc) -> dict | None:
    for script in doc.xpath('//script[@type="application/ld+json"]'):
        raw = script.text_content()
        if not raw or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in _walk(data):
            if "MusicPlaylist" in _types(obj):
                return obj
    return None


def _jsonld_tracks(ld: dict) -> list[Track]:
    out = []
    for lt in ld.get("track") or []:
        if not isinstance(lt, dict):
            continue
        iso = lt.get("duration")
        out.append(Track(
            song_id=_song_id_from_url(lt.get("url")),
            title=lt.get("name"),
            url=lt.get("url"),
            duration_iso=iso,
            duration_sec=_iso_to_sec(iso),
        ))
    return out


# --------------------------------------------------------------------------- #
#  Source 2: serialized-server-data
# --------------------------------------------------------------------------- #

def _extract_serialized(doc) -> list[Any]:
    """Every SvelteKit JSON payload on the page."""
    payloads = []
    scripts = doc.xpath('//script[@id="serialized-server-data"]') \
        or doc.xpath('//script[@type="application/json"]')
    for script in scripts:
        raw = script.text_content()
        if not raw or not raw.strip():
            continue
        try:
            payloads.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return payloads


def _is_song_lockup(obj: dict) -> bool:
    return (_dig(obj, "contentDescriptor", "kind") == "song"
            and isinstance(obj.get("title"), str))


def _links(obj: dict, key: str, kind: str) -> tuple[list[str], list[str]]:
    """(names, adam-ids) out of subtitleLinks / tertiaryLinks."""
    names, ids = [], []
    for link in obj.get(key) or []:
        if not isinstance(link, dict):
            continue
        cd = _dig(link, "segue", "destination", "contentDescriptor", default={})
        if kind and cd.get("kind") not in (kind, None):
            continue
        if link.get("title"):
            names.append(link["title"])
            aid = _dig(cd, "identifiers", "storeAdamID")
            if aid:
                ids.append(str(aid))
    return names, ids


def _serialized_tracks(payloads: list[Any]) -> list[Track]:
    out: list[Track] = []
    seen: set[str] = set()
    for payload in payloads:
        for obj in _walk(payload):
            if not _is_song_lockup(obj):
                continue
            sid = _dig(obj, "contentDescriptor", "identifiers", "storeAdamID")
            sid = str(sid) if sid else _song_id_from_url(
                _dig(obj, "contentDescriptor", "url"))
            if sid:
                if sid in seen:
                    continue
                seen.add(sid)

            artists, artist_ids = _links(obj, "subtitleLinks", "artist")
            if not artists and obj.get("artistName"):
                artists = [obj["artistName"]]
            albums, album_ids = _links(obj, "tertiaryLinks", "album")

            ms = obj.get("duration")
            sec = round(ms / 1000) if isinstance(ms, (int, float)) else None

            art = _dig(obj, "artwork", "dictionary", "url")
            if isinstance(art, str):
                art = art.replace("{w}", "600").replace("{h}", "600").replace("{f}", "jpg")

            out.append(Track(
                song_id=sid,
                title=obj.get("title"),
                artists=artists,
                artist_ids=artist_ids,
                album=albums[0] if albums else None,
                album_id=album_ids[0] if album_ids else None,
                composer=obj.get("composer") or None,
                duration_sec=sec,
                duration_iso=_sec_to_iso(sec),
                explicit=bool(obj.get("showExplicitBadge")),
                url=_dig(obj, "contentDescriptor", "url"),
                artwork=art,
            ))
    return out


def _serialized_header(payloads: list[Any]) -> dict[str, Any]:
    """
    The playlist header inside the payload. Apple's shape drifts between
    releases, so this searches heuristically: the longest meaningful
    description that is not the SEO stub.
    """
    out: dict[str, Any] = {}
    best_desc = None
    for payload in payloads:
        for obj in _walk(payload):
            for key in ("description", "editorialNotes", "modalDescription"):
                val = obj.get(key)
                if isinstance(val, dict):
                    val = val.get("standard") or val.get("short") or val.get("content")
                if not isinstance(val, str):
                    continue
                val = " ".join(val.split())
                if len(val) < 40 or _SEO_DESC.match(val):
                    continue
                if best_desc is None or len(val) > len(best_desc):
                    best_desc = val
            if _dig(obj, "contentDescriptor", "kind") == "playlist":
                if obj.get("title") and "title" not in out:
                    out["title"] = obj["title"]
                names, _ids = _links(obj, "subtitleLinks", "")
                if names and "author" not in out:
                    out["author"] = names[0]
    if best_desc:
        out["description"] = best_desc
    return out


# --------------------------------------------------------------------------- #
#  Source 3: DOM
# --------------------------------------------------------------------------- #

def _dom_header(doc) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out["title"] = _txt(_first(doc.xpath(
        '//h1[@data-testid="non-editable-product-title"]'
        ' | //h1[contains(@class,"headings__title")]'
        ' | //*[@data-testid="product-title"]')))

    sub = _first(doc.xpath(
        '//*[@data-testid="product-subtitles"]'
        ' | //div[contains(@class,"headings__subtitles")]'))
    if sub is not None:
        link = _first(sub.xpath('.//a[@href]'))
        out["author"] = _txt(link) or _txt(sub)
        out["author_url"] = link.get("href") if link is not None else None

    # The truncation there is pure CSS — the DOM holds the full text
    out["description"] = _txt(_first(doc.xpath(
        '//*[@data-testid="description"]//*[@data-testid="truncate-text"]'
        ' | //*[@data-testid="description"]//p'
        ' | //*[@data-testid="truncate-text"]')))

    src = _first(doc.xpath(
        '//*[@data-testid="artwork-component"]//source[@srcset]/@srcset'))
    if src:
        out["artwork"] = src.split(",")[-1].strip().split(" ")[0]

    out["url"] = _first(doc.xpath('//link[@rel="canonical"]/@href')) \
        or _first(doc.xpath('//meta[@property="og:url"]/@content'))
    return {k: v for k, v in out.items() if v}


def _dom_tracks(doc) -> list[Track]:
    rows = doc.xpath(
        '//*[@data-testid="track-list-item"]'
        ' | //*[contains(concat(" ",normalize-space(@class)," ")," songs-list-row ")]')
    out: list[Track] = []

    for i, row in enumerate(rows):
        song_href = _first(row.xpath('.//a[contains(@href,"/song/")]/@href'))

        artist_links = row.xpath('.//*[@data-testid="track-column-secondary"]//a[@href]') \
            or row.xpath('.//*[@data-testid="track-title-by-line"]//a[@href]')
        artists = [_txt(a) for a in artist_links if _txt(a)]
        artist_ids = [aid for aid in
                      (_id_from(a.get("href"), _ARTIST_ID) for a in artist_links) if aid]
        if not artists:
            plain = _txt(_first(row.xpath('.//*[@data-testid="track-column-secondary"]'))) \
                or _txt(_first(row.xpath('.//*[@data-testid="track-title-by-line"]')))
            artists = [plain] if plain else []

        album_link = _first(row.xpath('.//*[@data-testid="track-column-tertiary"]//a[@href]'))
        album = _txt(album_link) or _txt(
            _first(row.xpath('.//*[@data-testid="track-column-tertiary"]')))
        album_id = _id_from(album_link.get("href"), _ALBUM_ID) if album_link is not None else None

        dur_node = _first(row.xpath('.//time[@datetime]'))
        dur_iso = dur_node.get("datetime") if dur_node is not None else None

        row_pos = row.get("data-row")
        out.append(Track(
            position=int(row_pos) + 1 if row_pos and row_pos.isdigit() else i + 1,
            song_id=_song_id_from_url(song_href),
            title=_txt(_first(row.xpath('.//*[@data-testid="track-title"]'))),
            artists=artists,
            artist_ids=artist_ids,
            album=album,
            album_id=album_id,
            duration_iso=dur_iso,
            duration_sec=_iso_to_sec(dur_iso),
            explicit=bool(row.xpath('.//*[@data-testid="explicit-badge"]')),
            url=song_href,
        ))
    return out


# --------------------------------------------------------------------------- #
#  Merging
# --------------------------------------------------------------------------- #

def _merge(*sources: list[Track]) -> list[Track]:
    """
    ``sources`` in descending priority. Ordering comes from the first
    non-empty source; missing fields are filled from the rest by song_id.
    """
    src = [s for s in sources if s]
    if not src:
        return []

    indexes = [{t.song_id: t for t in s if t.song_id} for s in src]
    base = src[0]

    for i, t in enumerate(base, start=1):
        if t.position is None:
            t.position = i
        if not t.song_id:
            continue
        for idx in indexes[1:]:
            other = idx.get(t.song_id)
            if other is not None:
                t.fill_from(other)

    # tracks absent from the base source (e.g. lazy-loaded on scroll)
    known = {t.song_id for t in base if t.song_id}
    extra = []
    for s in src[1:]:
        for t in s:
            if t.song_id and t.song_id not in known:
                known.add(t.song_id)
                extra.append(t)
    for i, t in enumerate(extra, start=len(base) + 1):
        t.position = t.position or i
    return base + extra


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #

def parse_apple_playlist(html: str | bytes, url: str | None = None) -> Playlist:
    """Parse the HTML of an Apple Music playlist or catalogue page.

    :param html: raw HTML (str or bytes) — from an HTTP fetch or saved from a browser
    :param url:  the page URL, a fallback source for the id and the canonical link
    """
    doc = lhtml.fromstring(html)

    ld = _extract_jsonld(doc) or {}
    payloads = _extract_serialized(doc)

    dom_head, dom_tr = _dom_header(doc), _dom_tracks(doc)
    ser_head, ser_tr = _serialized_header(payloads), _serialized_tracks(payloads)
    ld_tr = _jsonld_tracks(ld)

    sources = [n for n, ok in
               (("dom", dom_tr), ("serialized", ser_tr), ("json-ld", ld_tr)) if ok]

    # --- header ----------------------------------------------------------
    author = dom_head.get("author") or ser_head.get("author")
    if not author:
        a = ld.get("author")
        if isinstance(a, list):
            a = a[0] if a else None
        author = a.get("name") if isinstance(a, dict) else (a if isinstance(a, str) else None)

    seo_desc = ld.get("description") \
        or _first(doc.xpath('//meta[@property="og:description"]/@content'))

    description = dom_head.get("description") or ser_head.get("description")
    if not description and seo_desc and not _SEO_DESC.match(seo_desc):
        description = seo_desc

    page_url = dom_head.get("url") or ld.get("url") or url

    # JSON-LD carries the canonical track ORDER, so when it is present it
    # becomes the base and DOM/serialized only fill in the missing fields.
    tracks = _merge(ld_tr, dom_tr, ser_tr) if ld_tr else _merge(dom_tr, ser_tr)

    pl = Playlist(
        playlist_id=_id_from(page_url, _PLAYLIST_ID) or _id_from(url, _PLAYLIST_ID),
        title=dom_head.get("title") or ld.get("name") or ser_head.get("title")
              or _first(doc.xpath('//meta[@property="og:title"]/@content')),
        author=author,
        author_url=dom_head.get("author_url"),
        description=description,
        seo_description=seo_desc,
        url=page_url,
        artwork=dom_head.get("artwork")
                or _first(doc.xpath('//meta[@property="og:image"]/@content')),
        date_published=ld.get("datePublished"),
        num_tracks=ld.get("numTracks"),
        sources=sources,
        tracks=tracks,
    )
    if pl.num_tracks is None:
        pl.num_tracks = len(tracks)
    return pl


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as f:
        pl = parse_apple_playlist(f.read())
    print(json.dumps(pl.to_dict(), ensure_ascii=False, indent=2))