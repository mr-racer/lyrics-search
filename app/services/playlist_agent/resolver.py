"""Resolve LLM-proposed song titles against the library catalog.

Per item, two-stage matching:
  1. **exact** — normalized (``text_normalize.fold``), feat-stripped title
     equality, with an optional artist constraint;
  2. **fuzzy** — first of the top-3 ``catalog_search_service.search_catalog_tracks``
     hits (token + transliteration + IDF scoring) that passes the strict
     similarity gate (``_FUZZY_TITLE_MIN``) when exact misses. The BM25F query
     is the title alone — the artist is checked afterwards, never mixed into
     the query (it would flood the ranking with the artist's discography).

Every result carries its ``match`` mode ("exact" | "fuzzy" | "none") so the
playlist agent can tell the LLM whether a title matched cleanly, only
approximately, or not at all — the LLM decides whether to trust a fuzzy hit.

``resolve_songs`` is decoupled from the real catalog via a small duck-typed
protocol so it can be unit-tested with a fake; ``CatalogAdapter`` binds it to
the real ``catalog_search_service`` for one (qdrant, collection).
"""
import re

from app.services.name_match import score_names
from app.services.text_normalize import fold

# Fuzzy hits must actually resemble the asked-for song: minimum
# transliteration-aware similarity between the query title and the candidate
# title (and, when the artist is known, between the artists too). Below this
# a BM25F hit is treated as noise ("none"), not a match.
_FUZZY_TITLE_MIN = 0.6


def _norm(s):
    return fold(s or "")


# Library tags and web tracklists disagree on where the feature lives:
# "Hurricane (ft. Lil Baby)" vs "Hurricane". Title comparison cuts the tail
# starting at the first feat marker so both sides collapse to the same key.
_FEAT_TOKENS = {"feat", "ft", "featuring"}


def _title_key(s):
    toks = fold(s or "").split()
    for i, t in enumerate(toks):
        if t in _FEAT_TOKENS:
            return " ".join(toks[:i]) or " ".join(toks)
    return " ".join(toks)


def _artist_matches(query_artist, song_artist):
    na, ns = _norm(query_artist), _norm(song_artist)
    return bool(na) and bool(ns) and (na in ns or ns in na)


def _similar(query, value):
    if not query or not value:
        return 0.0
    return score_names(query, [value])[0][0]


def _fuzzy_acceptable(qtitle, qartist, hit):
    """Strict gate for a BM25F candidate: title similarity above the floor,
    and — when the query names an artist — the hit's artist must match by
    substring or by the same similarity floor."""
    if _similar(qtitle, hit.get("title")) < _FUZZY_TITLE_MIN:
        return False
    if qartist:
        return (_artist_matches(qartist, hit.get("artist"))
                or _similar(qartist, hit.get("artist")) >= _FUZZY_TITLE_MIN)
    return True


def resolve_songs(items, catalog, artist_filter=None):
    """Resolve ``items`` against ``catalog``.

    Parameters
    ----------
    items         : list of ``{"title": str, "artist": str | None}``.
    catalog       : object exposing
                    ``iter_songs() -> iterable[{"track_id","title","artist"}]``
                    and ``search_tracks_fuzzy(query: str, limit: int) -> list``
                    of track dicts (``track_id``/``title``/``artist``).
    artist_filter : optional fallback artist applied to items whose own
                    ``artist`` is missing (e.g. an "artist hits" playlist).

    Returns a list aligned with ``items``; each element is
    ``{"query_title", "match", "track_id", "title", "artist"}`` where ``match``
    is "exact", "fuzzy", or "none".
    """
    by_title = {}
    for s in catalog.iter_songs():
        by_title.setdefault(_title_key(s.get("title")), []).append(s)

    results = []
    for item in items:
        qtitle = (item.get("title") or "").strip()
        qartist = item.get("artist") or artist_filter
        picked = None
        mode = "none"

        # exact — normalized, feat-stripped title equality (+ optional artist constraint)
        candidates = by_title.get(_title_key(qtitle), [])
        if candidates:
            if qartist:
                picked = next(
                    (c for c in candidates if _artist_matches(qartist, c.get("artist"))),
                    None,
                )
            else:
                picked = candidates[0]
            if picked is not None:
                mode = "exact"

        # fuzzy fallback — first of the top-3 that survives the strict gate.
        # The query is the feat-stripped title ALONE: with the artist appended,
        # tracks-mode BM25F explodes the artist hit over the whole discography
        # and the most-played tracks push the real title out of the top-3.
        # The artist constraint is enforced after the fact by _fuzzy_acceptable.
        if picked is None and qtitle:
            query = _title_key(qtitle)
            if len(query) < 2 and qartist:
                # single-char titles ("7") die on the search's minimum query
                # length — the artist is the only usable signal left
                query = f"{query} {qartist}".strip()
            hits = catalog.search_tracks_fuzzy(query, limit=3)
            picked = next((h for h in hits if _fuzzy_acceptable(qtitle, qartist, h)), None)
            if picked is not None:
                mode = "fuzzy"

        if picked is None:
            results.append({"query_title": qtitle, "match": "none",
                            "track_id": None, "title": None, "artist": None})
        else:
            results.append({
                "query_title": qtitle, "match": mode,
                "track_id": picked.get("track_id"),
                "title": picked.get("title"),
                "artist": picked.get("artist"),
            })
    return results


# ─── Tracklist auto-matching ─────────────────────────────────────────────────
# A 500-song soundtrack cannot be "copy-typed" through a small LLM: the model
# reads a sampled excerpt, proposes a couple dozen titles, and the library
# intersection comes out nearly empty (observed: GTA V — 0-1 matches per pass).
# Instead the FULL extracted tracklist is intersected with the library in code;
# the model only curates the verified result.

# "Artist — Title", "Artist - Title", "Title | Artist", with optional leading
# numbering and trailing "(2014)" year.
_LINE_SPLIT_RE = re.compile(r"\s(?:—|–|\||-)\s")
_LINE_NUM_RE = re.compile(r"^\d{1,3}[.)]\s+")
_LINE_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_QUOTES = "\"'«»“”‘’„"


def parse_track_line(line):
    """Parse one track-like line into ``{"artist", "title"}`` or ``None``.

    The left/right orientation is NOT decided here — sites disagree
    ("Artist — Title" vs "Title - Artist"); ``resolve_tracklines`` tries both.
    By convention the left side is returned as "artist".
    """
    if not line:
        return None
    s = _LINE_NUM_RE.sub("", line.strip())
    s = _LINE_YEAR_RE.sub("", s)
    parts = _LINE_SPLIT_RE.split(s, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (p.strip().strip(_QUOTES).strip() for p in parts)
    if not left or not right:
        return None
    # Both sides must look like names, not sentence fragments.
    if len(left) > 80 or len(right) > 120:
        return None
    return {"artist": left, "title": right}


def extract_year_range(wish):
    """Явный годовой констрейнт из текста желания → (min_year, max_year) | None.

    Общие формы: «после/after/since 2020», «до/before 2000», «2010-2015»,
    декады «00-х», «90s», «'80s», «нулевых». Голый год без предлога намеренно
    игнорируется («gta 5», названия с числами). Фильтр применяется кодом к
    авто-матчам: модель, умирая на финальном выводе, не успевает отфильтровать
    эпоху сама, и фоллбэк без этого уносит в плейлист всю карьеру артиста."""
    if not wish:
        return None
    low = wish.lower()
    m = re.search(r"(?:после|after|since|начиная с)\s+((?:19|20)\d{2})", low)
    if m:
        return (int(m.group(1)) + (0 if "начиная" in m.group(0) or "since" in m.group(0) else 1), 3000)
    m = re.search(r"(?:до|before)\s+((?:19|20)\d{2})", low)
    if m:
        return (1900, int(m.group(1)) - 1)
    m = re.search(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\b", low)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:^|\s)['’]?(\d0)(?:-?х|s|х)(?:\s|$|[,.!?])", low)
    if m:
        dec = int(m.group(1))
        base = 1900 + dec if dec >= 30 else 2000 + dec
        return (base, base + 9)
    if re.search(r"нулев", low):
        return (2000, 2009)
    return None


def _fuzzy_acceptable_strict(qtitle, qartist, hit):
    """Гейт для МАССОВОГО авто-пересечения (сотни строк — ложняки стреляют
    чаще, а модель их не ревьюит). Отличие легитимного нечёткого хита от
    ложного — направление вложенности: запрос содержится в названии кандидата
    («Swimming Pools (Drank)» ⊂ «…[Extended Version]»), а у ложняка кандидат
    короче запроса («Dangerous» при запросе «Dangerous Tonight»)."""
    qk, hk = _title_key(qtitle), _title_key(hit.get("title"))
    title_ok = (bool(qk) and bool(hk) and qk in hk) \
        or _similar(qtitle, hit.get("title")) >= 0.75
    if not title_ok:
        return False
    if qartist:
        return (_artist_matches(qartist, hit.get("artist"))
                or _similar(qartist, hit.get("artist")) >= 0.75)
    return True


def resolve_tracklines(lines, catalog, max_fuzzy=120):
    """Intersect extracted tracklist ``lines`` with the library, deterministically.

    Exact matching (both orientations) over every parsed line; the fuzzy
    BM25F fallback is capped at ``max_fuzzy`` misses to bound latency on
    huge soundtracks. Returns ONLY the matches:
    ``[{"track_id", "title", "artist", "match"}]``, input order, deduped.
    """
    parsed = []
    seen = set()
    for ln in lines:
        p = parse_track_line(ln)
        if p is None:
            continue
        key = (fold(p["artist"]), fold(p["title"]))
        if key in seen or (key[1], key[0]) in seen:
            continue
        seen.add(key)
        # Год из исходной строки (год ОРИГИНАЛЬНОГО релиза со страницы, не из
        # тега библиотеки) — модель фильтрует эпоху по компактному списку
        # матчей, не видя самой страницы.
        ym = re.search(r"\((\d{4})\)", ln)
        p["year"] = ym.group(1) if ym else None
        parsed.append(p)
    if not parsed:
        return []

    by_title = {}
    for s in catalog.iter_songs():
        by_title.setdefault(_title_key(s.get("title")), []).append(s)

    out, out_ids, fuzzy_left = [], set(), max_fuzzy
    for p in parsed:
        picked = None
        mode = "none"
        for artist, title in ((p["artist"], p["title"]), (p["title"], p["artist"])):
            candidates = by_title.get(_title_key(title), [])
            picked = next(
                (c for c in candidates if _artist_matches(artist, c.get("artist"))),
                None,
            )
            if picked is not None:
                mode = "exact"
                break
        if picked is None and fuzzy_left > 0:
            fuzzy_left -= 1
            hits = catalog.search_tracks_fuzzy(_title_key(p["title"]), limit=3)
            picked = next(
                (h for h in hits
                 if _fuzzy_acceptable_strict(p["title"], p["artist"], h)),
                None,
            )
            if picked is not None:
                mode = "fuzzy"
        if picked is None:
            continue
        tid = picked.get("track_id")
        if tid in out_ids:
            continue
        out_ids.add(tid)
        out.append({"track_id": tid, "title": picked.get("title"),
                    "artist": picked.get("artist"), "match": mode,
                    "year": p.get("year")})
    return out


class CatalogAdapter:
    """Bind ``resolve_songs``'s duck-typed protocol to the real
    ``catalog_search_service`` for one ``(qdrant, collection_name)``."""

    def __init__(self, qdrant, collection_name):
        self._qdrant = qdrant
        self._collection = collection_name

    def iter_songs(self):
        from app.services import catalog_search_service as css

        index = css._get_index(self._qdrant, self._collection)
        if index is None:
            return []
        return [
            {"track_id": d.meta.get("track_id"),
             "title": d.meta.get("title"),
             "artist": d.meta.get("artist", "")}
            for d in index.songs.docs
        ]

    def search_tracks_fuzzy(self, query, limit=3):
        from app.services import catalog_search_service as css

        return css.search_catalog_tracks(self._qdrant, self._collection, query, limit=limit)
