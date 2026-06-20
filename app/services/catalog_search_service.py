"""Catalog search engine — non-LLM, BM25F over title/album/artist, multi-entity.

Three corpora (songs / albums / artists) are built from the cached ``light_points``
projection. Each is scored with a field-weighted BM25F (so a title match isn't
diluted by lyrics — lyrics aren't indexed here at all), plus exact/prefix-of-name
bonuses and a small, capped history boost. Two output modes:

- **entity mode** (``search_entities``) — UI / ``GET /search/catalog``: the
  best-matching entity of any type wins (album-name query → the album, not its
  songs; artist-name query → the artist, not 50 songs).
- **tracks mode** (``search_tracks``) — the recsys ``library_search`` LLM tool:
  album/artist hits are *exploded* into their member tracks and deduped, so the
  LLM gets playable songs.

The pure functions (``build_index``/``search_entities``/``search_tracks``) take
fabricated points + history and are unit-testable without Qdrant. Thin wrappers at
the bottom source points from ``light_points`` and history from ``MetadataDB`` and
memoize the index per collection.

Spec: docs/superpowers/specs/2026-06-20-catalog-search-design.md
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

from .text_normalize import fold, tokenize, translit_variants

logger = logging.getLogger(__name__)


class CatalogConfig:
    """Centralized, tunable knobs (see spec §6/§7/§12)."""
    K1 = 1.2
    B = 0.75
    SAT = 4.0            # rel = bm25f / (bm25f + SAT) — cross-corpus comparable
    EXACT_BONUS = 0.6
    PREFIX_BONUS = 0.3
    HIST_ALPHA = 0.15
    R_LIKE = 0.10
    R_DISLIKE = 0.15
    HIST_CAP = 0.25      # EXACT(0.6) > PREFIX(0.3)+CAP(0.25) → history never lifts a
                         # partial match over an exact name match.
    MIN_QUERY_LEN = 2
    DEFAULT_LIMIT = 12
    TRACKS_LIMIT = 20
    FIELD_WEIGHTS = {
        "songs": {"title": 1.0, "artist": 0.15},
        "albums": {"album": 1.0, "artist": 0.20},
        "artists": {"artist": 1.0},
    }
    PRIMARY_FIELD = {"songs": "title", "albums": "album", "artists": "artist"}


# ── index structures ─────────────────────────────────────────────────────────
@dataclass
class _Doc:
    meta: dict                       # public hit fields (no score)
    fields: dict                     # fname -> list[set[str]] (variant set per token)
    tokenset: set                    # union of all field tokens (candidate check)
    primary_tokens: list             # list[set] of the primary field (exact/prefix)
    track_ids: list                  # member track ids
    members: list                    # member track recs (playable dicts)


@dataclass
class _Corpus:
    kind: str
    docs: list
    df: dict                         # term -> document frequency
    avglen: dict                     # fname -> average field length
    vocab: tuple                     # all terms (for prefix scan)
    N: int


@dataclass
class CatalogIndex:
    songs: _Corpus
    albums: _Corpus
    artists: _Corpus


# ── building ─────────────────────────────────────────────────────────────────
def _rec(track_id: str, payload: dict) -> dict:
    return {
        "track_id": track_id,
        "title": payload.get("title") or "",
        "artist": payload.get("artist") or "",
        "album": payload.get("album") or "",
        "artists": payload.get("artists") or ([payload.get("artist")] if payload.get("artist") else []),
        "artist_slugs": payload.get("artist_slugs") or [],
        "primary_artist_slug": payload.get("primary_artist_slug") or "",
        "year": payload.get("year"),
        "genre": payload.get("genre"),
        "duration": payload.get("duration"),
        "file_path": payload.get("file_path") or "",
        "cover_art_path": payload.get("cover_art_path"),
    }


def _make_doc(fieldvals: dict, primary_field: str, meta: dict, members: list) -> _Doc:
    fields = {
        fn: [translit_variants(tok) for tok in tokenize(val or "")]
        for fn, val in fieldvals.items()
    }
    tokenset: set = set()
    for toks in fields.values():
        for s in toks:
            tokenset |= s
    return _Doc(
        meta=meta,
        fields=fields,
        tokenset=tokenset,
        primary_tokens=fields.get(primary_field, []),
        track_ids=[m["track_id"] for m in members],
        members=members,
    )


def _build_corpus(kind: str, docs: list) -> _Corpus:
    df: dict = {}
    for d in docs:
        for t in d.tokenset:
            df[t] = df.get(t, 0) + 1
    avglen = {}
    for fn in CatalogConfig.FIELD_WEIGHTS[kind]:
        lens = [len(d.fields.get(fn) or []) for d in docs]
        avglen[fn] = (sum(lens) / len(lens)) if lens else 0.0
    return _Corpus(kind=kind, docs=docs, df=df, avglen=avglen, vocab=tuple(df.keys()), N=len(docs))


def build_index(points: list) -> CatalogIndex:
    """Build the three-corpus index from ``[(track_id, payload), ...]``."""
    recs = [_rec(tid, pl or {}) for tid, pl in points]

    # Songs — one doc per track; matched on title (+ low-weight artist).
    song_docs = []
    for r in recs:
        meta = {
            "type": "song", "track_id": r["track_id"], "title": r["title"],
            "artist": r["artist"], "album": r["album"],
            "cover_art_path": r["cover_art_path"],
        }
        song_docs.append(_make_doc(
            {"title": r["title"], "artist": " ".join(x for x in r["artists"] if x)},
            "title", meta, [r],
        ))

    # Albums — grouped by (folded album, primary artist slug).
    album_groups: dict = {}
    for r in recs:
        if not r["album"].strip():
            continue
        key = (fold(r["album"]), r["primary_artist_slug"])
        album_groups.setdefault(key, []).append(r)
    album_docs = []
    for (_, slug), members in album_groups.items():
        first = members[0]
        meta = {
            "type": "album", "album": first["album"], "artist": first["artist"],
            "artist_slug": slug, "cover_art_path": first["cover_art_path"],
            "track_count": len(members),
        }
        album_docs.append(_make_doc(
            {"album": first["album"], "artist": first["artist"]},
            "album", meta, members,
        ))

    # Artists — grouped by slug (every artist/slug pair on the track).
    artist_groups: dict = {}
    artist_names: dict = {}
    for r in recs:
        pairs = list(zip(r["artists"], r["artist_slugs"]))
        if not pairs and r["primary_artist_slug"]:
            pairs = [(r["artist"], r["primary_artist_slug"])]
        for name, slug in pairs:
            if not slug:
                continue
            artist_groups.setdefault(slug, []).append(r)
            artist_names.setdefault(slug, name or slug)
    artist_docs = []
    for slug, members in artist_groups.items():
        name = artist_names.get(slug, slug)
        meta = {
            "type": "artist", "artist": name, "artist_slug": slug,
            "track_count": len(members), "image": None, "cover_art_path": None,
        }
        artist_docs.append(_make_doc({"artist": name}, "artist", meta, members))

    return CatalogIndex(
        songs=_build_corpus("songs", song_docs),
        albums=_build_corpus("albums", album_docs),
        artists=_build_corpus("artists", artist_docs),
    )


# ── scoring ──────────────────────────────────────────────────────────────────
def _parse_query(query: str) -> list:
    return [v for v in (translit_variants(tok) for tok in tokenize(query)) if v]


def _idf(corpus: _Corpus, term: str) -> float:
    df_t = corpus.df.get(term, 0)
    return math.log(1 + (corpus.N - df_t + 0.5) / (df_t + 0.5))


def _query_terms(corpus: _Corpus, q: list) -> list:
    """Per query token → (match_set, idf). Last token is a prefix match."""
    out = []
    last = len(q) - 1
    for i, V in enumerate(q):
        if i == last:
            S = {t for t in corpus.vocab if any(t.startswith(v) for v in V)}
        else:
            S = {v for v in V if v in corpus.df}
        if not S:
            out.append((set(), 0.0))
            continue
        out.append((S, max(_idf(corpus, t) for t in S)))
    return out


def _bm25f(corpus: _Corpus, doc: _Doc, terms: list) -> float:
    weights = CatalogConfig.FIELD_WEIGHTS[corpus.kind]
    score = 0.0
    for S, idf in terms:
        if not S or idf <= 0:
            continue
        weighted_tf = 0.0
        for fn, w in weights.items():
            toks = doc.fields.get(fn)
            if not toks:
                continue
            cnt = sum(1 for ts in toks if not ts.isdisjoint(S))
            if not cnt:
                continue
            avgl = corpus.avglen.get(fn) or 1.0
            norm = 1 - CatalogConfig.B + CatalogConfig.B * (len(toks) / avgl)
            weighted_tf += w * cnt / norm
        if weighted_tf > 0:
            score += idf * weighted_tf / (CatalogConfig.K1 + weighted_tf)
    return score


def _name_bonus(q: list, primary: list) -> float:
    lq, lp = len(q), len(primary)
    if lp == 0 or lq == 0:
        return 0.0
    if lq == lp and all(q[i] & primary[i] for i in range(lq)):
        return CatalogConfig.EXACT_BONUS
    if lq <= lp:
        head_ok = all(q[i] & primary[i] for i in range(lq - 1))
        last_ok = any(pt.startswith(qt) for qt in q[lq - 1] for pt in primary[lq - 1])
        if head_ok and last_ok:
            return CatalogConfig.PREFIX_BONUS
    return 0.0


def _history_boost(track_ids: list, history: dict | None, max_plays: int) -> float:
    if not history:
        return 0.0
    pc = history.get("play_counts") or {}
    rc = history.get("reactions") or {}
    plays = sum(pc.get(t, 0) for t in track_ids)
    plays_norm = math.log1p(plays) / math.log1p(max_plays) if max_plays > 0 else 0.0
    net = 0
    for t in track_ids:
        r = rc.get(t)
        if r == "like":
            net += 1
        elif r == "dislike":
            net -= 1
    reaction = CatalogConfig.R_LIKE if net > 0 else (-CatalogConfig.R_DISLIKE if net < 0 else 0.0)
    raw = CatalogConfig.HIST_ALPHA * plays_norm + reaction
    return max(-CatalogConfig.HIST_CAP, min(CatalogConfig.HIST_CAP, raw))


def _score_corpus(corpus: _Corpus, q: list, history: dict | None, max_plays: int) -> list:
    terms = _query_terms(corpus, q)
    sall: set = set()
    for S, _ in terms:
        sall |= S
    if not sall:
        return []
    hits = []
    for doc in corpus.docs:
        if doc.tokenset.isdisjoint(sall):
            continue
        bm = _bm25f(corpus, doc, terms)
        if bm <= 0:
            continue
        rel = bm / (bm + CatalogConfig.SAT)
        score = rel + _name_bonus(q, doc.primary_tokens) + _history_boost(
            doc.track_ids, history, max_plays,
        )
        hits.append({**doc.meta, "score": score, "_members": doc.members})
    return hits


def _max_plays(history: dict | None) -> int:
    pc = (history or {}).get("play_counts") or {}
    return max(pc.values()) if pc else 0


def search_entities(index: CatalogIndex, query: str, *, history: dict | None = None,
                    limit: int = CatalogConfig.DEFAULT_LIMIT) -> list:
    """Entity mode: best-matching artist/album/song of any type ranked together."""
    q = _parse_query(query)
    if not q:
        return []
    mp = _max_plays(history)
    hits = []
    for corpus in (index.songs, index.albums, index.artists):
        hits.extend(_score_corpus(corpus, q, history, mp))
    hits.sort(key=lambda h: -h["score"])
    out = []
    for h in hits[:limit]:
        h = dict(h)
        h.pop("_members", None)
        out.append(h)
    return out


def search_tracks(index: CatalogIndex, query: str, *, history: dict | None = None,
                  limit: int = CatalogConfig.TRACKS_LIMIT) -> list:
    """Tracks mode: explode album/artist hits into their member tracks, dedup."""
    q = _parse_query(query)
    if not q:
        return []
    mp = _max_plays(history)
    scored = []
    for corpus in (index.songs, index.albums, index.artists):
        scored.extend(_score_corpus(corpus, q, history, mp))
    best: dict = {}
    for hit in scored:
        for rec in hit["_members"]:
            tid = rec["track_id"]
            if tid not in best or hit["score"] > best[tid][0]:
                best[tid] = (hit["score"], rec)
    pc = (history or {}).get("play_counts") or {}
    ordered = sorted(best.values(), key=lambda sr: (-sr[0], -pc.get(sr[1]["track_id"], 0)))
    keys = ("track_id", "title", "artist", "album", "year", "genre", "duration",
            "file_path", "cover_art_path")
    return [{k: rec.get(k) for k in keys} for _, rec in ordered[:limit]]


# ── Qdrant/MetadataDB-backed wrappers (integration-tested) ───────────────────
_INDEX_CACHE: dict = {}
_INDEX_LOCK = threading.Lock()


def _get_index(qdrant, collection_name: str) -> CatalogIndex | None:
    """Memoized per-collection index, rebuilt when the light_points count changes."""
    from app.resources.qdrant_utils import light_points
    points = light_points(qdrant, collection_name)
    count = len(points)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(collection_name)
        if cached is not None and cached[0] == count:
            return cached[1]
    index = build_index(points)
    with _INDEX_LOCK:
        _INDEX_CACHE[collection_name] = (count, index)
    return index


def invalidate(collection_name: str | None = None) -> None:
    with _INDEX_LOCK:
        if collection_name is None:
            _INDEX_CACHE.clear()
        else:
            _INDEX_CACHE.pop(collection_name, None)


def _load_history(collection_name: str) -> dict:
    from app.resources.metadata_db import MetadataDB
    try:
        play_counts = MetadataDB.get_play_counts_by_track(collection_name)
    except Exception:
        play_counts = {}
    try:
        reactions = {
            tid: reaction
            for tid, reaction, _ in MetadataDB.get_reactions_with_updated_at(collection_name)
        }
    except Exception:
        reactions = {}
    return {"play_counts": play_counts, "reactions": reactions}


def _fill_artist_images(hits: list, collection_name: str) -> None:
    from app.resources.metadata_db import MetadataDB
    for h in hits:
        if h.get("type") != "artist" or not h.get("artist_slug"):
            continue
        try:
            row = MetadataDB.get_artist_audiodb(h["artist_slug"], collection_name) or {}
            h["image"] = row.get("thumb_path") or row.get("cutout_path")
        except Exception:
            pass


def search_catalog(qdrant, collection_name: str, query: str,
                   limit: int = CatalogConfig.DEFAULT_LIMIT) -> list:
    """Entity-mode catalog search for one collection (UI endpoint)."""
    if not query or len(query.strip()) < CatalogConfig.MIN_QUERY_LEN:
        return []
    index = _get_index(qdrant, collection_name)
    if index is None:
        return []
    history = _load_history(collection_name)
    hits = search_entities(index, query, history=history, limit=limit)
    _fill_artist_images(hits, collection_name)
    return hits


def search_catalog_tracks(qdrant, collection_name: str, query: str,
                          limit: int = CatalogConfig.TRACKS_LIMIT) -> list:
    """Tracks-mode catalog search for one collection (recsys library_search tool)."""
    if not query or len(query.strip()) < CatalogConfig.MIN_QUERY_LEN:
        return []
    index = _get_index(qdrant, collection_name)
    if index is None:
        return []
    history = _load_history(collection_name)
    return search_tracks(index, query, history=history, limit=limit)
