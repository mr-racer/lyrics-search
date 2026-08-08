"""Retrieval over raw facts: dense from Qdrant, BM25 and fusion in Python.

Replaces the token-overlap filter this module's caller used to run
(``facts_executor._fact_evidence``), which compared a *Russian* refined
statement against *English* source facts and required two shared content
tokens. Cross-script overlap only ever fires on Latin proper names, so the
material that actually explained a fact was thrown away before the model saw it.

Three parts:

**Dense** — ``facts_index`` searches this account's ``facts_acct_{id}``
collection and returns ids. Qwen3-Embedding is multilingual, so a Russian query
matches English facts directly; this is where the cross-language work happens.

**BM25** — a plain single-field BM25 over the retrieved candidates, computed
here. It contributes precision on names, numbers and titles, which dense
similarity blurs. It is NOT Qdrant's server-side BM25: that would need the fact
text stored in the collection, and keeping text out of Qdrant is the point.
``catalog_search_service._bm25f`` is not reused either — it is BM25**F**, bound
to the catalog's field weights and corpus shape.

**Pseudo-relevance feedback** — a Russian query shares almost no tokens with an
English document, so BM25 alone would score everything zero. Rather than paying
an LLM call to translate, the query is expanded with the strongest terms from
the top dense hits, which are already in the documents' own language. Standard
PRF, and free.

Cross-account isolation is structural — the vector index is partitioned per
account, so this module cannot reach another account's facts even by mistake.
The ``fact_visibility`` check that remains is about STALENESS, not leakage: a
track the user deleted leaves its vectors behind in their own facts collection,
and a fact about a song they no longer own should not turn up as a neighbour.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from app.services.text_normalize import tokenize, translit_variants

logger = logging.getLogger(__name__)

# How many neighbours to pull from the whole pool alongside the subject's own
# facts. Deliberately small: the failure this module exists to fix was "too
# much unrelated material", and a neighbour has to out-rank the subject's own
# facts to be worth a slot.
SUBJECT_LIMIT = 24
GLOBAL_LIMIT = 24

# BM25. Textbook defaults; the corpus here is tens of short documents, so there
# is nothing to tune them against yet.
K1 = 1.2
B = 0.75

# Pseudo-relevance feedback: how many top dense hits to mine, how many terms to
# lift, and how much a lifted term counts against a term the user actually
# typed. Kept conservative — PRF drifts off-topic when it is greedy.
PRF_DOCS = 3
PRF_TERMS = 8
PRF_WEIGHT = 0.45

# Reciprocal rank fusion. 60 is the constant from the original RRF paper and
# what Qdrant itself uses for the track search, so the two behave alike.
RRF_K = 60

_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "у", "о", "об", "от", "до", "за", "по", "из",
    "что", "это", "эта", "этот", "как", "для", "же", "а", "но", "не", "то", "тот",
    "был", "была", "было", "были", "есть", "его", "её", "их", "там", "тут",
    "песня", "песни", "песне", "трек", "треке", "трека", "альбом", "альбома",
    "группа", "группы", "артист", "артиста", "исполнитель",
    "the", "a", "an", "of", "in", "on", "at", "by", "for", "to", "and", "or",
    "is", "was", "were", "are", "with", "from", "this", "that", "it", "its", "as",
    "song", "songs", "track", "tracks", "album", "band", "artist", "he", "she",
    "they", "his", "her", "their", "who", "which", "when", "what", "about",
}


def content_tokens(text: str) -> list[str]:
    """Topic-bearing tokens: folded, de-noised, short and common words dropped."""
    return [t for t in tokenize(text) if len(t) > 2 and t not in _STOPWORDS]


# ── BM25 ─────────────────────────────────────────────────────────────────────


class _Bm25:
    """Single-field BM25 over a small in-memory corpus of fact texts."""

    def __init__(self, docs: list[list[str]]):
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avglen = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.df = df

    def idf(self, term: str) -> float:
        return math.log(1 + (self.n - self.df.get(term, 0) + 0.5)
                        / (self.df.get(term, 0) + 0.5))

    def score(self, weighted_terms: list[tuple[str, float]]) -> list[float]:
        """Score every document against ``[(term, weight), ...]``.

        Each query term is matched through its transliteration variants, so a
        name typed in Cyrillic still finds the Latin spelling in an English
        source ("Дайана" → "diana").
        """
        out = [0.0] * self.n
        for term, weight in weighted_terms:
            variants = translit_variants(term) or {term}
            present = [v for v in variants if v in self.df]
            if not present:
                continue
            # One idf for the term, taken from its best-attested spelling, so a
            # rare variant does not out-weigh the word itself.
            idf = max(self.idf(v) for v in present)
            if idf <= 0:
                continue
            for i in range(self.n):
                freq = sum(self.tf[i].get(v, 0) for v in present)
                if not freq:
                    continue
                norm = 1 - B + B * (self.lengths[i] / (self.avglen or 1.0))
                out[i] += weight * idf * freq * (K1 + 1) / (freq + K1 * norm)
        return out


def _prf_terms(bm25: _Bm25, seed_indices: list[int],
               already: set[str] | None = None) -> list[tuple[str, float]]:
    """Terms lifted from the top dense hits, weighted below the user's own.

    ``already`` holds the terms the user actually typed; re-adding one would
    quietly score it at 1.45 instead of 1.0.
    """
    already = already or set()
    scored: dict[str, float] = {}
    for i in seed_indices[:PRF_DOCS]:
        if not (0 <= i < bm25.n):
            continue
        for term, freq in bm25.tf[i].items():
            if term in already:
                continue
            idf = bm25.idf(term)
            if idf <= 0:
                continue
            scored[term] = max(scored.get(term, 0.0), freq * idf)
    top = sorted(scored.items(), key=lambda kv: -kv[1])[:PRF_TERMS]
    return [(term, PRF_WEIGHT) for term, _ in top]


def _rrf(*rankings: list[int]) -> dict[int, float]:
    """Reciprocal rank fusion over candidate indices."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


# ── the public entry point ───────────────────────────────────────────────────


def _join_texts(hits: list[dict]) -> list[dict]:
    """Attach fact text from SQLite. Hits whose row vanished are dropped."""
    from app.resources.metadata_db import MetadataDB

    by_kind: dict[str, list[int]] = {}
    for h in hits:
        by_kind.setdefault(h["kind"], []).append(h["row_id"])

    rows: dict[tuple[str, int], dict] = {}
    for kind, ids in by_kind.items():
        try:
            for row_id, row in MetadataDB.get_facts_by_ids(kind, ids).items():
                rows[(kind, row_id)] = row
        except Exception:
            logger.warning("[facts_retrieval] fact join failed for %s", kind,
                           exc_info=True)

    out = []
    for h in hits:
        row = rows.get((h["kind"], h["row_id"]))
        if not row or not (row.get("fact") or "").strip():
            continue
        out.append({**h, "text": row["fact"], "source": row.get("source") or "",
                    "category": row.get("category") or "",
                    "slug": row.get("slug") or h.get("slug") or ""})
    return out


def _drop_invisible(hits: list[dict], collection_name: str,
                    always_allow: set[str]) -> list[dict]:
    """Drop hits for entities this account no longer owns.

    Not a cross-account gate — the index is per-account, so nothing from another
    account can be here. This catches the account's OWN stale vectors: deleting
    a track removes its ``fact_visibility`` row but leaves the embeddings, and a
    neighbour fact about a song the user deleted has no business in an answer.
    ``always_allow`` is the subject itself, which needs no lookup.
    """
    from app.resources.metadata_db import MetadataDB

    by_kind: dict[str, set[str]] = {}
    for h in hits:
        if h["slug"] in always_allow:
            continue
        by_kind.setdefault(h["kind"], set()).add(h["slug"])

    visible: dict[str, set[str]] = {}
    for kind, slugs in by_kind.items():
        try:
            visible[kind] = MetadataDB.filter_visible_slugs(
                kind, sorted(slugs), collection_name)
        except Exception:
            # Fail CLOSED: an unreadable visibility table must not turn into
            # "surface everything, including what was deleted".
            logger.warning("[facts_retrieval] visibility check failed", exc_info=True)
            visible[kind] = set()

    return [h for h in hits
            if h["slug"] in always_allow or h["slug"] in visible.get(h["kind"], set())]


def retrieve(qdrant, *, collection_name: str, query: str,
             subject_slugs: dict[str, str], limit: int = 12) -> list[dict]:
    """Facts most likely to explain ``query``, best first.

    ``collection_name`` is the caller's own account collection (``acct_{id}``)
    — the facts collection is derived from it, never passed in.

    ``subject_slugs`` maps kind → slug for what the question is about, e.g.
    ``{"song": "queen-bohemian-rhapsody", "artist": "queen"}``. Those entities
    are indexed on the spot if needed and searched with priority; the rest of
    this account's pool is searched too, because the explanation for a sample or
    a cover routinely lives in the *other* song's facts.

    Returns ``[{"text", "source", "category", "kind", "row_id", "slug",
    "dense_score", "scope"}]``. Never raises — an unreachable Qdrant or a cold
    index degrades to fewer results, or none.
    """
    from app.resources.model_registry import ModelRegistry
    from app.services import facts_index

    query = (query or "").strip()
    if not query:
        return []

    own = {slug for slug in subject_slugs.values() if slug}
    for kind, slug in subject_slugs.items():
        if slug:
            facts_index.index_entity(qdrant, collection_name, kind, slug)

    try:
        query_vector = ModelRegistry.encode_text(query, is_query=True)
    except Exception:
        logger.warning("[facts_retrieval] query encode failed", exc_info=True)
        return []

    subject_hits = (facts_index.search(qdrant, collection_name, query_vector,
                                       limit=SUBJECT_LIMIT, slugs=sorted(own))
                    if own else [])
    global_hits = facts_index.search(qdrant, collection_name, query_vector,
                                     limit=GLOBAL_LIMIT)

    seen: set[tuple[str, int]] = set()
    merged: list[dict] = []
    for hit, scope in ([(h, "subject") for h in subject_hits]
                       + [(h, "neighbour") for h in global_hits]):
        key = (hit["kind"], hit["row_id"])
        if key in seen:
            continue
        seen.add(key)
        merged.append({**hit, "scope": scope})

    merged = _join_texts(merged)
    merged = _drop_invisible(merged, collection_name, own)
    if not merged:
        return []

    # Rank by similarity alone — the two searches share a model and a space, so
    # their scores are comparable. The subject-scoped search exists to guarantee
    # the subject's own facts are CONSIDERED (they would otherwise lose the
    # global top-k to better-worded neighbours), not to guarantee they win; the
    # English original a refined one-liner grew from is the closest text there
    # is and reaches the top on its own merits. ``scope`` stays as metadata.
    merged.sort(key=lambda h: -h["score"])
    dense_ranking = list(range(len(merged)))

    bm25 = _Bm25([content_tokens(h["text"]) for h in merged])
    query_terms = content_tokens(query)
    terms: list[tuple[str, float]] = [(t, 1.0) for t in query_terms]
    terms += _prf_terms(bm25, dense_ranking, set(query_terms))
    lexical = bm25.score(terms)
    lexical_ranking = sorted(range(len(merged)),
                             key=lambda i: -lexical[i])
    lexical_ranking = [i for i in lexical_ranking if lexical[i] > 0]

    fused = _rrf(dense_ranking, lexical_ranking)
    order = sorted(range(len(merged)), key=lambda i: -fused.get(i, 0.0))

    out = []
    for i in order[:limit]:
        hit = merged[i]
        out.append({
            "text": hit["text"], "source": hit["source"],
            "category": hit["category"], "kind": hit["kind"],
            "row_id": hit["row_id"], "slug": hit["slug"],
            "dense_score": hit["score"], "scope": hit["scope"],
        })
    return out
