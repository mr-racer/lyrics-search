"""The vector index over raw facts.

One Qdrant collection, ``facts``, holding the dense vector of every raw
songfacts/Genius fact — and almost nothing else. Three design decisions worth
knowing before touching this:

**Vectors only.** The payload carries ``{kind, row_id, slug}`` and no text.
Qdrant is slow at handing back payloads at volume — this codebase already routes
around that (``light_points``' memoised scroll, ``PAYLOAD_EXCLUDE_HEAVY``,
credits read from the SQLite mirror) — so search returns ids and the words are
joined from SQLite by :meth:`MetadataDB.get_facts_by_ids`.

**Shared, not per-account.** Every other collection is ``acct_{id}``, but
``song_facts``/``artist_facts`` are themselves a shared pool keyed by slug with
per-account visibility in ``fact_visibility``. Embedding the same fact once per
account would buy nothing. Isolation is enforced on the way out, in
``facts_retrieval``, never here.

**Filled lazily.** No indexing stage, no backfill gate: the subject of a
question gets indexed on the spot (~80 short texts, well under a second on the
resident GPU model), and a background warm-up walks the rest of the account's
pool so that cross-entity retrieval eventually sees everything.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger(__name__)

COLLECTION = "facts"

# Stable ids, so re-indexing an entity overwrites rather than duplicates.
# Namespaced because song_facts.id and artist_facts.id are independent
# autoincrement sequences — without the kind in the name they would collide.
_NAMESPACE = uuid.UUID("6f9b2a1e-6a3c-4d0e-9c2b-9a5f4e1d7c33")

KINDS = ("song", "artist")

# Facts shorter than this are stubs ("January 8, 1947 - January 10, 2016") and
# only add noise to a similarity search.
MIN_FACT_CHARS = 40
_ENCODE_BATCH = 32

# Entities already indexed in this process. Purely an optimisation — a miss
# costs one extra retrieve, never a wrong answer.
_indexed: set[tuple[str, str]] = set()
_lock = threading.Lock()


def point_id(kind: str, row_id: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{row_id}"))


def ensure_collection(qdrant) -> bool:
    """Create ``facts`` if missing. Returns False when Qdrant is unreachable —
    every caller degrades to "no dense leg" rather than raising."""
    from qdrant_client import models
    from app.resources.model_registry import ModelRegistry

    try:
        if qdrant.collection_exists(COLLECTION):
            return True
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                ModelRegistry.VECTOR_NAME: models.VectorParams(
                    size=ModelRegistry.VECTOR_DIM, distance=models.Distance.COSINE,
                ),
            },
        )
        # Subject-scoped retrieval filters on these two, so they must be indexed
        # server-side; without the index Qdrant falls back to a full scan.
        for field in ("slug", "kind"):
            try:
                qdrant.create_payload_index(
                    collection_name=COLLECTION, field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 — already exists is fine
                pass
        logger.info("[facts_index] created collection %r", COLLECTION)
        return True
    except Exception:
        logger.warning("[facts_index] could not ensure collection", exc_info=True)
        return False


def index_entity(qdrant, kind: str, slug: str, *, force: bool = False) -> int:
    """Embed and upsert every raw fact of one entity. Returns rows written.

    Idempotent: point ids are derived from the row id, so a re-run overwrites in
    place. Cheap to call on every question about a subject.
    """
    from qdrant_client import models
    from app.resources.metadata_db import MetadataDB
    from app.resources.model_registry import ModelRegistry

    if kind not in KINDS or not slug:
        return 0
    key = (kind, slug)
    if not force and key in _indexed:
        return 0

    try:
        rows = [r for r in MetadataDB.get_facts_with_ids(kind, slug)
                if len((r.get("fact") or "").strip()) >= MIN_FACT_CHARS]
    except Exception:
        logger.warning("[facts_index] fact read failed for %s/%s", kind, slug,
                       exc_info=True)
        return 0
    if not rows:
        with _lock:
            _indexed.add(key)
        return 0

    if not ensure_collection(qdrant):
        return 0

    try:
        vectors = ModelRegistry.encode_text(
            [r["fact"] for r in rows],           # document side — no prefix
            batch_size=_ENCODE_BATCH, convert_to_numpy=True,
        )
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id(kind, r["id"]),
                    vector={ModelRegistry.VECTOR_NAME: vec.tolist()},
                    payload={"kind": kind, "row_id": r["id"], "slug": slug},
                )
                for r, vec in zip(rows, vectors)
            ],
        )
    except Exception:
        logger.warning("[facts_index] upsert failed for %s/%s", kind, slug,
                       exc_info=True)
        return 0

    with _lock:
        _indexed.add(key)
    logger.info("[facts_index] indexed %d facts for %s/%s", len(rows), kind, slug)
    return len(rows)


def search(qdrant, query_vector, *, limit: int, slugs: list[str] | None = None,
           kind: str | None = None) -> list[dict]:
    """Nearest facts as ``[{"kind", "row_id", "slug", "score"}]``.

    ``slugs`` narrows to specific entities (the subject and its neighbours);
    None searches the whole pool, which is what surfaces an explanation stored
    under a *different* song. Text is never returned — join it from SQLite.
    """
    from qdrant_client import models
    from app.resources.model_registry import ModelRegistry

    conditions = []
    if slugs:
        conditions.append(models.FieldCondition(
            key="slug", match=models.MatchAny(any=list(slugs)),
        ))
    if kind:
        conditions.append(models.FieldCondition(
            key="kind", match=models.MatchValue(value=kind),
        ))
    query_filter = models.Filter(must=conditions) if conditions else None

    try:
        res = qdrant.query_points(
            collection_name=COLLECTION,
            query=list(query_vector),
            using=ModelRegistry.VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception:
        logger.warning("[facts_index] search failed", exc_info=True)
        return []

    out = []
    for point in res.points:
        payload = point.payload or {}
        if payload.get("row_id") is None:
            continue
        out.append({
            "kind": payload.get("kind") or "song",
            "row_id": int(payload["row_id"]),
            "slug": payload.get("slug") or "",
            "score": float(point.score or 0.0),
        })
    return out


def warm_collection(qdrant, collection_name: str, *, budget: int = 200) -> int:
    """Index up to ``budget`` not-yet-seen entities of one account.

    Runs in a background thread. Bounded on purpose: a first-run library of
    thousands of entities should trickle in over several questions rather than
    monopolise the GPU on the first one.
    """
    from app.resources.metadata_db import MetadataDB

    written = 0
    for kind in KINDS:
        try:
            slugs = MetadataDB.get_visible_fact_slugs(kind, collection_name)
        except Exception:
            logger.warning("[facts_index] warm-up slug read failed", exc_info=True)
            continue
        for slug in slugs:
            if (kind, slug) in _indexed:
                continue
            if written >= budget:
                return written
            written += 1
            index_entity(qdrant, kind, slug)
    return written


def forget_cache() -> None:
    """Drop the "already indexed" memo. For tests and factory reset."""
    with _lock:
        _indexed.clear()
