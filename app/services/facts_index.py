"""The vector index over raw facts.

One Qdrant collection per account — ``facts_acct_{id}``, alongside that
account's own ``acct_{id}`` — holding the dense vector of every raw
songfacts/Genius fact it can see. Three design decisions worth knowing before
touching this:

**Vectors only.** The payload carries ``{kind, row_id, slug}`` and no text.
Qdrant is slow at handing back payloads at volume — this codebase already routes
around that (``light_points``' memoised scroll, ``PAYLOAD_EXCLUDE_HEAVY``,
credits read from the SQLite mirror) — so search returns ids and the words are
joined from SQLite by :meth:`MetadataDB.get_facts_by_ids`.

**Per-account, like every other collection.** ``song_facts``/``artist_facts``
are a shared pool keyed by slug — two accounts owning the same song share the
literal rows, and ``fact_visibility`` is what separates them. The vector index
deliberately does NOT copy that shape: it is partitioned per account, so a
cross-entity search physically cannot reach another account's material. The
cost is a duplicated embedding for a song two accounts both own, which is
cheap; the alternative made isolation depend on remembering to apply a filter.

**Filled on demand, never in bulk.** No indexing stage, no backfill, no
background sweep: the subject of a question is embedded on the spot (~80 short
texts, well under a second on the resident GPU model) and nothing else is. A
bulk warm-up was tried and removed — it turned the first question about a track
into minutes of GPU work for facts nobody had asked about, and on a box where
the model lands on the CPU it is far worse than that. The pool therefore fills
as the listener explores, and cross-entity retrieval sees whatever has been
visited so far.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger(__name__)

def collection_for(account_collection: str) -> str:
    """``acct_abc`` -> ``facts_acct_abc``.

    Always derive it from the caller's own collection (which itself comes from
    ``derive_collection_for_user``) — never accept a facts-collection name from
    anywhere else.
    """
    return f"facts_{account_collection}"


# Stable ids, so re-indexing an entity overwrites rather than duplicates.
# Namespaced because song_facts.id and artist_facts.id are independent
# autoincrement sequences — without the kind in the name they would collide.
_NAMESPACE = uuid.UUID("6f9b2a1e-6a3c-4d0e-9c2b-9a5f4e1d7c33")

KINDS = ("song", "artist")

# Facts shorter than this are stubs ("January 8, 1947 - January 10, 2016") and
# only add noise to a similarity search.
MIN_FACT_CHARS = 40
_ENCODE_BATCH = 32

# Entities already indexed in this process, keyed by (account collection, kind,
# slug). Purely an optimisation — a miss costs one extra encode, never a wrong
# answer. Keyed by account too: the same slug lives in as many facts collections
# as there are accounts that own it.
_indexed: set[tuple[str, str, str]] = set()
_lock = threading.Lock()


def point_id(kind: str, row_id: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{row_id}"))


def ensure_collection(qdrant, collection: str) -> bool:
    """Create the account's facts collection if missing. Returns False when
    Qdrant is unreachable — every caller degrades to "no dense leg" rather than
    raising."""
    from qdrant_client import models
    from app.resources.model_registry import ModelRegistry

    try:
        if qdrant.collection_exists(collection):
            return True
        qdrant.create_collection(
            collection_name=collection,
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
                    collection_name=collection, field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 — already exists is fine
                pass
        logger.info("[facts_index] created collection %r", collection)
        return True
    except Exception:
        logger.warning("[facts_index] could not ensure %r", collection, exc_info=True)
        return False


def drop_collection(qdrant, account_collection: str) -> None:
    """Drop an account's facts collection.

    Called wherever ``acct_{id}`` itself is dropped: a wipe that left the facts
    behind would hand them to whoever took that account id next.
    """
    collection = collection_for(account_collection)
    try:
        qdrant.delete_collection(collection)
        logger.info("[facts_index] dropped %r", collection)
    except Exception as e:  # noqa: BLE001 — "no such collection" is the normal case
        logger.info("[facts_index] drop of %r skipped: %s", collection, e)
    with _lock:
        for key in [k for k in _indexed if k[0] == account_collection]:
            _indexed.discard(key)


def index_entity(qdrant, account_collection: str, kind: str, slug: str,
                 *, force: bool = False) -> int:
    """Embed and upsert every raw fact of one entity. Returns rows written.

    Idempotent: point ids are derived from the row id, so a re-run overwrites in
    place. Cheap to call on every question about a subject.
    """
    from qdrant_client import models
    from app.resources.metadata_db import MetadataDB
    from app.resources.model_registry import ModelRegistry

    if kind not in KINDS or not slug or not account_collection:
        return 0
    key = (account_collection, kind, slug)
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

    collection = collection_for(account_collection)
    if not ensure_collection(qdrant, collection):
        return 0

    try:
        vectors = ModelRegistry.encode_text(
            [r["fact"] for r in rows],           # document side — no prefix
            batch_size=_ENCODE_BATCH, convert_to_numpy=True,
        )
        qdrant.upsert(
            collection_name=collection,
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
    logger.info("[facts_index] indexed %d facts for %s/%s into %s",
                len(rows), kind, slug, collection)
    return len(rows)


def search(qdrant, account_collection: str, query_vector, *, limit: int,
           slugs: list[str] | None = None, kind: str | None = None) -> list[dict]:
    """Nearest facts as ``[{"kind", "row_id", "slug", "score"}]``.

    ``slugs`` narrows to specific entities (the subject); None searches this
    account's whole indexed pool, which is what surfaces an explanation stored
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
            collection_name=collection_for(account_collection),
            # float(), not list(): a numpy array yields np.float32 items, which
            # the client will not serialise.
            query=[float(x) for x in query_vector],
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


def forget_cache() -> None:
    """Drop the "already indexed" memo. For tests and factory reset."""
    with _lock:
        _indexed.clear()
