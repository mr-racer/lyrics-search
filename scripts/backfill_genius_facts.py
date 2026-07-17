"""Debug-only: backfill Genius.com song facts (description + line annotations)
and producer/label metadata for ONE account — without re-indexing its library.

Facts are written to the same ``song_facts`` table / slug scheme that
``track_chat_service.resolve_song_facts_list`` already reads from, so Genius
facts show up in both the "chat about song" and "explain this line" prompts
with zero changes to that code. Producer/label are written to
``track_metadata`` via a narrow update (title/artist/album etc. are left
untouched); the credits also land in ``songs.producers_genius``/``label``
(slug-keyed), where ``song_facts_service.apply_song_relations`` picks them up
for the player UI — producers merged with the GLiNER extraction, deduped.

Usage
-----
  # sanity-check the Genius URLs a slugifier would build, no network calls:
  python -m scripts.backfill_genius_facts --email you@example.com --sample 20

  # see what WOULD be processed, hit no network:
  python -m scripts.backfill_genius_facts --email you@example.com --dry-run

  # full run:
  python -m scripts.backfill_genius_facts --email you@example.com

  # re-process songs that already have genius.com facts:
  python -m scripts.backfill_genius_facts --email you@example.com --force

Run from the repo root with the project's env active, so ``.env`` and
``cache/metadata.db`` resolve. See ``scripts/backfill_account_enrichment.py``
for the Qdrant-reachability notes this script shares (same account resolution,
same trust_env=False Qdrant client).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB
from app.services import genius_service
from app.services.proxy_config import get_proxy_url
from app.services.song_facts_service import get_song_facts_key

logger = logging.getLogger("backfill_genius_facts")

BATCH = 256
# Mirrors app/api/helpers.py::_VALID_USER_ID — keep in sync.
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ── account resolution (mirrors backfill_account_enrichment.py) ─────────────
def list_accounts() -> None:
    rows = MetadataDB._connect().execute(
        "SELECT role, email, id FROM users ORDER BY role DESC, email"
    ).fetchall()
    if not rows:
        print("(no accounts in metadata.db)")
        return
    print("\nAccounts in metadata.db:")
    for role, email, uid in rows:
        print(f"  {role:<6} | {email:<32} | id={uid} | collection=acct_{uid}")


def resolve_account(email: str | None, account_id: str | None) -> Tuple[str, str]:
    user_id: str | None = None
    if account_id:
        user_id = account_id.strip()
    elif email:
        row = MetadataDB.get_user_by_email(email.strip().lower())
        if row is None:
            raise SystemExit(
                f"No account found for email {email!r}. "
                f"Check the address, or pass --account-id directly."
            )
        user_id = row["id"]
    else:
        raise SystemExit("Provide --email or --account-id.")

    if not user_id or not _VALID_USER_ID.match(user_id):
        raise SystemExit(f"Invalid user id {user_id!r} (expected [A-Za-z0-9_-]+).")

    return user_id, f"acct_{user_id}"


# ── collection scan ──────────────────────────────────────────────────────────
def _iter_points(qdrant: QdrantClient, collection: str):
    offset = None
    while True:
        points, next_offset = qdrant.scroll(
            collection_name=collection,
            limit=BATCH,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            return
        yield points
        if next_offset is None:
            return
        offset = next_offset


def collect_songs(
    qdrant: QdrantClient, collection: str
) -> Tuple[Dict[Tuple[str, str], List[str]], int]:
    """Scroll the whole collection, mapping (artist, title) -> [track_id, ...].

    Unlike backfill_account_enrichment's collect_artists_and_songs, this keeps
    track_id per song (needed to write producer/label onto every matching
    track_metadata row) instead of discarding it after dedup.
    """
    songs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    n_total = 0
    for batch in _iter_points(qdrant, collection):
        for point in batch:
            n_total += 1
            payload = point.payload or {}
            artist = (payload.get("artist") or "").strip()
            title = (payload.get("title") or "").strip()
            if artist and title:
                songs[(artist, title)].append(str(point.id))
    return songs, n_total


# ── genius-song processing ──────────────────────────────────────────────────
def _already_processed(song_slug: str) -> bool:
    conn = MetadataDB._connect()
    row = conn.execute(
        "SELECT 1 FROM song_facts WHERE song_slug = ? AND source = 'genius.com' LIMIT 1",
        (song_slug,),
    ).fetchone()
    return row is not None


async def process_song(
    artist: str,
    title: str,
    track_ids: List[str],
    collection: str,
    delay: float,
    force: bool,
) -> str:
    """Returns 'skipped' | 'ok' | 'failed'."""
    song_slug = get_song_facts_key(artist, title)

    if not force and _already_processed(song_slug):
        logger.info("[genius] skip (already processed): %s — %s", artist, title)
        return "skipped"

    url = genius_service.build_genius_url(artist, title)
    try:
        parsed = await asyncio.to_thread(genius_service.fetch_song_data, url, delay)
    except Exception as e:
        logger.warning("[genius] failed for '%s — %s' (%s): %s", artist, title, url, e)
        return "failed"

    facts = genius_service.build_song_facts(parsed)
    description_facts = [f for f, cat in facts if cat == "genius_description"]
    annotation_facts = [f for f, cat in facts if cat == "genius_annotation"]
    if description_facts:
        MetadataDB.add_song_facts_batch(
            song_slug, collection, description_facts,
            source="genius.com", category="genius_description",
        )
    if annotation_facts:
        MetadataDB.add_song_facts_batch(
            song_slug, collection, annotation_facts,
            source="genius.com", category="genius_annotation",
        )

    producer, label = genius_service.build_producer_label(parsed)
    if parsed["producers"] or label:
        # Mirror the live pipeline (genius_facts_service): songs.producers_genius
        # + songs.label are what apply_song_relations overlays (merged with the
        # GLiNER extraction) onto player/search/artist read paths.
        MetadataDB.set_song_genius_credits(song_slug, parsed["producers"], label, collection)
    if producer or label:
        for track_id in track_ids:
            MetadataDB.update_track_producer_label(collection, track_id, producer, label)

    logger.info(
        "[genius] ok: %s — %s (%d facts, producer=%r, label=%r)",
        artist, title, len(facts), producer, label,
    )
    return "ok"


# ── diagnostics ──────────────────────────────────────────────────────────────
def log_proxy_diagnostic() -> None:
    from_env_file = get_proxy_url()
    from_environ = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    effective = from_env_file or from_environ
    if effective:
        source = ".env" if from_env_file else "shell env"
        logger.info("[genius] external proxy: %s (from %s)", effective, source)
    else:
        logger.info("[genius] external proxy: none — Genius requests go direct.")


# ── entrypoint ───────────────────────────────────────────────────────────────
async def main_async(args: argparse.Namespace) -> None:
    MetadataDB.init()

    if args.list_accounts:
        list_accounts()
        return
    if not args.email and not args.account_id:
        print("Specify --email or --account-id (or --list-accounts to see options).")
        list_accounts()
        raise SystemExit(2)

    user_id, collection = resolve_account(args.email, args.account_id)
    logger.info("[genius] account=%s collection=%s", user_id, collection)
    log_proxy_diagnostic()

    qdrant = QdrantClient(url=args.qdrant_url, trust_env=False)
    try:
        exists = qdrant.collection_exists(collection)
    except Exception as e:
        raise SystemExit(f"Cannot reach Qdrant at {args.qdrant_url}: {e}")
    if not exists:
        raise SystemExit(f"Collection {collection!r} does not exist on {args.qdrant_url}.")

    songs, n_total = collect_songs(qdrant, collection)
    unique_songs = sorted(songs.keys())
    logger.info(
        "[genius] collection has %d tracks -> %d unique songs",
        n_total, len(unique_songs),
    )
    if not unique_songs:
        raise SystemExit("No (artist, title) pairs found in payloads — nothing to backfill.")

    if args.sample:
        print(f"\n=== SAMPLE URLS (first {args.sample}, no network calls) ===")
        for artist, title in unique_songs[: args.sample]:
            url = genius_service.build_genius_url(artist, title)
            print(f"  {artist} — {title}\n    -> {url}")
        return

    if args.dry_run:
        sample_s = ", ".join(f"{a} — {t}" for a, t in unique_songs[:10])
        print("\n=== DRY RUN (no network calls) ===")
        print(f"account     : {user_id}")
        print(f"collection  : {collection}")
        print(f"tracks      : {n_total}")
        print(f"songs       : {len(unique_songs)}  e.g. {sample_s}")
        print(f"force       : {args.force}")
        return

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed_songs: List[str] = []
    for artist, title in unique_songs:
        result = await process_song(
            artist, title, songs[(artist, title)], collection, args.delay, args.force,
        )
        counts[result] += 1
        if result == "failed":
            failed_songs.append(f"{artist} — {title}")
        await asyncio.sleep(args.delay)

    print("\n=== DONE ===")
    print(f"account    : {user_id}")
    print(f"collection : {collection}")
    print(f"songs      : {len(unique_songs)}   ok: {counts['ok']}   "
          f"skipped: {counts['skipped']}   failed: {counts['failed']}")
    if failed_songs:
        print("\nFailed songs (re-run with --force after investigating, or ignore):")
        for s in failed_songs:
            print(f"  - {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=None,
                        help="account email (use --list-accounts to see options)")
    parser.add_argument("--account-id", default=None,
                        help="account id; overrides --email (collection = acct_<id>)")
    parser.add_argument("--list-accounts", action="store_true",
                        help="print accounts in metadata.db and exit")
    parser.add_argument("--qdrant-url",
                        default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--delay", type=float, default=0.3,
                        help="seconds between Genius requests (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be processed; hit no network")
    parser.add_argument("--force", action="store_true",
                        help="re-process songs that already have genius.com facts")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="print the first N built Genius URLs and exit (no network)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
