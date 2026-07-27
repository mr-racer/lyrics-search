"""Debug-only: backfill Genius facts + all five ai_tasks for ONE account's
existing library, without touching text/CLAP embeddings or the AudioDB /
songfacts.com fact-scraping stage.

Two phases, run in order:

1. Genius force-refresh — re-fetch Genius facts + producer/label for every
   song in the collection (default: force, ignoring "already processed" —
   ``613e3f2`` fixed a curl_cffi block that could make earlier fetches
   silently fail). Reuses ``collect_songs``/``process_song`` from
   ``scripts.backfill_genius_facts`` directly.

2. All five registered ``ai_tasks`` (sonic_vibe, refined_facts,
   fact_relations, lyric_gems, artist_bio) via the exact production code
   path ``LibraryService._run_ai_tasks`` uses during a real indexing run —
   same order, same LLM-reachability gate, same per-task skip-if-cached
   idempotency. Nothing here forces those five to reprocess already-cached
   songs/tracks/artists.

See docs/superpowers/specs/2026-07-20-post-genius-ai-backfill-design.md.

Usage
-----
  # see what would run, no network/LLM calls:
  python -m scripts.backfill_gliner_genius --email you@example.com --dry-run

  # full run (Genius force-refresh + all five ai_tasks):
  python -m scripts.backfill_gliner_genius --email you@example.com

  # Genius-only pass:
  python -m scripts.backfill_gliner_genius --email you@example.com --skip-ai-tasks

  # ai_tasks-only pass (Genius already backfilled):
  python -m scripts.backfill_gliner_genius --email you@example.com --skip-genius

  # incremental re-run: don't re-hit Genius for songs already processed:
  python -m scripts.backfill_gliner_genius --email you@example.com --no-force-genius

Run from the repo root (inside the container) with the project's env active,
so ``.env`` and ``cache/metadata.db`` resolve.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re

from qdrant_client import QdrantClient

from app.resources.db_client import DbClient
from app.resources.metadata_db import MetadataDB
from app.services.library_service import LibraryService
from app.services.llm_client import is_llm_available
from scripts.backfill_genius_facts import (
    collect_songs,
    list_accounts,
    log_proxy_diagnostic,
    process_song,
    resolve_account,
)

logger = logging.getLogger("backfill_gliner_genius")

AI_TASK_ORDER = ("sonic_vibe", "refined_facts", "fact_relations", "lyric_gems", "artist_bio")
# Mirrors app/api/helpers.py::_VALID_USER_ID — keep in sync (also enforced by
# resolve_account, imported above; kept here only for the --dry-run message).
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ── phase 1: Genius force-refresh ────────────────────────────────────────────
async def run_genius_phase(
    qdrant: QdrantClient, collection: str, delay: float, force: bool, dry_run: bool,
) -> None:
    songs, album_artist_slugs, n_total = collect_songs(qdrant, collection)
    unique_songs = sorted(songs.keys())
    logger.info(
        "[genius] collection has %d tracks -> %d unique songs",
        n_total, len(unique_songs),
    )
    if not unique_songs:
        logger.warning("[genius] no (artist, title) pairs found — nothing to do")
        return

    if dry_run:
        sample_s = ", ".join(f"{a} — {t}" for a, t in unique_songs[:10])
        print(f"\n=== GENIUS DRY RUN (force={force}) ===")
        print(f"songs : {len(unique_songs)}  e.g. {sample_s}")
        return

    log_proxy_diagnostic()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed_songs: list[str] = []
    for artist, title in unique_songs:
        result = await process_song(
            artist, title, songs[(artist, title)], collection, delay, force,
            album_artist_slug=album_artist_slugs.get((artist, title)),
        )
        counts[result] += 1
        if result == "failed":
            failed_songs.append(f"{artist} — {title}")
        await asyncio.sleep(delay)

    print("\n=== GENIUS PHASE DONE ===")
    print(f"songs : {len(unique_songs)}   ok: {counts['ok']}   "
          f"skipped: {counts['skipped']}   failed: {counts['failed']}")
    if failed_songs:
        print("\nFailed songs (re-run with --force-genius after investigating, or ignore):")
        for s in failed_songs:
            print(f"  - {s}")


# ── phase 2: all five ai_tasks ───────────────────────────────────────────────
async def run_ai_tasks_phase(
    qdrant: QdrantClient, qdrant_url: str, collection: str, lang: str, dry_run: bool,
) -> None:
    try:
        n_total = int(qdrant.count(collection_name=collection, exact=True).count)
    except Exception as e:
        logger.warning("[ai_tasks] could not count collection %s: %s", collection, e)
        n_total = 0

    if dry_run:
        print(f"\n=== AI TASKS DRY RUN ===")
        print(f"tracks : {n_total}")
        print(f"tasks (in order): {', '.join(AI_TASK_ORDER)}")
        return

    if not await is_llm_available():
        print(
            "\n=== AI TASKS SKIPPED (no LLM reachable) ===\n"
            f"Would have run: {', '.join(AI_TASK_ORDER)}"
        )
        return

    print(f"\n=== AI TASKS START (tracks={n_total}) ===")
    with DbClient(qdrant_url=qdrant_url, collection_name=collection) as db_client:
        service = LibraryService(db_client=db_client)
        # Reuses the exact production code path (order + LLM gate + per-task
        # skip-if-cached idempotency) that a real indexing run drives — see
        # library_service.LibraryService._run_ai_tasks. job=None is already a
        # supported call shape; progress surfaces via its own logger.info lines.
        await service._run_ai_tasks(collection, n_total, lang=lang, job=None)
    print("=== AI TASKS DONE ===")


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
    logger.info("[backfill] account=%s collection=%s", user_id, collection)

    qdrant = QdrantClient(url=args.qdrant_url, trust_env=False)
    try:
        exists = qdrant.collection_exists(collection)
    except Exception as e:
        raise SystemExit(f"Cannot reach Qdrant at {args.qdrant_url}: {e}")
    if not exists:
        raise SystemExit(f"Collection {collection!r} does not exist on {args.qdrant_url}.")

    if not args.skip_genius:
        await run_genius_phase(
            qdrant, collection, args.delay, force=not args.no_force_genius, dry_run=args.dry_run,
        )
    else:
        print("=== GENIUS PHASE SKIPPED (--skip-genius) ===")

    if not args.skip_ai_tasks:
        await run_ai_tasks_phase(qdrant, args.qdrant_url, collection, args.lang, args.dry_run)
    else:
        print("=== AI TASKS PHASE SKIPPED (--skip-ai-tasks) ===")


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
    parser.add_argument("--lang", default="ru",
                        help="lang for sonic_vibe/refined_facts/fact_relations/"
                             "lyric_gems/artist_bio (default: ru)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="seconds between Genius requests (default: 0.3)")
    parser.add_argument("--no-force-genius", action="store_true",
                        help="skip songs that already have genius.com facts, "
                             "instead of re-fetching everything")
    parser.add_argument("--skip-genius", action="store_true",
                        help="skip the Genius force-refresh phase")
    parser.add_argument("--skip-ai-tasks", action="store_true",
                        help="skip the sonic_vibe/refined_facts/fact_relations/"
                             "lyric_gems/artist_bio phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="print workload counts for both phases; no network/LLM calls")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
