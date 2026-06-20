"""Debug-only: backfill facts, artist images and (optionally) AI bios for ONE
account — WITHOUT re-indexing its library.

This mirrors the FACTS/AI sequence of ``LibraryService._enrich_uploads`` but
sources the artist/song list from a Qdrant ``scroll`` over the account's
existing collection (like ``scripts.backfill_sonic_payload``) instead of from a
fresh upload batch. Nothing touches the vectors; we only (re)populate the SQLite
metadata store and download artist images.

Usage
-----
  # non-LLM enrichment (songfacts + artist images + AudioDB bio/meta):
  python -m scripts.backfill_account_enrichment --email you@example.com

  # also run the LLM tasks (refined_facts + artist_bio) — needs a reachable LLM:
  python -m scripts.backfill_account_enrichment --email you@example.com --llm

  # see what WOULD be fetched, hit nothing:
  python -m scripts.backfill_account_enrichment --email you@example.com --dry-run

Run from the repo root with the ``musix`` conda env active, so ``.env`` and
``cache/metadata.db`` resolve.

Proxy / Qdrant note
-------------------
The Qdrant client is built with ``trust_env=False`` (mirroring
``app/resources/db_client.py``) so a shell-exported ``HTTP_PROXY`` never hijacks
the internal ``localhost:6333`` connection to a Dockerised Qdrant. The external
fetchers (songfacts, AudioDB) keep their proxy: they read it from ``.env`` via
``proxy_config``, and ``requests`` falls back to the shell ``HTTP_PROXY`` when
``.env`` has none. The effective external proxy is logged at startup.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from typing import Iterable, List, Tuple

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB
from app.services.artist_facts_service import fetch_facts_for_artists
from app.services.audiodb_service import fetch_audiodb_for_artists
from app.services.proxy_config import get_proxy_url
from app.services.song_facts_service import fetch_facts_for_songs

logger = logging.getLogger("backfill_account_enrichment")

BATCH = 256
# Mirrors app/api/helpers.py::_VALID_USER_ID — keep in sync.
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ── account resolution ───────────────────────────────────────────────────────
def list_accounts() -> None:
    """Print every account in metadata.db with its derived collection name.

    Useful when running on the deployment host where you may not remember the
    exact email / id to target."""
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
    """Return ``(user_id, collection_name)`` for the requested account.

    ``--account-id`` wins over ``--email``. The collection name is derived
    exactly like ``app/api/helpers.py::derive_collection_for_user``.
    """
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
def _iter_points(qdrant: QdrantClient, collection: str) -> Iterable[list]:
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


def collect_artists_and_songs(
    qdrant: QdrantClient, collection: str
) -> Tuple[List[str], List[Tuple[str, str]], int]:
    """Scroll the whole collection, returning sorted unique artists, sorted
    unique (artist, title) pairs, and the total track count."""
    artists: set[str] = set()
    songs: set[Tuple[str, str]] = set()
    n_total = 0
    for batch in _iter_points(qdrant, collection):
        for point in batch:
            n_total += 1
            payload = point.payload or {}
            artist = (payload.get("artist") or "").strip()
            title = (payload.get("title") or "").strip()
            if artist:
                artists.add(artist)
            if artist and title:
                songs.add((artist, title))
    return sorted(artists), sorted(songs), n_total


# ── enrichment stages ────────────────────────────────────────────────────────
async def run_non_llm(
    artists: List[str], songs: List[Tuple[str, str]], collection: str, delay: float
) -> None:
    """songfacts (artists + songs) then AudioDB (images + bio/meta). Mirrors the
    FACTS stage of ``LibraryService._enrich_uploads``."""
    if artists or songs:
        logger.info(
            "[backfill] songfacts: %d artists, %d songs ...", len(artists), len(songs)
        )
        results = await asyncio.gather(
            fetch_facts_for_artists(artists, collection, delay=delay),
            fetch_facts_for_songs(songs, collection, delay=delay),
            return_exceptions=True,
        )
        artist_facts, song_facts = results
        if isinstance(artist_facts, Exception):
            logger.warning("[backfill] artist facts failed: %s", artist_facts)
            artist_facts = {}
        if isinstance(song_facts, Exception):
            logger.warning("[backfill] song facts failed: %s", song_facts)
            song_facts = {}
        logger.info(
            "[backfill] songfacts done: %d/%d artists, %d/%d songs had facts",
            len(artist_facts), len(artists), len(song_facts), len(songs),
        )

    if artists:
        logger.info("[backfill] AudioDB images/bio for %d artists ...", len(artists))
        audiodb = await fetch_audiodb_for_artists(artists, collection, delay=delay)
        if isinstance(audiodb, dict):
            enriched = sum(1 for v in audiodb.values() if v)
            logger.info(
                "[backfill] AudioDB done: %d/%d artists enriched (images + meta)",
                enriched, len(audiodb),
            )


async def run_llm(collection: str, lang: str, n_total: int, qdrant_url: str) -> None:
    """Optional refined_facts + artist_bio via the AI-indexing job runner. Needs
    a reachable LLM; mirrors the AI tail of ``_enrich_uploads``."""
    from app.resources.db_client import DbClient
    from app.services import ai_indexing_service
    from app.services import ai_tasks  # noqa: F401 — side-effect: register tasks
    from app.services.llm_client import (
        is_llm_available, resolve_base_url, resolve_model,
    )

    if not await is_llm_available():
        logger.warning(
            "[backfill] LLM unreachable — skipping refined_facts/artist_bio. "
            "Start the LLM and re-run with --llm."
        )
        return

    base_url, model = resolve_base_url(), resolve_model()
    logger.info(
        "[backfill] LLM reachable (base_url=%s model=%s) — running AI tasks on %s",
        base_url, model, collection,
    )
    with DbClient(qdrant_url=qdrant_url, collection_name=collection) as db:
        for task_type in ("refined_facts", "artist_bio"):
            try:
                job_id = ai_indexing_service.start_job(
                    task_type=task_type,
                    collection_name=collection,
                    lang=lang,
                    db_client=db,
                    llm_client=None,
                    n_total=n_total,
                    llm_base_url=base_url,
                    llm_model=model,
                    bio_source=("web" if task_type == "artist_bio" else "facts"),
                )
                await ai_indexing_service.wait_for_job(job_id)
                logger.info("[backfill] AI task '%s' finished", task_type)
            except ValueError as e:
                logger.warning("[backfill] AI task '%s' skipped: %s", task_type, e)
            except Exception:
                logger.exception("[backfill] AI task '%s' failed", task_type)


# ── diagnostics ──────────────────────────────────────────────────────────────
def log_proxy_diagnostic() -> None:
    """Show which proxy songfacts/AudioDB will actually use, and warn on a
    scheme-less value that ``requests`` would silently ignore."""
    from_env_file = get_proxy_url()
    from_environ = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    effective = from_env_file or from_environ
    if effective:
        source = ".env" if from_env_file else "shell env"
        logger.info("[backfill] external proxy (songfacts/AudioDB): %s (from %s)",
                    effective, source)
        if "://" not in effective:
            logger.warning(
                "[backfill] proxy %r has no scheme — requests will IGNORE it. "
                "Use e.g. http://127.0.0.1:12334", effective,
            )
    else:
        logger.info("[backfill] external proxy: none — songfacts/AudioDB go direct.")


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
    log_proxy_diagnostic()

    # trust_env=False: never route the internal Qdrant connection through a
    # shell-exported HTTP_PROXY (see module docstring).
    qdrant = QdrantClient(url=args.qdrant_url, trust_env=False)
    try:
        exists = qdrant.collection_exists(collection)
    except Exception as e:
        raise SystemExit(
            f"Cannot reach Qdrant at {args.qdrant_url}: {e}\n"
            f"In this project's docker-compose, Qdrant is NOT published to the host "
            f"(`expose`, not `ports`) — the app reaches it in-network as "
            f"http://qdrant:6333. So either run this script INSIDE the container "
            f"(it auto-picks QDRANT_URL=http://qdrant:6333):\n"
            f"    docker cp scripts/backfill_account_enrichment.py musix:/app/scripts/\n"
            f"    docker compose exec musix \\\n"
            f"        python -m scripts.backfill_account_enrichment --account-id <id>\n"
            f"or publish Qdrant to the host (e.g. ports: \"6333:6333\") and pass "
            f"--qdrant-url http://localhost:6333."
        )
    if not exists:
        raise SystemExit(
            f"Collection {collection!r} does not exist on {args.qdrant_url}. "
            f"Is this the right account / Qdrant URL?"
        )

    artists, songs, n_total = collect_artists_and_songs(qdrant, collection)
    logger.info(
        "[backfill] collection has %d tracks → %d unique artists, %d unique songs",
        n_total, len(artists), len(songs),
    )
    if not artists:
        raise SystemExit("No artists found in payloads — nothing to backfill.")

    if args.dry_run:
        sample_a = ", ".join(artists[:10])
        sample_s = ", ".join(f"{a} — {t}" for a, t in songs[:10])
        print("\n=== DRY RUN (no network calls) ===")
        print(f"account     : {user_id}")
        print(f"collection  : {collection}")
        print(f"tracks      : {n_total}")
        print(f"artists     : {len(artists)}  e.g. {sample_a}")
        print(f"songs       : {len(songs)}  e.g. {sample_s}")
        print(f"would run   : songfacts + AudioDB"
              + (" + refined_facts + artist_bio (LLM)" if args.llm else ""))
        return

    await run_non_llm(artists, songs, collection, delay=args.delay)
    if args.llm:
        await run_llm(collection, args.lang, n_total, args.qdrant_url)

    print("\n=== DONE ===")
    print(f"account    : {user_id}")
    print(f"collection : {collection}")
    print(f"artists    : {len(artists)}   songs: {len(songs)}   tracks: {n_total}")
    print("Facts/images/bio are in cache/metadata.db; images under "
          "frontend/covers/artists/. Reload the player to see them.")


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
    parser.add_argument("--lang", default="ru", help="language for facts/bio (default: ru)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between external requests (default: 0.5)")
    parser.add_argument("--llm", action="store_true",
                        help="also run refined_facts + artist_bio (needs a reachable LLM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be fetched; hit no network")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
