"""Debug-only: recompute the CLAP top-pairs cache (the player's «похожие /
контраст» rail + the home discovery card) for ONE account — WITHOUT re-indexing.

This calls ``similarity_service.analyze_collection`` directly. CLAP vectors are
read straight from Qdrant; audio is NOT re-encoded — so this is the cheap way to
refresh the cache after a similarity-logic change (e.g. the build-time
same-album exclusion) instead of running a full re-index just for its tail
ANALYSIS stage.

The cache file lands at ``cache/top_pairs/acct_<id>.json`` (overwriting any
stale one), which the per-track ``/library/top-pairs/{track_id}`` endpoint and
the home ``/library/top-pairs`` showcase both read.

Usage
-----
  # recompute for one account (by email or id):
  python -m scripts.recompute_similarity --email you@example.com
  python -m scripts.recompute_similarity --account-id <id>

  # see what WOULD run (no compute, no write):
  python -m scripts.recompute_similarity --email you@example.com --dry-run

  # list accounts when you don't remember the email/id:
  python -m scripts.recompute_similarity --list-accounts

Run from the repo root with the ``musix`` conda env active, so ``.env`` and
``cache/`` resolve. The Qdrant client uses ``trust_env=False`` (mirroring
``app/resources/db_client.py``) so a shell-exported ``HTTP_PROXY`` never hijacks
the internal ``localhost:6333`` connection. If Qdrant is only reachable in the
Docker network, run this inside the container (see the error hint below).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from typing import Tuple

from qdrant_client import QdrantClient

from app.resources.metadata_db import MetadataDB
from app.services.similarity_service import analyze_collection

logger = logging.getLogger("recompute_similarity")

# Mirrors app/api/helpers.py::_VALID_USER_ID — keep in sync.
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# ── account resolution ───────────────────────────────────────────────────────
def list_accounts() -> None:
    """Print every account in metadata.db with its derived collection name."""
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


# ── progress ─────────────────────────────────────────────────────────────────
async def _on_progress(stage, current, total, message) -> None:
    logger.info("[recompute] %s", message)


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
    logger.info("[recompute] account=%s collection=%s", user_id, collection)

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
            f"    docker cp scripts/recompute_similarity.py musix:/app/scripts/\n"
            f"    docker compose exec musix \\\n"
            f"        python -m scripts.recompute_similarity --account-id <id>\n"
            f"or publish Qdrant to the host (e.g. ports: \"6333:6333\") and pass "
            f"--qdrant-url http://localhost:6333."
        )
    if not exists:
        raise SystemExit(
            f"Collection {collection!r} does not exist on {args.qdrant_url}. "
            f"Is this the right account / Qdrant URL?"
        )

    if args.dry_run:
        try:
            count = qdrant.count(collection, exact=True).count
        except Exception:
            count = "?"
        print("\n=== DRY RUN (no compute, no write) ===")
        print(f"account    : {user_id}")
        print(f"collection : {collection}  ({count} points)")
        print("would run  : analyze_collection → cache/top_pairs/"
              f"{collection}.json")
        print("note       : sub-30s tracks are skipped, and same-album neighbours "
              "are excluded from the SIMILAR list at build time.")
        return

    cache_path = await analyze_collection(
        qdrant_client=qdrant,
        collection_name=collection,
        progress_callback=_on_progress,
    )

    print("\n=== DONE ===")
    print(f"account    : {user_id}")
    print(f"collection : {collection}")
    print(f"cache      : {cache_path}")
    print("Reload the player — «похожие / контраст» now fill from the fresh cache.")


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
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would run; compute nothing, write nothing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
