"""Owner-only bulk-import: hardlink an existing folder into managed media/<owner>/audio/.

Server-mode escape hatch — the owner already has files on disk and doesn't
want to upload them through HTTP. For each audio file in --source:

  1. Compute SHA-256 (streamed; no full in-memory load).
  2. Skip if (owner, sha) is already in `pending_uploads` (idempotent).
  3. Try ``os.link(src, dst)`` to create a hardlink at
     ``media/<owner_id>/audio/<sha>.<ext>``. On cross-device or filesystems
     without hardlink support (most notably FAT32/exFAT), fall back to
     ``shutil.copy2``.
  4. Insert `pending_uploads` row with status='uploaded'.

After all files are processed, trigger ``IndexingService.index_uploads(owner_id, all_rows)``
synchronously (CLI, not async) so the operator sees progress in the terminal.

Usage
-----
    python -m scripts.hardlink_owner_library \\
        --source /home/me/Music \\
        --owner-email me@example.com

Optional flags:
    --dry-run         List candidate files, do not hardlink or insert.
    --skip-indexing   Hardlink + insert rows but do not trigger indexing
                      (use when you want to batch-commit later via HTTP).

Refuses to run unless the user identified by --owner-email has role='owner'.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

# Importing app.services pulls in heavy chains (clap_features → laion_clap) in
# some import paths, which parse sys.argv at import time and would eat our CLI
# flags. Save + restore argv across the import. Pattern lifted from
# scripts/backfill_artist_slugs.py.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from app.resources.metadata_db import MetadataDB
    from app.services.uploads_service import (
        media_root_default,
        _AUDIO_EXTENSIONS,
    )
finally:
    sys.argv = _saved_argv

logger = logging.getLogger("hardlink_owner_library")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _sha256_streamed(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> str:
    """Try ``os.link`` first; fall back to ``shutil.copy2`` if hardlink fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "linked"
    except (OSError, NotImplementedError) as e:
        logger.debug("[hardlink] fallback copy for %s: %s", src.name, e)
        shutil.copy2(src, dst)
        return "copied"


def _resolve_owner(email: str) -> str:
    """Look up users.id for the given email. Refuse non-owner."""
    MetadataDB.init()
    conn = MetadataDB.get()
    row = conn.execute(
        "SELECT id, role FROM users WHERE email = ? LIMIT 1",
        (email.lower(),),
    ).fetchone()
    if not row:
        logger.error("no user with email %s", email)
        sys.exit(2)
    user_id, role = row[0], row[1]
    if role != "owner":
        logger.error("user %s is role=%s; only owner may run this script", email, role)
        sys.exit(3)
    return user_id


def run(
    *,
    source: str,
    owner_email: str,
    owner_id_lookup: Callable[[str], str] = _resolve_owner,
    indexer: Callable | None = None,
    dry_run: bool = False,
) -> int:
    """Hardlink + insert. Returns count of files processed (excluding skipped).

    ``owner_id_lookup`` and ``indexer`` are injectable for testing.
    """
    src_root = Path(source).resolve()
    if not src_root.is_dir():
        logger.error("--source is not a directory: %s", src_root)
        sys.exit(1)

    owner_id = owner_id_lookup(owner_email)
    media_root = Path(os.environ.get("MUSIX_MEDIA_ROOT") or media_root_default())

    files = [
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
    ]
    logger.info("found %d audio files under %s", len(files), src_root)

    if dry_run:
        for p in files:
            logger.info("[DRY] would link: %s", p)
        return len(files)

    MetadataDB.init()
    processed = 0
    for src in files:
        sha = _sha256_streamed(src)
        # Idempotent — skip if (owner, sha) already exists with any status.
        conn = MetadataDB.get()
        existing = conn.execute(
            "SELECT upload_id FROM pending_uploads WHERE account_id = ? AND sha256 = ? LIMIT 1",
            (owner_id, sha),
        ).fetchone()
        if existing:
            logger.info("[SKIP] sha=%s already in DB (upload_id=%s)", sha[:8], existing[0])
            continue
        ext = src.suffix.lower()
        dst = media_root / owner_id / "audio" / f"{sha}{ext}"
        action = _link_or_copy(src, dst)
        upload_id = MetadataDB.create_pending_upload(
            account_id=owner_id,
            sha256=sha,
            original_filename=src.name,
            size_bytes=src.stat().st_size,
            storage_path=str(dst),
        )
        logger.info("[%s] %s → %s (upload_id=%s)", action.upper(), src.name, dst, upload_id)
        processed += 1

    if indexer is not None and processed > 0:
        rows = MetadataDB.list_pending_uploads_by_account(owner_id, status="uploaded")
        logger.info("triggering indexing for %d uploaded rows", len(rows))
        indexer(account_id=owner_id, upload_rows=rows)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--source", required=True, help="Folder to scan recursively")
    parser.add_argument("--owner-email", required=True, help="Owner's email (must be role='owner')")
    parser.add_argument("--dry-run", action="store_true", help="List files; do not modify anything")
    parser.add_argument("--skip-indexing", action="store_true", help="Hardlink + insert rows only")
    args = parser.parse_args()

    indexer = None
    if not args.skip_indexing:
        # Lazy-import the heavy IndexingService chain only when actually used.
        _saved = sys.argv
        sys.argv = sys.argv[:1]
        try:
            from app.resources.db_client import DbClient
            from app.services.indexing_service import IndexingService
        finally:
            sys.argv = _saved
        db = DbClient()
        db._connect()
        svc = IndexingService(db.search_engine)

        def indexer(*, account_id, upload_rows):
            return svc.index_uploads(account_id=account_id, upload_rows=upload_rows)

    run(
        source=args.source,
        owner_email=args.owner_email,
        indexer=indexer,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
