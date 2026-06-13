"""Factory reset — wipe ALL instance state so the next start is a clean first
install (setup wizard, zero accounts). DESTRUCTIVE.

Removes:
  - the SQLite metadata DB (accounts, invites, instance_config, instance_settings,
    pending_uploads, history, …) incl. its -wal/-shm sidecars
  - cache/   (facts, songfacts, transcoded audio, similarity cache, …)
  - media/   (server-mode uploads)
  - frontend/covers/   (extracted album/artist art)
  - EVERY Qdrant collection (acct_*, music_explorer, and any legacy names)

KEEPS (expensive to rebuild, not user data): weights/, .hf-cache/, searxng/.

It does NOT (cannot) clear the browser localStorage, which holds your JWT — so
after running this, clear it too or you'll still look "logged in". See the
printed reminder.

Paths follow the app's own resolution (MUSIX_METADATA_DB, QDRANT_URL), so this
hits the SAME db/qdrant the server uses — run it from the checkout you run the
server from.

Usage:
  1. STOP the server first (Ctrl-C the uvicorn, or `docker compose stop musix`).
  2. python scripts/reset_instance.py            # dry run — shows what it'd do
  3. python scripts/reset_instance.py --yes      # actually wipe
     optional: --keep-qdrant   (leave vectors alone)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
DB_PATH = Path(os.environ.get("MUSIX_METADATA_DB", str(ROOT / "cache" / "metadata.db")))

# Contents wiped; the directory itself is left in place (the app recreates files).
WIPE_DIRS = ["cache", "media", "frontend/covers"]


def _db_files() -> list[Path]:
    return [Path(str(DB_PATH) + s) for s in ("", "-wal", "-shm")]


def _wipe_dir_contents(d: Path) -> int:
    if not d.exists():
        return 0
    n = 0
    for child in sorted(d.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        n += 1
    return n


def _qdrant_collections() -> list[str]:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/collections", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [c["name"] for c in data["result"]["collections"]]
    except Exception as e:
        print(f"  [WARN] could not reach Qdrant at {QDRANT_URL}: {e}")
        return []


def _qdrant_delete(name: str) -> bool:
    req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status in (200, 202)
    except Exception as e:
        print(f"  [WARN] delete {name!r} failed: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Factory-reset the MusiX instance.")
    ap.add_argument("--yes", action="store_true", help="actually wipe (without this it's a dry run)")
    ap.add_argument("--keep-qdrant", action="store_true", help="don't drop Qdrant collections")
    args = ap.parse_args()

    cols = [] if args.keep_qdrant else _qdrant_collections()

    print("=" * 64)
    print("FACTORY RESET — deletes ALL accounts, uploads, vectors, caches.")
    print(f"  repo root : {ROOT}")
    print(f"  metadata  : {DB_PATH}  ({'exists' if DB_PATH.exists() else 'absent'})")
    print(f"  qdrant    : {QDRANT_URL}")
    print(f"  dirs      : {', '.join(WIPE_DIRS)}")
    if not args.keep_qdrant:
        print(f"  qdrant collections to drop: {cols or '(none)'}")
    print("=" * 64)

    if not args.yes:
        print("\nDRY RUN — nothing deleted. Re-run with --yes to wipe.")
        print("Remember to STOP the server first (open DB file would block the wipe).")
        return 0

    # 1. SQLite DB (explicit, in case MUSIX_METADATA_DB points outside cache/)
    print("\n[1/3] Removing metadata DB…")
    for p in _db_files():
        if p.exists():
            try:
                p.unlink()
                print(f"  removed {p.name}")
            except PermissionError:
                print(f"  [LOCKED] {p} — the server is still running. Stop it and retry.")
                return 1

    # 2. State directories
    print("\n[1b] Clearing state dirs…")
    for rel in WIPE_DIRS:
        d = ROOT / rel
        try:
            n = _wipe_dir_contents(d)
            print(f"  cleared {rel}/ ({n} entries)")
        except PermissionError:
            print(f"  [LOCKED] {rel}/ — the server is still running. Stop it and retry.")
            return 1

    # 3. Qdrant
    if args.keep_qdrant:
        print("\n[2/3] Skipping Qdrant (--keep-qdrant).")
    else:
        print("\n[2/3] Dropping Qdrant collections…")
        for name in cols:
            print(f"  {'dropped' if _qdrant_delete(name) else 'FAILED '} {name}")
        if not cols:
            print("  (none)")

    print("\n[3/3] Browser localStorage holds your JWT — clear it too:")
    print("  • app open → DevTools (F12) → Console → run:  localStorage.clear()  → reload")
    print("  • or simply open the app in a fresh incognito window")

    print("\n✓ Done. Start the server and you'll get a clean first-install wizard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
