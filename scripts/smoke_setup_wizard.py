"""First-run setup wizard — pre-merge smoke.

1. Babel-parse frontend/index.html      (memory: feedback_smoke_test_must_parse)
2. Identifier grep-gate                 (memory: feedback_jsx_runtime_referror)
3. Live in-process API round-trip on a TEMP metadata DB:
   uninitialized → /instance/config 404 → POST /instance/setup 200 →
   /instance/config 200 → /auth/me with returned token → second setup 409.

Run: python -m scripts.smoke_setup_wizard
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Same argv-neutralizing pattern as scripts/create_owner.py — app imports pull
# in CLAP whose constructor parses sys.argv.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]


def smoke_babel_parse() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from check_jsx import main as check_jsx_main  # scripts/check_jsx.py
    if check_jsx_main() != 0:
        sys.exit(1)


def smoke_ident_grep() -> None:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    required = [
        "function SetupWizard",   # component defined
        "<SetupWizard",           # ...and mounted from Root
        "const WIZ_TIERS",        # slider tiers
        "const WIZ_STAGE_LABELS", # indexing stage labels
        "/instance/setup",        # bootstrap endpoint wired
        "needsSetup",             # Root gating state
        "advanceFromAi",          # step machine fully built (Task 8 ran)
    ]
    missing = [s for s in required if s not in html]
    if missing:
        print(f"[FAIL] missing identifiers in frontend/index.html: {missing}")
        sys.exit(1)
    print("[OK] all wizard identifiers present")


def smoke_live_setup() -> None:
    import sqlite3
    import app.resources.metadata_db as mod
    from app.resources.metadata_db import MetadataDB

    # Temp DB + cross-thread connection (TestClient runs handlers in a
    # worker thread) — same technique as tests/integration/conftest.py.
    tmp = Path(tempfile.mkdtemp()) / "smoke_setup.db"
    mod.DB_PATH = tmp
    MetadataDB._instance = None

    @classmethod
    def _patched_connect(cls):
        if cls._instance is None:
            conn = sqlite3.connect(
                str(mod.DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            cls._instance = conn
        return cls._instance
    MetadataDB._connect = _patched_connect

    from fastapi.testclient import TestClient
    from app.api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 404, f"expected 404 pre-setup, got {r.status_code}"
        r = c.post("/api/v1/instance/setup", json={
            "email": "smoke@example.com", "password": "smoke123", "mode": "sharing",
        })
        assert r.status_code == 200, f"setup failed: {r.status_code} {r.text}"
        token = r.json()["token"]
        assert c.get("/api/v1/instance/config").json() == {"mode": "sharing"}
        me = c.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200, me.text
        r2 = c.post("/api/v1/instance/setup", json={
            "email": "smoke2@example.com", "password": "smoke123", "mode": "server",
        })
        assert r2.status_code == 409, f"expected 409 on second setup, got {r2.status_code}"
    # ASCII arrows: Windows consoles default to cp1251 and choke on U+2192.
    print("[OK] live setup round-trip (404 -> 200 -> 409) on temp DB")


if __name__ == "__main__":
    try:
        smoke_babel_parse()
        smoke_ident_grep()
        smoke_live_setup()
    finally:
        sys.argv = _saved_argv
    print("\nAll setup-wizard smoke checks passed.")
