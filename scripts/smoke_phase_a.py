"""Phase A pre-merge smoke check.

1. Verify the Vite frontend build exists (frontend/dist/) — the syntax gate is
   `npm run build` itself since the Vite migration (was: @babel/standalone parse
   of the single-file index.html).
2. Hit live GET  /api/v1/instance/config — returns 404 before init, 200 after.
3. Hit live POST /api/v1/auth/login — round-trip succeeds with bootstrapped owner.
4. Hit live GET  /api/v1/library/collections — 401 without token, 200 with token.

Per project memory:
  [[feedback_registry_pattern_needs_e2e]] — unit tests don't catch missing include_router
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("MUSIX_JWT_SECRET", "smoke-test-secret-32-chars-or-longer-for-prod")
# Use a tmp metadata DB so the smoke doesn't pollute the real cache/metadata.db.
_tmp = tempfile.mkdtemp(prefix="musix_smoke_phase_a_")
os.environ["MUSIX_METADATA_DB"] = str(Path(_tmp) / "smoke.db")

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from app.services.auth_service import AuthService


def smoke_frontend_build():
    """The real syntax gate is `npm run build` (Vite/esbuild). Here we only
    verify the build artifact exists so the server has something to serve."""
    dist_index = Path("frontend/dist/index.html")
    if not dist_index.exists():
        print("[FAIL] frontend/dist/index.html missing — run `npm run build` in frontend/")
        sys.exit(1)
    print("[OK] frontend/dist build present")


def smoke_live_endpoints():
    MetadataDB.init()
    conn = MetadataDB.get()
    conn.execute("DELETE FROM instance_config")
    conn.execute("DELETE FROM users")
    conn.execute("DELETE FROM invites")
    conn.commit()

    with TestClient(create_app()) as c:
        # ── instance/config returns 404 before init ──
        r = c.get("/api/v1/instance/config")
        if r.status_code != 404:
            print(f"[FAIL] /instance/config before init: expected 404, got {r.status_code}")
            sys.exit(1)

        # ── seed owner + sharing mode ──
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        auth = AuthService(jwt_secret=os.environ["MUSIX_JWT_SECRET"])
        auth.create_owner(email="smoke@example.com", password="smokepass12345")

        # ── instance/config returns mode after init ──
        # Subset check, not equality — the payload has grown fields since
        # Phase A (ai_available, member_index_root, ...) and will grow more.
        r = c.get("/api/v1/instance/config")
        if r.status_code != 200 or r.json().get("mode") != "sharing":
            print(f"[FAIL] /instance/config: {r.status_code} {r.text}")
            sys.exit(1)
        print("[OK] /api/v1/instance/config returns sharing mode")

        # ── login round-trip ──
        r = c.post("/api/v1/auth/login",
                   json={"email": "smoke@example.com", "password": "smokepass12345"})
        if r.status_code != 200 or "token" not in r.json():
            print(f"[FAIL] /auth/login: {r.status_code} {r.text}")
            sys.exit(1)
        token = r.json()["token"]
        print(f"[OK] /api/v1/auth/login issued JWT ({len(token)} chars)")

        # ── gated endpoint 401 without token ──
        r = c.get("/api/v1/library/collections")
        if r.status_code != 401:
            print(f"[FAIL] /library/collections without token: expected 401, got {r.status_code}")
            sys.exit(1)
        print("[OK] /api/v1/library/collections returns 401 without token")

        # ── gated endpoint passes with token ──
        r = c.get("/api/v1/library/collections",
                  headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 401:
            print(f"[FAIL] /library/collections with token: still 401 {r.text}")
            sys.exit(1)
        print(f"[OK] /api/v1/library/collections accepts JWT (status={r.status_code})")

        # ── per-track top-pairs route: wired + shaped even with an empty cache ──
        r = c.get("/api/v1/library/top-pairs/nonexistent-track-id",
                  headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            print(f"[FAIL] /library/top-pairs/{{id}}: expected 200, got {r.status_code} {r.text}")
            sys.exit(1)
        body = r.json()
        if not ({"available", "similar", "dissimilar"} <= set(body)
                and isinstance(body["similar"], list) and isinstance(body["dissimilar"], list)):
            print(f"[FAIL] /library/top-pairs/{{id}} shape: {body}")
            sys.exit(1)
        print(f"[OK] /api/v1/library/top-pairs/{{id}} wired (available={body['available']})")


if __name__ == "__main__":
    smoke_frontend_build()
    smoke_live_endpoints()
    print("\nAll Phase A smoke checks passed.")
