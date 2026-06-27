"""Externally-reachable base URL for building absolute links (invite links).

The admin's browser origin is unreliable for this: behind a reverse proxy the
site is exposed under a different domain than the LAN address the owner uses, so
``window.location.origin`` produces a link members can't open. We therefore
compute the base URL on the server, where the real config lives:

  1. ``PUBLIC_BASE_URL`` — explicit, wins. Read from ``os.environ`` first (so a
     docker-compose ``environment:`` entry works) and from a ``.env`` file
     directly as a fallback (mirrors ``proxy_config``: the bind-mounted
     ``/app/.env`` / repo-root ``.env`` survive even when the value isn't in the
     process environment).
  2. Otherwise the machine's LAN IP, as ``http://{ip}:{port}`` — so a freshly
     set-up server on a home network still hands out a working link
     (``http://192.168.x.y:8000/...``).

Callers append ``/#/register?invite=...``; the returned base has no trailing
slash.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional


def _env_value(key: str) -> Optional[str]:
    """Resolve ``key`` from ``os.environ`` then from the first ``.env`` file that
    declares it. Candidate ``.env`` order mirrors ``proxy_config``:
    ``/app/.env`` (docker bind mount), repo-root ``.env`` (cwd-independent),
    ``./.env`` (cwd). Surrounding quotes are stripped; comments are skipped.
    Never mutates ``os.environ``."""
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        return val.strip()

    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    seen: set[str] = set()
    for candidate in (Path("/app/.env"), repo_root_env, Path(".env")):
        try:
            key_path = str(candidate.resolve())
        except OSError:
            continue
        if key_path in seen or not candidate.is_file():
            continue
        seen.add(key_path)
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    return None


def _detect_lan_ip() -> str:
    """Best-effort LAN IP of this machine. Opens a UDP socket and 'connects' to a
    public address — no packets are sent; the OS just picks the egress interface,
    whose local address we read back. Falls back to loopback on any error."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _app_port() -> str:
    """Port to embed in the LAN-IP fallback URL: ``APP_PORT`` > ``PORT`` > 8000."""
    return _env_value("APP_PORT") or _env_value("PORT") or "8000"


def public_base_url() -> str:
    """Externally-reachable base URL, no trailing slash. ``PUBLIC_BASE_URL`` wins;
    otherwise ``http://{lan_ip}:{port}``."""
    explicit = _env_value("PUBLIC_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    return f"http://{_detect_lan_ip()}:{_app_port()}"


def invite_url(code: str) -> str:
    """Full registration URL for an invite ``code``. The register form parses
    ``#/register?invite=<code>`` from the hash."""
    return f"{public_base_url()}/#/register?invite={code}"
