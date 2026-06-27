"""Device-flow login for Yandex Music.

``Client.device_auth`` is *blocking* — it polls Yandex until the user enters the
code (or it times out). We therefore run it on a daemon thread per session and
expose the state to HTTP pollers via an in-memory registry (sessions are
short-lived; only the final encrypted token is persisted, by token_store).

Flow:
    start_session(account_id)
        → spawns the device_auth thread, waits briefly for the `on_code`
          callback, returns {session_id, verification_url, user_code, status}
    get_session(session_id)
        → {status: starting|pending|authorized|expired|error, ...}
    on authorize the worker thread saves the token and flips status to 'authorized'
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from app.services.yandex import token_store
from app.services.yandex.client_factory import build_client

logger = logging.getLogger(__name__)

# How long to wait (s) for the device code before returning from start_session.
_CODE_WAIT_TIMEOUT = 20.0
# Overall device-auth timeout (s) — how long the user has to confirm.
_DEVICE_AUTH_TIMEOUT = 600.0

_SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _set(session_id: str, **fields) -> None:
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is not None:
            s.update(fields)


def _public(session: dict) -> dict:
    """Strip internals (events, account_id) before returning to the route."""
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "verification_url": session.get("verification_url"),
        "user_code": session.get("user_code"),
        "error": session.get("error"),
    }


def start_session(account_id: str) -> dict:
    """Start a device-flow login for ``account_id`` and return the code to show."""
    session_id = uuid.uuid4().hex
    code_ready = threading.Event()
    session = {
        "session_id": session_id,
        "account_id": account_id,
        "status": "starting",
        "verification_url": None,
        "user_code": None,
        "error": None,
        "created_at": time.time(),
    }
    with _LOCK:
        _SESSIONS[session_id] = session

    def on_code(code) -> None:
        _set(
            session_id,
            verification_url=getattr(code, "verification_url", None),
            user_code=getattr(code, "user_code", None),
            status="pending",
        )
        code_ready.set()

    def run() -> None:
        try:
            from yandex_music import Client

            client = Client()
            token = client.device_auth(
                on_code=on_code, timeout=_DEVICE_AUTH_TIMEOUT,
            )
            if token is None:
                _set(session_id, status="expired",
                     error="device authorization timed out")
                code_ready.set()
                return

            expires_at = None
            if getattr(token, "expires_in", None):
                expires_at = time.time() + float(token.expires_in)

            # Best-effort: resolve the Yandex uid for display (non-fatal).
            yandex_uid = None
            try:
                authed = build_client(token.access_token)
                yandex_uid = str(authed.me.account.uid) if authed.me and authed.me.account else None
            except Exception:
                logger.debug("[yandex/auth] could not resolve yandex uid", exc_info=True)

            token_store.save_token(
                account_id,
                access_token=token.access_token,
                refresh_token=getattr(token, "refresh_token", None),
                expires_at=expires_at,
                yandex_uid=yandex_uid,
            )
            _set(session_id, status="authorized", yandex_uid=yandex_uid)
        except Exception as e:  # noqa: BLE001 - report to the poller, don't crash
            logger.warning("[yandex/auth] device_auth failed: %s", e, exc_info=True)
            _set(session_id, status="error", error=str(e))
            code_ready.set()

    threading.Thread(target=run, name=f"ym-auth-{session_id[:8]}", daemon=True).start()
    # Wait briefly so the HTTP response already carries the code the user types.
    code_ready.wait(timeout=_CODE_WAIT_TIMEOUT)
    with _LOCK:
        return _public(_SESSIONS[session_id])


def get_session(session_id: str) -> dict | None:
    with _LOCK:
        s = _SESSIONS.get(session_id)
        return _public(s) if s else None


def cleanup_sessions(older_than_seconds: float = 3600.0) -> int:
    """Drop finished/old sessions from the registry. Returns count removed."""
    cutoff = time.time() - older_than_seconds
    removed = 0
    with _LOCK:
        for sid in list(_SESSIONS):
            s = _SESSIONS[sid]
            terminal = s["status"] in ("authorized", "expired", "error")
            if s["created_at"] < cutoff or terminal and s["created_at"] < time.time() - 60:
                del _SESSIONS[sid]
                removed += 1
    return removed
