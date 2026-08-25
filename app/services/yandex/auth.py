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

# A blip on the way out must not end a login that waits ten minutes for a human.
# This host reaches the internet over Wi-Fi behind a VPN, and a route rebuild
# there surfaces as a one-second ``[Errno 101] Network is unreachable`` — seen
# against nine different hosts, Yandex among them. Three attempts, because the
# failure lasts about a second and anything that outlives three tries is not a
# blip and should be reported rather than waited out.
_NET_RETRIES = 3
_NET_RETRY_PAUSE = 3.0

# What the CLIENT is told. Codes, not sentences: the frontend already localises
# ``expired`` off the status and has both languages, so a Russian string sent
# from here would reach an English user untranslated. The exception text stays
# in the log, where it helps — a urllib3 stack trace in a login dialog names
# nothing the person can act on, and this failure is usually not theirs.
_REASON_NETWORK = "network"
_REASON_UNKNOWN = "unknown"

_SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _network_error_types() -> tuple:
    """Exception types that mean "the route went away", resolved lazily.

    Lazily for the same reason ``Client`` is imported inside the worker: this
    module sits on import paths that must not pull the Yandex stack in. Only
    genuine transport failures belong here — a malformed response retried three
    times just makes the user wait three times as long for the same answer.
    """
    types_: list = []
    try:
        from yandex_music.exceptions import NetworkError
        types_.append(NetworkError)
    except Exception:  # noqa: BLE001 - an older/absent client is not fatal here
        pass
    try:
        from requests.exceptions import ConnectionError as _ConnErr
        from requests.exceptions import Timeout as _Timeout
        types_.extend((_ConnErr, _Timeout))
    except Exception:  # noqa: BLE001
        pass
    return tuple(types_)


def _network_error(message: str) -> Exception:
    """Build the error the transport raises — the type the tests need to throw."""
    kinds = _network_error_types()
    return kinds[0](message) if kinds else ConnectionError(message)


def _build_yandex_client():
    """A fresh ``yandex_music.Client``. Seam: the tests replace this."""
    from yandex_music import Client

    return Client()


def _device_auth_with_retries(on_code) -> object | None:
    """``device_auth``, surviving a transient loss of route.

    Each attempt asks Yandex for a NEW device code, so ``on_code`` fires again
    and the session's code is replaced — the frontend polls status every two
    seconds and shows the new one, which is the honest thing to do because the
    old device session really is gone.

    The overall deadline is CARRIED, not restarted: three retries that each got
    a fresh ten minutes would turn a ten-minute login into half an hour of a
    dialog the user stopped looking at.
    """
    deadline = time.monotonic() + _DEVICE_AUTH_TIMEOUT
    transient = _network_error_types()

    for attempt in range(1, _NET_RETRIES + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return _build_yandex_client().device_auth(
                on_code=on_code, timeout=remaining)
        except transient as e:
            logger.warning(
                "[yandex/auth] transport failed on attempt %d/%d: %s",
                attempt, _NET_RETRIES, e)
            if attempt == _NET_RETRIES:
                raise
            time.sleep(min(_NET_RETRY_PAUSE,
                           max(0.0, deadline - time.monotonic())))
    return None


def _set(session_id: str, **fields) -> None:
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s is not None:
            s.update(fields)


def _public(session: dict) -> dict:
    """Strip internals (events, account_id) before returning to the route.

    ``reason`` is what the client renders from; ``error`` stays a short,
    non-technical line for anything that reads it raw. The exception text is
    deliberately not in either — it goes to the log.
    """
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "verification_url": session.get("verification_url"),
        "user_code": session.get("user_code"),
        "reason": session.get("reason"),
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
        "reason": None,
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
            token = _device_auth_with_retries(on_code)
            if token is None:
                _set(session_id, status="expired",
                     error="device authorization timed out")
                code_ready.set()
                return

            expires_at = None
            if getattr(token, "expires_in", None):
                expires_at = time.time() + float(token.expires_in)

            # Best-effort: resolve the Yandex uid + display login (non-fatal).
            yandex_uid = None
            yandex_login = None
            try:
                authed = build_client(token.access_token)
                acc = authed.me.account if authed.me else None
                yandex_uid = str(acc.uid) if acc else None
                if acc is not None:
                    yandex_login = getattr(acc, "login", None) or getattr(acc, "display_name", None)
            except Exception:
                logger.debug("[yandex/auth] could not resolve yandex uid", exc_info=True)

            token_store.save_token(
                account_id,
                access_token=token.access_token,
                refresh_token=getattr(token, "refresh_token", None),
                expires_at=expires_at,
                yandex_uid=yandex_uid,
                yandex_login=yandex_login,
            )
            _set(session_id, status="authorized", yandex_uid=yandex_uid)
        except Exception as e:  # noqa: BLE001 - report to the poller, don't crash
            logger.warning("[yandex/auth] device_auth failed: %s", e, exc_info=True)
            transient = _network_error_types()
            is_network = bool(transient) and isinstance(e, transient)
            _set(
                session_id,
                status="error",
                reason=_REASON_NETWORK if is_network else _REASON_UNKNOWN,
                # Short and non-technical: the client localises from ``reason``,
                # and ``str(e)`` here is what put a urllib3 stack trace in the
                # login dialog.
                error=("could not reach Yandex" if is_network
                       else "Yandex login failed"),
            )
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
