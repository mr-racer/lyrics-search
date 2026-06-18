"""Optional HTTP/HTTPS proxy configuration for outbound requests during indexing.

Reads ``HTTP_PROXY`` from the environment (or ``os.environ`` fallback). When set,
returns a dict suitable for both ``requests.get(proxy=...)`` and
``httpx.get(proxy=...)``. When unset or empty, returns ``None`` so every call
falls back to direct networking — no proxy at all.

Supported formats:
    - Single URL::

        HTTP_PROXY=http://user:pass@proxy.example.com:8080

      Both HTTP and HTTPS traffic route through the same server.

    - Separate values (comma-separated, ``http=https=...`` also works)::

        HTTP_PROXY=http=http://...,https=http://...

    The simple single-URL form is the most common — it covers a standard
    transparent proxy or an auth-enabled one with user:pass baked in."""

from __future__ import annotations

import os


def get_proxy() -> dict | None:
    """Return ``{"http": ..., "https": ...}`` when a proxy env var is set, else ``None``.

    Checks ``HTTP_PROXY`` first, then falls back to the conventional lower-case
    ``http_proxy`` (honoured by many libraries out of the box — we mirror it so
    Docker / docker-compose users can set either).
    """
    raw = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
    proxy = raw.strip()

    if not proxy:
        return None

    # If the value already looks like a key=value dict, pass it through.
    if "=" in proxy and (proxy.startswith("http=") or proxy.startswith("https=")):
        result = {}
        for part in proxy.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip()
        return result

    # Single URL — apply to both http and https.
    return {"http": proxy, "https": proxy}
