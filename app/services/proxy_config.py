"""Optional HTTP/HTTPS proxy for outbound requests during indexing.

Reads ``HTTP_PROXY`` from the ``.env`` file **directly** (not from ``os.environ``).
This is intentional: docker-compose auto-loads ``.env`` into the container
environment, and libraries like ``httpx`` / ``requests`` / ``QdrantClient``
auto-detect ``HTTP_PROXY`` from ``os.environ`` — which would route *every*
request through the proxy, including internal Docker traffic (Qdrant, SearXNG).

By reading the file ourselves and never exporting ``HTTP_PROXY`` into the
container environment, only code that explicitly calls ``get_proxy()`` and
passes ``proxies=...`` uses the proxy. Internal requests stay direct.

Supported formats:
    - Single URL::

        HTTP_PROXY=http://user:pass@proxy.example.com:8080

      Both HTTP and HTTPS traffic route through the same server.

    - Separate values (comma-separated)::

        HTTP_PROXY=http=http://...,https=http://...

    The simple single-URL form is the most common."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_proxy(raw: str) -> dict | None:
    """Parse a proxy URL string into a dict."""
    proxy = raw.strip()
    if not proxy:
        return None

    # key=value dict form
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


def get_proxy() -> dict | None:
    """Return proxy dict for outbound requests, or ``None``.

    Reads ``HTTP_PROXY`` from the ``.env`` file directly — NOT from
    ``os.environ`` — so that internal Docker traffic is unaffected.
    """
    # Try .env at project root (Docker: /app/.env, local: repo root)
    for candidate in (Path("/app/.env"), Path(".env")):
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == "HTTP_PROXY":
                    return _parse_proxy(value)
            break  # found .env, no HTTP_PROXY inside
    return None
