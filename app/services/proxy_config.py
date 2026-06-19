"""Optional HTTP/HTTPS proxy for *external* outbound requests during indexing.

The proxy value is read from the ``.env`` file **directly** — never from
``os.environ``. This is intentional: docker-compose blanks ``HTTP_PROXY`` in the
container environment, and libraries like ``httpx`` / ``requests`` /
``QdrantClient`` auto-detect ``HTTP_PROXY`` from ``os.environ`` (``trust_env``) —
which would route *every* request through the proxy, including internal traffic
(Qdrant, the local LLM, the local SearXNG). By keeping the proxy value out of the
process environment and applying it only where we explicitly pass it, internal
requests always stay direct.

Usage:
    - ``requests``: ``requests.get(url, proxies=get_proxy())``
    - ``httpx``:    ``httpx.get(url, proxy=get_proxy_url())``  (httpx wants a
      single URL string, *not* the requests-style dict — use the helper).

Supported ``.env`` formats (either ``HTTP_PROXY`` or ``HTTPS_PROXY`` is honoured):
    - Single URL::

        HTTP_PROXY=http://user:pass@proxy.example.com:8080

      Both HTTP and HTTPS traffic route through the same server.

    - Separate values (comma-separated)::

        HTTP_PROXY=http=http://...,https=http://...

    The simple single-URL form is the most common.
"""

from __future__ import annotations

from pathlib import Path

# .env keys we accept as a proxy source, in priority order.
_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY")


def _parse_proxy(raw: str) -> dict | None:
    """Parse a proxy URL string into a requests-style ``{"http":..,"https":..}`` dict."""
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


def _proxy_value_from_env_files() -> str | None:
    """Return the raw proxy value from the first ``.env`` file that declares it.

    Candidate locations, in order:
      1. ``/app/.env``              — the docker-compose bind mount
      2. ``<repo root>/.env``       — cwd-independent (anchored to this file:
                                       services → app → repo root)
      3. ``./.env``                 — current working directory

    The repo-root anchor matters: a bare ``Path(".env")`` resolves against the
    process CWD, which is not guaranteed to be the project root depending on how
    the server is launched. Within a file, ``HTTP_PROXY`` wins over ``HTTPS_PROXY``.
    """
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

        found: dict[str, str] = {}
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in _PROXY_KEYS:
                # Strip optional surrounding quotes commonly seen in .env files.
                found[key] = value.strip().strip('"').strip("'")
        for k in _PROXY_KEYS:
            if found.get(k):
                return found[k]
    return None


def get_proxy() -> dict | None:
    """Return a requests-style proxy dict ``{"http":..,"https":..}`` or ``None``.

    Reads ``HTTP_PROXY`` / ``HTTPS_PROXY`` from a ``.env`` file directly (see
    ``_proxy_value_from_env_files``) — never from ``os.environ`` — so that
    internal Docker/localhost traffic is never auto-proxied.
    """
    return _parse_proxy(_proxy_value_from_env_files() or "")


def get_proxy_url() -> str | None:
    """Return a single proxy URL string for httpx's ``proxy=`` param, or ``None``.

    Prefers the ``https`` mapping but falls back to ``http`` so the key=value
    form ``HTTP_PROXY=http=http://...`` (no ``https=`` entry) still proxies httpx
    calls instead of silently going direct.
    """
    proxies = get_proxy()
    if not proxies:
        return None
    return proxies.get("https") or proxies.get("http")
