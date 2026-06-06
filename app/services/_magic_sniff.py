"""Thin wrapper around python-magic that degrades gracefully when libmagic is missing.

Phase C upload path uses this to reject non-audio uploads before any disk
write commits. On a host without libmagic (rare on Linux, occasional on
Windows without python-magic-bin), the sniff returns 'application/octet-stream'
and the upload route falls back to extension-based rejection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import magic  # type: ignore[import-not-found]
    _MAGIC = magic.Magic(mime=True)
except Exception as e:  # pragma: no cover - depends on host libmagic
    logger.warning("[uploads] python-magic unavailable (%s); MIME sniffing degraded", e)
    _MAGIC = None


def sniff_audio_mime(head_bytes: bytes) -> str:
    """Return the detected MIME type for the first ~64 KB of a file, or octet-stream.

    The caller only needs to feed the FIRST chunk read from the upload stream;
    audio magic numbers sit in the first few bytes for all formats we accept.
    """
    if _MAGIC is None or not head_bytes:
        return "application/octet-stream"
    try:
        return _MAGIC.from_buffer(head_bytes) or "application/octet-stream"
    except Exception as e:
        logger.debug("[uploads] sniff failed: %s", e)
        return "application/octet-stream"
