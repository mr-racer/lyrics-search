"""Server-mode upload pipeline: stream → quarantine → SHA-verify → atomic promote.

Pure helpers, no FastAPI. The route layer (app/api/routes/library.py) does
the multipart parse and feeds the binary stream into ``write_to_quarantine``.

Disk layout (see spec §7.2):

    media/
    ├── <account_id>/audio/<sha256>.<ext>     ← managed library
    └── _quarantine/<upload_id>.tmp            ← incoming, before SHA verify

Content-addressed by SHA-256 inside the account namespace; same SHA across
two accounts produces two physical files (deliberate — privacy edge case for
cross-account dedup not worth MVP complexity, see spec §7.2).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

# Phase C: 200 MB cap on single upload. ~30 min of FLAC at 16/44.1k mono.
# Env-tunable via MUSIX_MAX_UPLOAD_MB; default chosen for hobby-scale libraries.
_DEFAULT_MAX_MB = int(os.environ.get("MUSIX_MAX_UPLOAD_MB", "200"))
MAX_UPLOAD_BYTES = _DEFAULT_MAX_MB * 1024 * 1024

# 64 KB head sniffed for MIME + buffered for the SHA-256 streaming pass.
HEAD_SNIFF_BYTES = 64 * 1024
STREAM_CHUNK = 1024 * 1024  # 1 MiB

# Mapping from libmagic MIME → preferred extension. M4A is ambiguous (audio/mp4
# also matches MP4 video), so we fall back to original filename's extension.
EXT_BY_MIME: dict[str, str] = {
    "audio/flac":   ".flac",
    "audio/x-flac": ".flac",
    "audio/mpeg":   ".mp3",
    "audio/mp3":    ".mp3",
    "audio/mp4":    ".m4a",   # ambiguous; choose_extension() also checks filename
    "audio/x-m4a":  ".m4a",
    "audio/aac":    ".aac",
    "audio/ogg":    ".ogg",
    "audio/wav":    ".wav",
    "audio/x-wav":  ".wav",
    "audio/opus":   ".opus",
}

# Whitelist of file extensions we accept regardless of MIME sniff result.
_AUDIO_EXTENSIONS = {".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


class UploadRejected(Exception):
    """Base for client-facing upload errors. Route maps to 400."""


class UploadOversize(UploadRejected):
    pass


class UploadWrongType(UploadRejected):
    pass


def choose_extension(mime: str, *, original: str) -> str:
    """Pick the canonical file extension.

    Preference: filename's extension when it's audio-whitelisted (preserves m4a
    vs mp4 distinction libmagic loses); otherwise the MIME → ext mapping;
    otherwise ``.bin``.
    """
    src_ext = Path(original).suffix.lower()
    if src_ext in _AUDIO_EXTENSIONS:
        return src_ext
    if mime in EXT_BY_MIME:
        return EXT_BY_MIME[mime]
    return ".bin"


def write_to_quarantine(
    stream: BinaryIO,
    quarantine_path: Path,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> tuple[str, int, bytes]:
    """Stream the upload to disk, computing SHA-256 incrementally.

    Returns: (sha256_hex, total_bytes, head_bytes_for_mime_sniff).
    Cleans up the quarantine file on any exception (oversize, IO error).
    Raises:
        UploadOversize: if the cumulative bytes exceed ``max_bytes``.
    """
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    head = bytearray()
    total = 0
    try:
        with quarantine_path.open("wb") as f:
            while True:
                chunk = stream.read(STREAM_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadOversize(
                        f"upload exceeds {max_bytes} bytes (got >={total})",
                    )
                hasher.update(chunk)
                f.write(chunk)
                if len(head) < HEAD_SNIFF_BYTES:
                    head.extend(chunk[: HEAD_SNIFF_BYTES - len(head)])
    except BaseException:
        # Any error: scrub the partial write so the cleanup sweeper doesn't
        # see a half-written file masquerading as a healthy quarantine entry.
        quarantine_path.unlink(missing_ok=True)
        raise
    return hasher.hexdigest(), total, bytes(head)


def atomic_promote_to_managed(
    *,
    quarantine_path: Path,
    media_root: Path,
    account_id: str,
    sha256: str,
    extension: str,
) -> Path:
    """Move quarantine file into ``media/<account>/audio/<sha>.<ext>`` atomically.

    If a file already lives at the destination (same account already uploaded
    this SHA earlier), we DROP the quarantine copy — content is identical by
    SHA-256, the existing file is canonical. Cross-platform atomic rename via
    ``Path.replace``.
    """
    dst_dir = media_root / account_id / "audio"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{sha256}{extension}"
    if dst.exists():
        quarantine_path.unlink(missing_ok=True)
        return dst
    # ``Path.replace`` is atomic on the same filesystem (POSIX rename(2)).
    # If media_root and quarantine live on different mounts (rare in our setup),
    # this still succeeds but isn't atomic — we accept that tradeoff for MVP.
    quarantine_path.replace(dst)
    return dst


def media_root_default() -> Path:
    """Project-relative ``media/`` directory. Created on first use."""
    return Path(__file__).resolve().parent.parent.parent / "media"


def quarantine_root_default() -> Path:
    """``media/_quarantine/``. Created on first use."""
    return media_root_default() / "_quarantine"


def sweep_old_quarantine(*, root: Path | None = None, older_than_seconds: int = 86400) -> int:
    """Delete ``*.tmp`` files in the quarantine older than the cutoff. Returns count."""
    import time
    root = root or quarantine_root_default()
    if not root.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    n = 0
    for p in root.glob("*.tmp"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError as e:
            logger.warning("[uploads] sweep failed for %s: %s", p, e)
    return n
