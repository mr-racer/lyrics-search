"""Audio streaming preparation — codec detection + on-the-fly ALAC→FLAC transcoding.

ALAC (Apple Lossless) is not natively decoded by Chrome, Edge, or Firefox —
only Safari supports it. To make ALAC-in-m4a files playable cross-browser
while preserving bit-exact lossless quality, we transcode them to FLAC on
the first request and cache the result on disk.

FLAC, MP3, AAC-in-m4a, OGG, WAV, OPUS are served as-is.

Public API:
  get_streamable_path(account_id, track_id, file_path) -> (Path, mime_type)
  drop_transcoded_for_tracks(account_id, track_ids: Iterable[str]) -> int
  get_cached_source / put_cached_source / drop_source_for_tracks / drop_account_sources
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Co-located with cache/metadata.db so deployment / backup conventions match.
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "transcoded"

# MIME types — single source of truth; search.py imports this map.
AUDIO_CONTENT_TYPES: dict[str, str] = {
    ".mp3":  "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a":  "audio/mp4",
    ".aac":  "audio/aac",
    ".ogg":  "audio/ogg",
    ".wav":  "audio/wav",
    ".opus": "audio/opus",
}

_FLAC_MIME = "audio/flac"

# Per-track locks so two concurrent first-time requests for the same ALAC
# track don't spawn two ffmpeg processes writing the same output file.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ── Hot-path caches ──────────────────────────────────────────────────────────
# <audio> issues many Range requests per track; without these caches every one
# of them re-did a synchronous Qdrant retrieve (and, for .m4a, a blocking
# ffprobe) on the event loop — stalling byte delivery for ALL listeners.

# (collection, track_id) -> (expires_at_monotonic, file_path, audio_codec|None)
_SOURCE_TTL_SECONDS = 600.0
_SOURCE_CACHE_MAX = 20_000
_source_cache: dict[tuple[str, str], tuple[float, str, str | None]] = {}

# str(path) -> ((mtime, size), codec|None) — ffprobe result, keyed by file identity
_CODEC_CACHE_MAX = 8_192
_codec_cache: dict[str, tuple[tuple[float, int], str | None]] = {}


def get_cached_source(collection: str, track_id: str) -> tuple[str, str | None] | None:
    """Return (file_path, audio_codec) for a track if cached and fresh, else None."""
    entry = _source_cache.get((collection, track_id))
    if entry is None:
        return None
    expires_at, file_path, codec = entry
    if time.monotonic() >= expires_at:
        _source_cache.pop((collection, track_id), None)
        return None
    return file_path, codec


def put_cached_source(collection: str, track_id: str, file_path: str, codec: str | None) -> None:
    if len(_source_cache) >= _SOURCE_CACHE_MAX:
        _source_cache.clear()
    _source_cache[(collection, track_id)] = (
        time.monotonic() + _SOURCE_TTL_SECONDS, file_path, codec,
    )


def drop_source_for_tracks(collection: str, track_ids: Iterable[str]) -> None:
    for tid in track_ids:
        _source_cache.pop((collection, tid), None)


def drop_account_sources(collection: str) -> None:
    for key in [k for k in _source_cache if k[0] == collection]:
        _source_cache.pop(key, None)


def _content_type(file_path: Path) -> str:
    return AUDIO_CONTENT_TYPES.get(file_path.suffix.lower(), "application/octet-stream")


def _detect_codec_ffprobe(file_path: Path) -> str | None:
    """Run ffprobe to read the audio codec name. Returns lowercase string or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(file_path),
            ],
            capture_output=True, text=True, timeout=10, check=True,
        )
        codec = (result.stdout or "").strip().lower()
        return codec or None
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning("[audio_streaming] ffprobe failed for %s: %s", file_path.name, e)
        return None


async def _needs_alac_transcode(file_path: Path, codec_hint: str | None = None) -> bool:
    """Is this an .m4a file using the ALAC codec? (AAC-in-m4a plays everywhere.)

    ``codec_hint`` comes from the Qdrant payload (written at index time by
    mutagen: "alac" or "mp4a.40.2") — when present, no ffprobe runs at all.
    Legacy libraries without the payload field fall back to ffprobe, executed
    in a thread and cached per (mtime, size) so it runs once per file, not once
    per Range request.
    """
    if file_path.suffix.lower() != ".m4a":
        return False
    if codec_hint:
        return codec_hint.lower().startswith("alac")

    try:
        st = file_path.stat()
        identity = (st.st_mtime, st.st_size)
    except OSError:
        identity = None

    key = str(file_path)
    if identity is not None:
        cached = _codec_cache.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1] == "alac"

    codec = await asyncio.get_running_loop().run_in_executor(
        None, _detect_codec_ffprobe, file_path,
    )
    if identity is not None:
        if len(_codec_cache) >= _CODEC_CACHE_MAX:
            _codec_cache.clear()
        _codec_cache[key] = (identity, codec)
    return codec == "alac"


def _transcode_alac_to_flac(src: Path, dst: Path) -> bool:
    """Lossless→lossless transcode. Returns True on success.

    -compression_level 5 is the FLAC default; trades CPU for file size.
    -map_metadata 0 preserves tags. Writes to a .tmp first then atomic rename
    so partial writes (crash, kill) don't leave a corrupt cache entry.
    """
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y",
                "-i", str(src),
                "-map", "0:a:0",
                "-c:a", "flac",
                "-compression_level", "5",
                "-map_metadata", "0",
                "-f", "flac",
                str(tmp),
            ],
            check=True, capture_output=True, timeout=180,
        )
        tmp.replace(dst)
        return True
    except FileNotFoundError:
        logger.error("[audio_streaming] ffmpeg not installed — cannot transcode ALAC")
        return False
    except subprocess.TimeoutExpired:
        logger.error("[audio_streaming] ffmpeg timeout for %s", src.name)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        logger.error("[audio_streaming] ffmpeg failed for %s: %s", src.name, stderr)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _cache_path(account_id: str, track_id: str) -> Path:
    """Per-account, content-addressed transcoded cache path (Phase B §6.6).

    Namespacing by ``account_id`` closes the cross-account leak: ``track_id`` is
    ``sha256(file_path)[:16]``, so two accounts whose libraries share a path
    would otherwise collide on a single flat ``cache/transcoded/<id>.flac`` and
    one would serve the other's audio. Callers pass the owning collection name
    as ``account_id`` (Phase D renames collections to ``acct_<id>``, at which
    point the key is literally per-account).
    """
    return _CACHE_DIR / account_id / f"{track_id}.flac"


async def get_streamable_path(
    account_id: str,
    track_id: str,
    file_path: Path,
    codec: str | None = None,
) -> tuple[Path, str]:
    """Resolve the path FastAPI should hand to FileResponse, plus its mime type.

    - Non-ALAC files: returned as-is with their native mime type.
    - ALAC m4a files: transcoded to FLAC (cached at
      ``cache/transcoded/<account_id>/<track_id>.flac``) and the cache path is
      returned with audio/flac. First call blocks ~1–2s; subsequent calls are
      instant. Concurrent first-calls are serialized per (account_id, track_id)
      so only one ffmpeg runs.

    ``codec``: optional hint from the track's Qdrant payload (index-time
    mutagen read) — skips codec detection entirely when present.
    """
    if not await _needs_alac_transcode(file_path, codec_hint=codec):
        return file_path, _content_type(file_path)

    cached = _cache_path(account_id, track_id)
    cached.parent.mkdir(parents=True, exist_ok=True)

    if cached.exists():
        return cached, _FLAC_MIME

    # Lock key includes account_id so two different accounts transcoding the
    # same (rare) track_id aren't serialised against each other.
    lock = _locks[f"{account_id}::{track_id}"]
    async with lock:
        # Re-check after acquiring the lock — another request may have just finished.
        if cached.exists():
            return cached, _FLAC_MIME

        logger.info("[audio_streaming] transcoding ALAC→FLAC: %s (acct=%s)", file_path.name, account_id)
        # ffmpeg is blocking; run in default thread executor so the event loop
        # stays responsive for other requests.
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _transcode_alac_to_flac, file_path, cached,
        )

    if not ok or not cached.exists():
        # Fall back to serving the original ALAC file. It won't play in
        # Chrome/Firefox but at least the HTTP response is honest, and Safari
        # users still get audio.
        logger.warning(
            "[audio_streaming] transcoding failed for %s — serving original ALAC",
            file_path.name,
        )
        return file_path, _content_type(file_path)

    return cached, _FLAC_MIME


def drop_transcoded_for_tracks(account_id: str, track_ids: Iterable[str]) -> int:
    """Remove cached transcoded files for given track ids in ``account_id`` only.

    Scoped to the account's namespace so purging one account's collection never
    touches another account's cache. Returns count deleted.
    """
    account_dir = _CACHE_DIR / account_id
    if not account_dir.exists():
        return 0
    n = 0
    for tid in track_ids:
        p = account_dir / f"{tid}.flac"
        if p.exists():
            try:
                p.unlink()
                n += 1
            except OSError as e:
                logger.warning("[audio_streaming] failed to remove %s: %s", p, e)
    return n


def drop_account_transcodes(account_id: str) -> bool:
    """Remove an account's ENTIRE transcoded-cache directory
    (``cache/transcoded/<account_id>/``). Used when deleting an account outright,
    where enumerating individual track ids (e.g. via Qdrant) may be impossible if
    the collection is already gone. Scoped to the account's own namespace, so it
    can never touch another account's cache. Returns True if a directory was
    removed."""
    import shutil
    account_dir = _CACHE_DIR / account_id
    if not account_dir.exists():
        return False
    try:
        shutil.rmtree(account_dir)
        return True
    except OSError as e:
        logger.warning("[audio_streaming] failed to remove cache dir %s: %s", account_dir, e)
        return False
