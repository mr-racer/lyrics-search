"""Cover art extraction and storage (content-addressed by SHA256).

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.id3 import APIC
from PIL import Image

logger = logging.getLogger(__name__)

# Frontend serves /covers/ from this directory.
COVERS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "covers"


def normalize_image(image_data: bytes) -> bytes:
    """Normalize image to consistent RGB/JPEG format.

    Handles CMYK, RGBA, palette modes, and strips ICC profiles
    for consistent appearance across all extracted cover art.
    """
    img = Image.open(io.BytesIO(image_data))

    # Convert to RGB for consistent output (handles CMYK, RGBA, palette, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=90, icc_profile=None)
    return output.getvalue()


def extract_cover_art(filepath: str) -> tuple[bytes, str] | None:
    """Extract embedded cover art from an audio file.

    Returns (image_bytes, "jpg") or None if no artwork is found.
    All extracted images are normalized to JPEG via normalize_image().
    Supported formats: MP3 (APIC), FLAC (embedded picture), M4A (covr).
    """
    audio = MutagenFile(filepath)
    if audio is None:
        return None

    # MP3 — APIC frames
    if hasattr(audio, "tags") and audio.tags is not None:
        for tag in audio.tags.values():
            if isinstance(tag, APIC):
                try:
                    return normalize_image(tag.data), "jpg"
                except Exception:
                    return tag.data, ("jpg" if tag.mime == "image/jpeg" else "png")

    # FLAC — embedded pictures
    if hasattr(audio, "pictures") and audio.pictures:
        pic = audio.pictures[0]
        try:
            return normalize_image(pic.data), "jpg"
        except Exception:
            ext = "png" if "png" in pic.mime else "jpg"
            return pic.data, ext

    # M4A/MP4 — covr atom (MP4Cover object from mutagen)
    if hasattr(audio, "get") and "covr" in audio:
        covers = audio["covr"]
        if covers:
            cover = covers[0]
            # MP4Cover.raw_data gives the actual image bytes (no header padding)
            raw = cover.raw_data if hasattr(cover, "raw_data") else bytes(cover)
            try:
                return normalize_image(raw), "jpg"
            except Exception:
                ext = "png" if raw.startswith(b'\x89PNG') else "jpg"
                return raw, ext

    return None


def save_cover_art(filepath: str, track_id: str) -> str | None:
    """Extract cover art and save to disk (content-addressed, deduplicated by SHA256).

    Returns relative URL path like '/covers/a3f2c1...jpg' or None.
    If an identical cover already exists on disk, returns the existing path — no duplicate write.
    """
    result = extract_cover_art(filepath)
    if result is None:
        return None

    image_data, ext = result
    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    # Content-addressed filename: first 16 hex chars of SHA256
    content_hash = hashlib.sha256(image_data).hexdigest()[:16]
    dest = COVERS_DIR / f"{content_hash}.{ext}"

    if not dest.exists():
        dest.write_bytes(image_data)

    return f"/covers/{content_hash}.{ext}"
