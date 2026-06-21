"""Download one Yandex track and embed Yandex metadata into the file.

Codec preference (spec §7.1): **flac → mp3@320 → aac**. FLAC is best-effort —
the public ``get_download_info`` usually only lists mp3/aac, so the picker simply
falls through. After download we write tags + embed the Yandex cover and lyrics
so the *unchanged* indexing pipeline (which reads tags, extracts embedded art via
cover_art.extract_cover_art, and prefers embedded lyrics) consumes them directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Order of preference. Within a codec we pick the highest available bitrate
# (mp3 capped at 320 — Yandex's max for the lossy path).
_CODEC_PREFERENCE = ("flac", "mp3", "aac")
_MP3_MAX_KBPS = 320

# codec → managed-library extension. AAC is delivered in an MP4 container.
_EXT_BY_CODEC = {"flac": ".flac", "mp3": ".mp3", "aac": ".m4a"}


def pick_best_download_info(infos: list):
    """Choose the best non-preview DownloadInfo by codec preference, then bitrate.

    Returns the chosen ``DownloadInfo`` or None (preview-only / nothing usable —
    e.g. no Plus subscription).
    """
    usable = [i for i in (infos or []) if not getattr(i, "preview", False)]
    if not usable:
        return None
    for codec in _CODEC_PREFERENCE:
        candidates = [i for i in usable if (i.codec or "").lower() == codec]
        if codec == "mp3":
            candidates = [i for i in candidates if (i.bitrate_in_kbps or 0) <= _MP3_MAX_KBPS]
        if candidates:
            return max(candidates, key=lambda i: i.bitrate_in_kbps or 0)
    # Unknown codec(s) only — take the highest bitrate so we still get audio.
    return max(usable, key=lambda i: i.bitrate_in_kbps or 0)


# ─── Metadata extraction from a Yandex Track ─────────────────────────────────

def track_tags(track) -> dict:
    """Flatten a Yandex ``Track`` into the tag dict we write into the file."""
    artists = ", ".join(a.name for a in (track.artists or []) if getattr(a, "name", None))
    album = track.albums[0] if track.albums else None
    track_no = None
    if album and getattr(album, "track_position", None):
        track_no = getattr(album.track_position, "index", None)
    return {
        "title": track.title,
        "artist": artists,
        "album": getattr(album, "title", None) if album else None,
        "year": getattr(album, "year", None) if album else None,
        "genre": getattr(album, "genre", None) if album else None,
        "track_number": track_no,
    }


def _fetch_lyrics(track) -> str | None:
    try:
        tl = track.get_lyrics("TEXT")
        if tl is None:
            return None
        text = tl.fetch_lyrics()
        return text.strip() if text else None
    except Exception:
        logger.debug("[yandex/downloader] lyrics fetch failed", exc_info=True)
        return None


def _fetch_cover(track) -> bytes | None:
    if not getattr(track, "cover_uri", None):
        return None
    try:
        return track.download_cover_bytes(size="600x600")
    except Exception:
        logger.debug("[yandex/downloader] cover fetch failed", exc_info=True)
        return None


# ─── Tag writers (per container) ─────────────────────────────────────────────

def _write_flac(path: str, tags: dict, cover: bytes | None, lyrics: str | None) -> None:
    from mutagen.flac import FLAC, Picture
    from mutagen.id3 import PictureType

    audio = FLAC(path)
    if tags.get("title"):  audio["title"] = tags["title"]
    if tags.get("artist"): audio["artist"] = tags["artist"]
    if tags.get("album"):  audio["album"] = tags["album"]
    if tags.get("year"):   audio["date"] = str(tags["year"])
    if tags.get("genre"):  audio["genre"] = tags["genre"]
    if tags.get("track_number"): audio["tracknumber"] = str(tags["track_number"])
    if lyrics:             audio["lyrics"] = lyrics
    if cover:
        pic = Picture()
        pic.type = PictureType.COVER_FRONT
        pic.mime = "image/jpeg"
        pic.data = cover
        audio.clear_pictures()
        audio.add_picture(pic)
    audio.save()


def _write_mp3(path: str, tags: dict, cover: bytes | None, lyrics: str | None) -> None:
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, TRCK, APIC, USLT
    from mutagen.id3 import ID3NoHeaderError

    try:
        audio = ID3(path)
    except ID3NoHeaderError:
        audio = ID3()
    if tags.get("title"):  audio.add(TIT2(encoding=3, text=tags["title"]))
    if tags.get("artist"): audio.add(TPE1(encoding=3, text=tags["artist"]))
    if tags.get("album"):  audio.add(TALB(encoding=3, text=tags["album"]))
    if tags.get("year"):   audio.add(TDRC(encoding=3, text=str(tags["year"])))
    if tags.get("genre"):  audio.add(TCON(encoding=3, text=tags["genre"]))
    if tags.get("track_number"): audio.add(TRCK(encoding=3, text=str(tags["track_number"])))
    if lyrics:             audio.add(USLT(encoding=3, lang="und", desc="", text=lyrics))
    if cover:
        audio.delall("APIC")
        audio.add(APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover))
    audio.save(path)


def _write_mp4(path: str, tags: dict, cover: bytes | None, lyrics: str | None) -> None:
    from mutagen.mp4 import MP4, MP4Cover

    audio = MP4(path)
    if tags.get("title"):  audio["\xa9nam"] = tags["title"]
    if tags.get("artist"): audio["\xa9ART"] = tags["artist"]
    if tags.get("album"):  audio["\xa9alb"] = tags["album"]
    if tags.get("year"):   audio["\xa9day"] = str(tags["year"])
    if tags.get("genre"):  audio["\xa9gen"] = tags["genre"]
    if tags.get("track_number"): audio["trkn"] = [(int(tags["track_number"]), 0)]
    if lyrics:             audio["\xa9lyr"] = lyrics
    if cover:
        audio["covr"] = [MP4Cover(cover, imageformat=MP4Cover.FORMAT_JPEG)]
    audio.save()


def _embed_metadata(path: Path, codec: str, track) -> None:
    """Write tags + cover + lyrics into the just-downloaded file (best-effort)."""
    tags = track_tags(track)
    cover = _fetch_cover(track)
    lyrics = _fetch_lyrics(track)
    suffix = path.suffix.lower()
    try:
        if suffix == ".flac":
            _write_flac(str(path), tags, cover, lyrics)
        elif suffix == ".mp3":
            _write_mp3(str(path), tags, cover, lyrics)
        elif suffix in (".m4a", ".aac", ".mp4"):
            _write_mp4(str(path), tags, cover, lyrics)
    except Exception:
        # Tags are nice-to-have; the enrichment stage can still backfill from the
        # catalog. Never fail a download because tagging hit a container quirk.
        logger.warning("[yandex/downloader] tag embed failed for %s", path, exc_info=True)


# ─── Public entry point ──────────────────────────────────────────────────────

def download_track(track, dest_dir: Path, *, throttle=None) -> dict:
    """Download ``track`` into ``dest_dir`` and embed Yandex metadata.

    Returns ``{ok, path, codec, bitrate}`` on success, or
    ``{ok: False, reason}`` when the track has no downloadable (full) audio.
    """
    if throttle is not None:
        throttle.wait()

    if getattr(track, "available", True) is False:
        return {"ok": False, "reason": "track not available in catalog/region"}

    try:
        infos = track.get_download_info()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"download info failed: {e}"}

    info = pick_best_download_info(infos)
    if info is None:
        return {"ok": False, "reason": "no full audio (preview only / no Plus subscription)"}

    codec = (info.codec or "mp3").lower()
    ext = _EXT_BY_CODEC.get(codec, ".mp3")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Temp name; the importer renames to <sha>.<ext> after hashing.
    safe = f"{track.id}{ext}"
    dest = dest_dir / safe

    try:
        info.download(str(dest))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"download failed: {e}"}

    _embed_metadata(dest, codec, track)
    return {
        "ok": True,
        "path": dest,
        "codec": codec,
        "bitrate": info.bitrate_in_kbps,
    }
