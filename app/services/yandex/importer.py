"""Orchestrate the download phase of a Yandex import.

``download_source`` is the testable core: authenticate from the stored token,
resolve the source into tracks, dedup against prior imports, then download (2
workers, 0.5 s throttle), embed metadata, promote each file into the account's
managed library, and register a ``pending_uploads`` row. It returns the list of
``upload_id``s plus a report of skipped tracks.

The caller (``LibraryService.enqueue_yandex_import``) then hands those upload_ids
to the EXISTING ``enqueue_upload_indexing`` flow — so indexing is byte-for-byte
the normal upload pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from app.resources.metadata_db import MetadataDB
from app.services import uploads_service
from app.services.yandex import downloader, playlists, token_store
from app.services.yandex.client_factory import Throttle, build_client

logger = logging.getLogger(__name__)

_DOWNLOAD_WORKERS = 2
_THROTTLE_INTERVAL = 0.5


class YandexNotLinkedError(RuntimeError):
    """The account has no stored Yandex token — the user must log in first."""


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _track_label(track) -> tuple[str, str]:
    artist = ", ".join(a.name for a in (track.artists or []) if getattr(a, "name", None))
    return artist, (track.title or "")


def _safe_name(artist: str, title: str, ext: str) -> str:
    base = f"{artist} - {title}".strip(" -") or "track"
    base = re.sub(r'[<>:"/\\|?*]', "_", base)[:120]
    return f"{base}{ext}"


def download_source(
    account_id: str,
    source,
    *,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[list[str], dict]:
    """Download every (not-yet-imported) track of a SINGLE ``source``.

    Thin wrapper around :func:`download_sources` kept for callers/tests that pass
    one source ("likes" | {"kind": N}).
    """
    return download_sources(account_id, [source], on_progress=on_progress)


def _resolve_merged_tracks(client, sources) -> list:
    """Resolve every source and merge into one track list, deduped by track id.

    A track that is both liked and in a selected playlist is downloaded once;
    first occurrence wins (order-preserving across the sources list).
    """
    seen: set[str] = set()
    merged: list = []
    for source in sources:
        for track in playlists.resolve_tracks(client, source):
            tid = str(track.id)
            if tid in seen:
                continue
            seen.add(tid)
            merged.append(track)
    return merged


def download_sources(
    account_id: str,
    sources: list,
    *,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[list[str], dict]:
    """Download every (not-yet-imported) track across ``sources`` for ``account_id``.

    ``sources`` is a list whose items are each ``"likes"`` or ``{"kind": <int>}``.
    Tracks are merged with dedup by track id, so selecting «Мне нравится» plus a
    playlist that overlaps it downloads each track once.

    Returns ``(upload_ids, report)`` where report is
    ``{total, downloaded, already, skipped:[{artist,title,reason}]}``.
    Raises YandexNotLinkedError if the account has no stored token.
    """
    blob = token_store.load_token(account_id)
    if not blob or not blob.get("access_token"):
        raise YandexNotLinkedError("yandex account not linked")

    client = build_client(blob["access_token"])
    tracks = _resolve_merged_tracks(client, sources)

    already = MetadataDB.get_imported_yandex_track_ids(account_id)
    not_imported = [t for t in tracks if str(t.id) not in already]

    throttle = Throttle(_THROTTLE_INTERVAL)
    media_root = uploads_service.media_root_default()
    tmp_dir = uploads_service.quarantine_root_default()

    upload_ids: list[str] = []
    skipped: list[dict] = []

    MetadataDB.init()

    # Drop tracks with no title or artist BEFORE downloading — the indexing
    # pipeline keys every track on "Artist — Title" and can do nothing with an
    # unidentifiable file, so spending bandwidth on it is wasted. Yandex catalog
    # entries almost always carry both; this is a safety net for podcast/edge
    # items. Recorded as 'skipped' (distinct from 'already' imported).
    pending = []
    for t in not_imported:
        artist, title = _track_label(t)
        if not artist.strip() or not title.strip():
            skipped.append({"artist": artist, "title": title, "reason": "no title/artist"})
            MetadataDB.upsert_yandex_import(
                account_id=account_id, yandex_track_id=str(t.id),
                status="skipped", reason="no title/artist",
            )
            continue
        pending.append(t)

    total = len(pending)
    done = 0

    def _work(track):
        return track, downloader.download_track(track, tmp_dir, throttle=throttle)

    workers = min(_DOWNLOAD_WORKERS, total) or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_work, t): t for t in pending}
        for future in as_completed(futures):
            track, res = future.result()
            done += 1
            artist, title = _track_label(track)
            ytid = str(track.id)

            if not res.get("ok"):
                reason = res.get("reason", "download failed")
                skipped.append({"artist": artist, "title": title, "reason": reason})
                MetadataDB.upsert_yandex_import(
                    account_id=account_id, yandex_track_id=ytid,
                    status="skipped", reason=reason,
                )
                if on_progress:
                    on_progress(done, total, f"[{done}/{total}] ✗ {artist} — {title}")
                continue

            try:
                path: Path = res["path"]
                sha = _sha256_of(path)
                ext = path.suffix.lower()
                size = path.stat().st_size
                final = uploads_service.atomic_promote_to_managed(
                    quarantine_path=path, media_root=media_root,
                    account_id=account_id, sha256=sha, extension=ext,
                )
                upload_id = MetadataDB.create_pending_upload(
                    account_id=account_id, sha256=sha,
                    original_filename=_safe_name(artist, title, ext),
                    size_bytes=size, storage_path=str(final),
                )
                MetadataDB.upsert_yandex_import(
                    account_id=account_id, yandex_track_id=ytid,
                    status="downloaded", upload_id=upload_id,
                )
                upload_ids.append(upload_id)
                if on_progress:
                    on_progress(done, total, f"[{done}/{total}] ✓ {artist} — {title}")
            except Exception as e:  # noqa: BLE001
                logger.exception("[yandex/importer] promote/register failed for %s", ytid)
                skipped.append({"artist": artist, "title": title, "reason": f"store failed: {e}"})
                MetadataDB.upsert_yandex_import(
                    account_id=account_id, yandex_track_id=ytid,
                    status="failed", reason=str(e),
                )

    report = {
        "total": len(tracks),
        "downloaded": len(upload_ids),
        "already": len(tracks) - len(not_imported),
        "skipped": skipped,
    }
    logger.info("[yandex/importer] account=%s sources=%s report=%s",
                account_id, sources, {k: v for k, v in report.items() if k != "skipped"})
    return upload_ids, report
