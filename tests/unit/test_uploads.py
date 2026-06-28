"""Unit tests for the server-mode upload pipeline.

Consolidated from three former modules (each kept as its own class group):
  - test_uploads_service.py             -> upload helpers (MIME sniff, quarantine, promote)
  - test_pending_uploads_db.py          -> pending_uploads CRUD on MetadataDB
  - test_indexing_service_index_uploads.py -> IndexingService.index_uploads
"""

import hashlib
import io
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.resources.metadata_db import MetadataDB
from app.services._magic_sniff import sniff_audio_mime
from app.services.uploads_service import (
    MAX_UPLOAD_BYTES, EXT_BY_MIME,
    UploadRejected, UploadOversize, UploadWrongType,
    write_to_quarantine, atomic_promote_to_managed, choose_extension,
)

# ===========================================================================
# from test_uploads_service.py
#
# libmagic may be absent on some hosts; the MIME assertions only hold when it's
# present. The original module guarded this with a module-level
# ``pytest.importorskip("magic")``. After consolidation a module-level skip
# would also skip the unrelated DB/indexing classes, so the guard is applied
# per-class instead (identical effect, scoped to the upload-helper classes).
# ===========================================================================
try:
    import magic as _magic  # noqa: F401
    _HAS_MAGIC = True
except Exception:
    _HAS_MAGIC = False

_requires_magic = pytest.mark.skipif(
    not _HAS_MAGIC,
    reason="python-magic / libmagic not installed",
)


@_requires_magic
class TestSniffAudioMime:
    def test_flac_header(self):
        # "fLaC" magic at offset 0 is the FLAC stream signature.
        data = b"fLaC\x00\x00\x00\x22" + b"\x00" * 64
        mime = sniff_audio_mime(data)
        assert mime in ("audio/flac", "audio/x-flac")

    def test_mp3_id3v2_header(self):
        # "ID3" magic at offset 0 marks an MP3 with ID3v2 tags, followed by an
        # MPEG-1 Layer III frame sync (\xff\xfb). libmagic needs the actual frame
        # to classify as audio — a bare ID3 header alone sniffs as octet-stream.
        data = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x44" + b"\x00" * 1024
        mime = sniff_audio_mime(data)
        assert mime.startswith("audio/")  # libmagic varies — audio/mpeg or audio/mp3

    def test_rejects_text(self):
        data = b"#!/bin/sh\necho hello\n"
        mime = sniff_audio_mime(data)
        assert not mime.startswith("audio/")

    def test_rejects_jpeg(self):
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 64
        mime = sniff_audio_mime(data)
        assert not mime.startswith("audio/")


@_requires_magic
class TestFixtures:
    def test_flac_fixture_loadable(self, audio_bytes):
        data = audio_bytes("tiny.flac")
        assert data[:4] == b"fLaC"

    def test_mp3_fixture_loadable(self, audio_bytes):
        data = audio_bytes("tiny.mp3")
        # MP3 files start with either an ID3 tag block or an MPEG frame sync.
        assert data[:3] == b"ID3" or data[0] == 0xFF


@_requires_magic
class TestChooseExtension:
    def test_flac(self):
        assert choose_extension("audio/flac", original="song.flac") == ".flac"

    def test_mp3(self):
        assert choose_extension("audio/mpeg", original="song.mp3") == ".mp3"

    def test_m4a_falls_back_to_filename(self):
        # libmagic reports audio/mp4 for m4a; we need to disambiguate via extension.
        assert choose_extension("audio/mp4", original="song.m4a") == ".m4a"

    def test_unknown_mime_uses_original_extension(self):
        assert choose_extension("application/octet-stream", original="song.flac") == ".flac"

    def test_no_extension_returns_bin(self):
        assert choose_extension("application/octet-stream", original="weirdfile") == ".bin"


@_requires_magic
class TestWriteToQuarantine:
    def test_writes_bytes_and_returns_sha(self, tmp_path, audio_bytes):
        data = audio_bytes("tiny.flac")
        quarantine_path = tmp_path / "u1.tmp"
        stream = io.BytesIO(data)
        sha, size, head = write_to_quarantine(
            stream, quarantine_path, max_bytes=MAX_UPLOAD_BYTES,
        )
        assert sha == hashlib.sha256(data).hexdigest()
        assert size == len(data)
        assert head[:4] == b"fLaC"
        assert quarantine_path.exists()
        assert quarantine_path.read_bytes() == data

    def test_oversize_raises_and_cleans_up(self, tmp_path):
        big = io.BytesIO(b"x" * 1024)
        out = tmp_path / "u1.tmp"
        with pytest.raises(UploadOversize):
            write_to_quarantine(big, out, max_bytes=100)
        assert not out.exists()


@_requires_magic
class TestPromote:
    def test_atomic_move_into_managed_layout(self, tmp_path):
        src = tmp_path / "q.tmp"
        src.write_bytes(b"hello")
        media_root = tmp_path / "media"
        dst = atomic_promote_to_managed(
            quarantine_path=src,
            media_root=media_root,
            account_id="acct_abc",
            sha256="deadbeef",
            extension=".flac",
        )
        assert dst == media_root / "acct_abc" / "audio" / "deadbeef.flac"
        assert dst.exists()
        assert dst.read_bytes() == b"hello"
        assert not src.exists()

    def test_promote_idempotent_when_dst_already_exists(self, tmp_path):
        media_root = tmp_path / "media"
        existing = media_root / "acct_abc" / "audio" / "deadbeef.flac"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"existing")
        src = tmp_path / "q.tmp"
        src.write_bytes(b"new bytes")
        dst = atomic_promote_to_managed(
            quarantine_path=src,
            media_root=media_root,
            account_id="acct_abc",
            sha256="deadbeef",
            extension=".flac",
        )
        # Existing file untouched (content-addressed → identity), quarantine cleaned.
        assert dst.read_bytes() == b"existing"
        assert not src.exists()


# ===========================================================================
# from test_pending_uploads_db.py
# ===========================================================================
def _isolate_metadata_db(tmp_path, monkeypatch):
    """Point MetadataDB at a tmp file and seed FK-referenced accounts.

    pending_uploads.account_id is a FOREIGN KEY into users(id) and the DB runs
    with PRAGMA foreign_keys=ON, so we seed the accounts the tests reference
    (in production account_id always comes from an authenticated current_user).

    Wired as a per-class autouse fixture below (not module-global) so it only
    touches the pending-uploads DB classes — the upload-helper and indexing
    classes manage their own DB state.
    """
    db_file = tmp_path / "metadata.db"
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_file)
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    for uid in ("u1", "u2"):
        MetadataDB.create_user(
            user_id=uid, email=f"{uid}@x.y", password_hash="h",
            role="member", created_at=1700000000.0,
        )
    yield
    MetadataDB._reset_for_tests()


class TestPendingUploadsSchema:
    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        yield from _isolate_metadata_db(tmp_path, monkeypatch)

    def test_table_exists(self):
        conn = MetadataDB.get()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_uploads)")}
        assert {"upload_id", "account_id", "sha256", "original_filename",
                "size_bytes", "storage_path", "status", "track_id",
                "error", "created_at"} <= cols

    def test_account_status_index_exists(self):
        conn = MetadataDB.get()
        idxs = {row[1] for row in conn.execute("PRAGMA index_list(pending_uploads)")}
        assert "idx_pending_uploads_account_status" in idxs


class TestPendingUploadsCrud:
    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path, monkeypatch):
        yield from _isolate_metadata_db(tmp_path, monkeypatch)

    def _make(self, account_id="u1", sha="0" * 64, name="song.mp3"):
        upload_id = MetadataDB.create_pending_upload(
            account_id=account_id,
            sha256=sha,
            original_filename=name,
            size_bytes=1024,
            storage_path=f"media/{account_id}/audio/{sha}.mp3",
        )
        return upload_id

    def test_create_returns_uuid_like_id(self):
        upload_id = self._make()
        assert isinstance(upload_id, str)
        assert len(upload_id) >= 16

    def test_get_returns_full_row(self):
        upload_id = self._make()
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["upload_id"] == upload_id
        assert row["status"] == "uploaded"
        assert row["account_id"] == "u1"
        assert row["track_id"] is None
        assert row["error"] is None

    def test_get_missing_returns_none(self):
        assert MetadataDB.get_pending_upload("no-such-id") is None

    def test_update_status_to_indexing(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(upload_id, status="indexing")
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "indexing"

    def test_update_status_to_done_sets_track_id(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(
            upload_id, status="done", track_id="abc123",
        )
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "done"
        assert row["track_id"] == "abc123"

    def test_update_status_to_failed_sets_error(self):
        upload_id = self._make()
        MetadataDB.update_pending_upload_status(
            upload_id, status="failed", error="encoding failed",
        )
        row = MetadataDB.get_pending_upload(upload_id)
        assert row["status"] == "failed"
        assert row["error"] == "encoding failed"

    def test_list_by_account_scoped_to_account(self):
        self._make(account_id="u1", sha="a" * 64)
        self._make(account_id="u1", sha="b" * 64)
        self._make(account_id="u2", sha="c" * 64)
        rows_u1 = MetadataDB.list_pending_uploads_by_account("u1")
        rows_u2 = MetadataDB.list_pending_uploads_by_account("u2")
        assert {r["sha256"] for r in rows_u1} == {"a" * 64, "b" * 64}
        assert {r["sha256"] for r in rows_u2} == {"c" * 64}

    def test_list_by_account_filter_by_status(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        self._make(account_id="u1", sha="b" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        only_uploaded = MetadataDB.list_pending_uploads_by_account(
            "u1", status="uploaded",
        )
        assert len(only_uploaded) == 1
        assert only_uploaded[0]["sha256"] == "b" * 64

    def test_idempotent_lookup_by_sha(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        existing = MetadataDB.find_done_upload_by_sha(
            account_id="u1", sha256="a" * 64,
        )
        assert existing is not None
        assert existing["track_id"] == "t1"
        # Cross-account isolation: same sha, different account = no match.
        assert MetadataDB.find_done_upload_by_sha(
            account_id="u2", sha256="a" * 64,
        ) is None

    def test_purge_old_done_uploads(self):
        u1 = self._make(account_id="u1", sha="a" * 64)
        MetadataDB.update_pending_upload_status(u1, status="done", track_id="t1")
        # Backdate to 8 days ago.
        conn = MetadataDB.get()
        conn.execute(
            "UPDATE pending_uploads SET created_at = ? WHERE upload_id = ?",
            (time.time() - 8 * 86400, u1),
        )
        conn.commit()
        # Younger row stays.
        u2 = self._make(account_id="u1", sha="b" * 64)
        MetadataDB.update_pending_upload_status(u2, status="done", track_id="t2")
        n = MetadataDB.purge_old_pending_uploads(older_than_seconds=7 * 86400)
        assert n == 1
        rows = MetadataDB.list_pending_uploads_by_account("u1")
        assert {r["upload_id"] for r in rows} == {u2}


# ===========================================================================
# from test_indexing_service_index_uploads.py
#
# IndexingService.index_uploads assembles a dict from pending_uploads + delegates to fit.
# ===========================================================================
class TestIndexUploadsSignature:
    def test_method_exists(self):
        from app.services.indexing_service import IndexingService
        assert callable(getattr(IndexingService, "index_uploads", None))


class TestIndexUploadsAssemblesData:
    def test_calls_fit_with_dict_keyed_by_artist_title(self, tmp_path, audio_path):
        from app.services.indexing_service import IndexingService

        engine = MagicMock()
        svc = IndexingService(engine)

        flac = tmp_path / "song.flac"
        flac.write_bytes(audio_path("tiny.flac").read_bytes())

        upload_rows = [
            {
                "upload_id": "u1",
                "account_id": "acct_x",
                "sha256": "deadbeef",
                "original_filename": "song.flac",
                "storage_path": str(flac),
                "size_bytes": flac.stat().st_size,
                "status": "uploaded",
                "track_id": None,
                "error": None,
                "created_at": 1.0,
            }
        ]

        # _fit_impl mocked (fit_with_progress delegates to it) so no live Qdrant;
        # get_lyrics patched so the test never hits the network.
        with patch.object(svc, "_fit_impl") as fit_mock, \
             patch("app.indexing.folder_scanner.get_lyrics", return_value=""), \
             patch("app.resources.metadata_db.MetadataDB.update_pending_upload_status"):
            svc.index_uploads(account_id="acct_x", upload_rows=upload_rows)
            fit_mock.assert_called_once()
            args, kwargs = fit_mock.call_args
            data = args[0] if args else kwargs.get("data")
            assert isinstance(data, dict)
            # Key shape matches scan_folder's "Artist — Title" convention.
            key = next(iter(data))
            assert " — " in key
            entry = data[key]
            # storage_path becomes file_path so CLAP encoder + payload builder
            # find the bytes.
            assert entry["file_path"] == str(flac)
            # Metadata extracted via the real folder_scanner.process_file (mutagen).
            assert entry["title"] == "Tiny Test Song"
            assert entry["artist"] == "Tiny Test Artist"

    def test_indexed_sink_is_populated(self, tmp_path, audio_path):
        """The optional indexed_sink out-param exposes the indexed batch so the
        server-mode upload runner can run the FACTS/enrichment stage on it."""
        from app.services.indexing_service import IndexingService

        svc = IndexingService(MagicMock())
        flac = tmp_path / "song.flac"
        flac.write_bytes(audio_path("tiny.flac").read_bytes())
        rows = [{
            "upload_id": "u1", "account_id": "acct_x", "sha256": "deadbeef",
            "original_filename": "song.flac", "storage_path": str(flac),
            "size_bytes": flac.stat().st_size, "status": "uploaded",
            "track_id": None, "error": None, "created_at": 1.0,
        }]
        sink: dict = {}
        with patch.object(svc, "_fit_impl"), \
             patch("app.indexing.folder_scanner.get_lyrics", return_value=""), \
             patch("app.resources.metadata_db.MetadataDB.update_pending_upload_status"):
            svc.index_uploads(account_id="acct_x", upload_rows=rows, indexed_sink=sink)

        assert sink, "indexed_sink should be populated with the indexed batch"
        key = next(iter(sink))
        assert " — " in key
        assert sink[key]["artist"] == "Tiny Test Artist"
        assert sink[key]["title"] == "Tiny Test Song"


class TestIndexUploadsProgressAndCovers:
    """The metadata+lyrics loop is the SLOW pre-embedding part of an upload
    job (network lyrics fetch per file). It must report progress — without it
    the wizard's SSE stream is silent until embeddings — and extract embedded
    cover art the same way the folder flow does in _metadata_to_tracks."""

    def _one_row(self, tmp_path, audio_path):
        flac = tmp_path / "song.flac"
        flac.write_bytes(audio_path("tiny.flac").read_bytes())
        return {
            "upload_id": "u1",
            "account_id": "acct_x",
            "sha256": "deadbeef",
            "original_filename": "song.flac",
            "storage_path": str(flac),
            "size_bytes": flac.stat().st_size,
            "status": "uploaded",
            "track_id": None,
            "error": None,
            "created_at": 1.0,
        }

    def test_scan_stage_reports_progress(self, tmp_path, audio_path):
        from app.services.indexing_service import IndexingService

        svc = IndexingService(MagicMock())
        events = []

        with patch.object(svc, "_fit_impl"), \
             patch("app.indexing.folder_scanner.get_lyrics", return_value=""), \
             patch("app.resources.metadata_db.MetadataDB.update_pending_upload_status"):
            svc.index_uploads(
                account_id="acct_x",
                upload_rows=[self._one_row(tmp_path, audio_path)],
                progress_callback=lambda *a: events.append(a),
            )

        scan_events = [e for e in events if e[0] == "scan"]
        assert scan_events, f"no 'scan' progress events emitted; got: {events}"
        # First event announces the stage (0/total), last one closes it (total/total).
        assert scan_events[0][1] == 0 and scan_events[0][2] == 1
        assert scan_events[-1][1] == 1 and scan_events[-1][2] == 1

    def test_cover_art_extracted_into_payload_data(self, tmp_path, audio_path):
        from app.services.indexing_service import IndexingService

        svc = IndexingService(MagicMock())

        with patch.object(svc, "_fit_impl") as fit_mock, \
             patch("app.indexing.folder_scanner.get_lyrics", return_value=""), \
             patch("app.indexing.cover_art.save_cover_art", return_value="/covers/abc.jpg"), \
             patch("app.resources.metadata_db.MetadataDB.update_pending_upload_status"):
            svc.index_uploads(
                account_id="acct_x",
                upload_rows=[self._one_row(tmp_path, audio_path)],
            )
            args, kwargs = fit_mock.call_args
            data = args[0] if args else kwargs.get("data")
            entry = data[next(iter(data))]
            assert entry["cover_art_path"] == "/covers/abc.jpg"


class TestIndexUploadsConcurrency:
    """The metadata+lyrics scan is network-bound (an online lyrics fetch per
    file). Server-mode uploads must fetch concurrently — like the folder-scan
    path (scan_and_enrich_folder) — not one file at a time."""

    def _row_n(self, tmp_path, n):
        f = tmp_path / f"song{n}.flac"
        f.write_bytes(b"\x00")  # content irrelevant — process_file is mocked
        return {
            "upload_id": f"u{n}",
            "account_id": "acct_x",
            "sha256": f"deadbeef{n:08d}",
            "original_filename": f"song{n}.flac",
            "storage_path": str(f),
            "size_bytes": 1,
            "status": "uploaded",
            "track_id": None,
            "error": None,
            "created_at": float(n),
        }

    def test_lyrics_fetch_runs_concurrently(self, tmp_path):
        from app.services.indexing_service import IndexingService

        svc = IndexingService(MagicMock())
        n = 3
        # A 3-party barrier is satisfied ONLY if all 3 process_file calls are
        # in flight simultaneously. Sequential processing strands the first
        # thread at the barrier → BrokenBarrierError on timeout → that row
        # drops to "failed" and never reaches _fit_impl.
        barrier = threading.Barrier(n, timeout=5)

        def fake_process_file(file_path, better, *, enrich_client=None):
            barrier.wait()
            stem = Path(file_path).stem
            return {"title": f"Song {stem}", "artist": f"Artist {stem}", "lyrics": "la"}

        rows = [self._row_n(tmp_path, i) for i in range(n)]

        with patch.object(svc, "_fit_impl") as fit_mock, \
             patch("app.indexing.folder_scanner.process_file", side_effect=fake_process_file), \
             patch("app.indexing.cover_art.save_cover_art", return_value=None), \
             patch("app.resources.metadata_db.MetadataDB.update_pending_upload_status"):
            svc.index_uploads(account_id="acct_x", upload_rows=rows)

        assert fit_mock.called, (
            "no rows reached embedding — process_file calls never overlapped, "
            "so the barrier timed out: uploads are still processed sequentially"
        )
        args, kwargs = fit_mock.call_args
        data = args[0] if args else kwargs.get("data")
        assert len(data) == n, f"expected {n} concurrently-processed tracks, got {len(data)}"


class TestIndexUploadsUpdatesStatus:
    def test_status_marked_done_with_track_id(self, tmp_path, audio_path, monkeypatch):
        from app.resources.metadata_db import MetadataDB
        from app.services.indexing_service import IndexingService

        db_file = tmp_path / "metadata.db"
        monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_file)
        monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
        MetadataDB._reset_for_tests()
        MetadataDB.init()
        try:
            # account_id is a FK into users(id) (foreign_keys=ON).
            MetadataDB.create_user(
                user_id="acct_x", email="x@x.y", password_hash="h",
                role="member", created_at=1700000000.0,
            )
            flac = tmp_path / "song.flac"
            flac.write_bytes(audio_path("tiny.flac").read_bytes())
            upload_id = MetadataDB.create_pending_upload(
                account_id="acct_x", sha256="deadbeef",
                original_filename="song.flac",
                size_bytes=flac.stat().st_size,
                storage_path=str(flac),
            )

            # Fake the post-upsert scroll: one point matching the assembled key.
            fake_point = MagicMock()
            fake_point.id = "t_fake"
            fake_point.payload = {"artist": "Tiny Test Artist", "title": "Tiny Test Song"}
            engine = MagicMock()
            engine.qdrant_client.scroll.return_value = ([fake_point], None)

            svc = IndexingService(engine)
            with patch.object(svc, "_fit_impl"), \
                 patch("app.indexing.folder_scanner.get_lyrics", return_value=""):
                out = svc.index_uploads(account_id="acct_x", upload_rows=[
                    MetadataDB.get_pending_upload(upload_id),
                ])

            assert out == {upload_id: "t_fake"}
            row = MetadataDB.get_pending_upload(upload_id)
            assert row["status"] == "done"
            assert row["track_id"] == "t_fake"
        finally:
            MetadataDB._reset_for_tests()
