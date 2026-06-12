"""IndexingService.index_uploads assembles a dict from pending_uploads + delegates to fit."""

from unittest.mock import MagicMock, patch

import pytest


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
