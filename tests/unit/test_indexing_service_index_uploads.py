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
