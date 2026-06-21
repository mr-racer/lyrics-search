"""Integration test for the Yandex import download phase.

Mocks the network edges (token, client, track resolution, per-track download)
but exercises the real SHA/promote/pending_uploads/yandex_imports plumbing.
"""

import types
from pathlib import Path

import pytest

from app.resources.metadata_db import MetadataDB
from app.services import uploads_service
from app.services.yandex import importer, token_store


def _track(tid, title, artist="A"):
    ns = types.SimpleNamespace
    return ns(id=tid, title=title, artists=[ns(name=artist)])


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    db_file = tmp_path / "metadata.db"
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", db_file)
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    monkeypatch.setenv("MUSIX_JWT_SECRET", "x" * 32)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    MetadataDB.create_user(
        user_id="u1", email="u1@x.y", password_hash="h",
        role="member", created_at=1700000000.0,
    )
    token_store.save_token("u1", access_token="AT")

    # Redirect media + quarantine under tmp.
    media = tmp_path / "media"
    monkeypatch.setattr(uploads_service, "media_root_default", lambda: media)
    monkeypatch.setattr(
        uploads_service, "quarantine_root_default", lambda: media / "_quarantine")

    # No real Yandex client / throttle wait.
    monkeypatch.setattr(importer, "build_client", lambda token: object())
    yield
    MetadataDB._reset_for_tests()


def _fake_download_factory(skip_ids=()):
    def _dl(track, dest_dir: Path, *, throttle=None):
        if track.id in skip_ids:
            return {"ok": False, "reason": "no Plus subscription"}
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / f"{track.id}.mp3"
        p.write_bytes(f"audio-{track.id}".encode())  # unique content → unique sha
        return {"ok": True, "path": p, "codec": "mp3", "bitrate": 320}
    return _dl


def test_happy_path_downloads_and_registers(monkeypatch):
    tracks = [_track(1, "One"), _track(2, "Two"), _track(3, "Three")]
    monkeypatch.setattr(importer.playlists, "resolve_tracks", lambda c, s: tracks)
    monkeypatch.setattr(importer.downloader, "download_track",
                        _fake_download_factory(skip_ids={2}))

    upload_ids, report = importer.download_source("u1", "likes")

    assert len(upload_ids) == 2          # track 2 skipped
    assert report["total"] == 3
    assert report["downloaded"] == 2
    assert len(report["skipped"]) == 1
    assert report["skipped"][0]["title"] == "Two"
    assert "Plus" in report["skipped"][0]["reason"]

    # pending_uploads rows exist and point into media/u1/audio.
    rows = MetadataDB.list_pending_uploads_by_account("u1")
    assert len(rows) == 2
    for r in rows:
        assert r["storage_path"].replace("\\", "/").startswith(
            str((Path(uploads_service.media_root_default()) / "u1" / "audio")).replace("\\", "/"))
        assert Path(r["storage_path"]).exists()

    # yandex_imports: 1,3 downloaded; 2 skipped.
    done = MetadataDB.get_imported_yandex_track_ids("u1")
    assert done == {"1", "3"}


def test_dedup_skips_already_imported(monkeypatch):
    tracks = [_track(1, "One"), _track(2, "Two")]
    monkeypatch.setattr(importer.playlists, "resolve_tracks", lambda c, s: tracks)
    monkeypatch.setattr(importer.downloader, "download_track", _fake_download_factory())

    first_ids, first = importer.download_source("u1", "likes")
    assert first["downloaded"] == 2

    # Second run: same tracks already imported → nothing new downloaded.
    second_ids, second = importer.download_source("u1", "likes")
    assert second_ids == []
    assert second["downloaded"] == 0
    assert second["already"] == 2


def test_not_linked_raises(monkeypatch):
    token_store.delete_token("u1")
    with pytest.raises(importer.YandexNotLinkedError):
        importer.download_source("u1", "likes")
