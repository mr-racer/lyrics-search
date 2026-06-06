"""DELETE /library/tracks/{track_id} — remove Qdrant point + disk file (server mode)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.dependencies import get_current_user
from app.domain.models import User


@pytest.fixture
def _server_mode(clean_metadata_db, tmp_path, monkeypatch):
    from app.resources.metadata_db import MetadataDB
    MetadataDB.set_instance_config(mode="server", created_at=1700000000.0)
    media_root = tmp_path / "media"
    monkeypatch.setattr("app.services.uploads_service.media_root_default", lambda: media_root)
    monkeypatch.setattr(
        "app.services.uploads_service.quarantine_root_default", lambda: media_root / "_quarantine",
    )
    return media_root


def _login(app, uid: str) -> None:
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uid, email=f"{uid}@x.y", role="member", created_at=0.0,
    )


class TestDeleteTrack:
    def test_deletes_qdrant_point_and_file(self, _server_mode):
        media_file = _server_mode / "acct_alice" / "audio" / "deadbeef.flac"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"fake audio")

        fake_qdrant = MagicMock()
        fake_qdrant.retrieve.return_value = [MagicMock(
            payload={"file_path": str(media_file), "title": "T", "artist": "A"},
        )]
        fake_db = MagicMock()
        fake_db.qdrant = fake_qdrant

        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            app.state.db_client = fake_db
            resp = c.delete("/api/v1/library/tracks/track_xyz")
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"ok": True, "track_id": "track_xyz"}

        fake_qdrant.delete.assert_called_once()
        assert not media_file.exists()

    def test_cross_account_blocked(self, _server_mode):
        # Track belongs to Bob (file path under acct_bob); Alice must not delete it.
        media_file = _server_mode / "acct_bob" / "audio" / "deadbeef.flac"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"x")

        fake_qdrant = MagicMock()
        fake_qdrant.retrieve.return_value = [MagicMock(payload={"file_path": str(media_file)})]
        fake_db = MagicMock()
        fake_db.qdrant = fake_qdrant

        app = create_app()
        _login(app, "acct_alice")
        with TestClient(app) as c:
            app.state.db_client = fake_db
            resp = c.delete("/api/v1/library/tracks/track_xyz")
            assert resp.status_code == 404
        assert media_file.exists()
        fake_qdrant.delete.assert_not_called()
