"""Integration test: GET /instance/config (no auth)."""
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


def test_instance_config_404_when_uninitialized():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 404
        assert r.json()["detail"] == "instance not initialized"


def test_instance_config_returns_sharing_mode():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 200
        assert r.json() == {"mode": "sharing"}


def test_instance_config_returns_server_mode():
    app = create_app()
    with TestClient(app) as c:
        conn = MetadataDB.get()
        conn.execute("DELETE FROM instance_config")
        conn.commit()
        MetadataDB.set_instance_config(mode="server", created_at=1.0)
        r = c.get("/api/v1/instance/config")
        assert r.status_code == 200
        assert r.json() == {"mode": "server"}
