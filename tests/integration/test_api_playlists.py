"""Integration tests for /api/v1/playlists (Plan 19)."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        MetadataDB.init()
        conn = MetadataDB.get()
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM playlists")
        conn.commit()
        yield c


def test_create_returns_201_and_summary(client):
    r = client.post("/api/v1/playlists", json={
        "collection_name": "music_explorer",
        "name": "Late night",
        "description": "ночной дрифт",
    })
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Late night"
    assert body["description"] == "ночной дрифт"
    assert body["track_count"] == 0
    assert body["cover_track_ids"] == []
    assert body["cover_art_paths"] == []
    assert isinstance(body["id"], int)


def test_list_empty_collection(client):
    r = client.get("/api/v1/playlists", params={"collection_name": "fresh"})
    assert r.status_code == 200
    assert r.json() == {"playlists": [], "collection_name": "fresh"}


def test_list_sorted_by_updated_desc(client):
    a = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "A"}).json()
    b = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "B"}).json()
    import time
    time.sleep(1.05)
    client.put(f"/api/v1/playlists/{a['id']}", json={"name": "A renamed"})
    r = client.get("/api/v1/playlists", params={"collection_name": "c"}).json()
    assert [p["id"] for p in r["playlists"]] == [a["id"], b["id"]]


def test_get_detail_empty_playlist(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    r = client.get(f"/api/v1/playlists/{p['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["tracks"] == []
    assert body["missing_track_ids"] == []
    assert body["collection_name"] == "c"


def test_rename_updates_summary(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Old"}).json()
    r = client.put(f"/api/v1/playlists/{p['id']}", json={"name": "New", "description": "now with desc"})
    assert r.status_code == 200
    assert r.json()["name"] == "New"
    assert r.json()["description"] == "now with desc"


def test_update_clear_description(client):
    p = client.post("/api/v1/playlists", json={
        "collection_name": "c", "name": "M", "description": "had this",
    }).json()
    r = client.put(f"/api/v1/playlists/{p['id']}", json={"clear_description": True})
    assert r.status_code == 200
    assert r.json()["description"] is None


def test_delete_returns_204(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Bye"}).json()
    r = client.delete(f"/api/v1/playlists/{p['id']}")
    assert r.status_code == 204
    follow = client.get(f"/api/v1/playlists/{p['id']}")
    assert follow.status_code == 404


def test_add_track_increments_count(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    r1 = client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t1"})
    assert r1.status_code == 201
    assert r1.json()["track_count"] == 1
    r2 = client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t2"})
    assert r2.json()["track_count"] == 2


def test_remove_track_returns_204(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t1"})
    r = client.delete(f"/api/v1/playlists/{p['id']}/tracks/t1")
    assert r.status_code == 204


def test_reorder_renumbers_dense(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    for tid in ["t1", "t2", "t3"]:
        client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": tid})
    r = client.post(f"/api/v1/playlists/{p['id']}/reorder", json={"track_ids": ["t3", "t1", "t2"]})
    assert r.status_code == 200
    rows = MetadataDB.list_playlist_tracks(p["id"])
    assert [r["track_id"] for r in rows] == ["t3", "t1", "t2"]
    assert [r["position"] for r in rows] == [1, 2, 3]


def test_create_empty_name_returns_422(client):
    r = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "   "})
    assert r.status_code == 422


def test_create_duplicate_name_in_same_collection_returns_409(client):
    client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Mix"})
    r = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Mix"})
    assert r.status_code == 409


def test_create_same_name_different_collection_ok(client):
    r1 = client.post("/api/v1/playlists", json={"collection_name": "a", "name": "Mix"})
    r2 = client.post("/api/v1/playlists", json={"collection_name": "b", "name": "Mix"})
    assert r1.status_code == 201 and r2.status_code == 201


def test_get_missing_returns_404(client):
    r = client.get("/api/v1/playlists/9999999")
    assert r.status_code == 404


def test_rename_to_existing_name_returns_409(client):
    client.post("/api/v1/playlists", json={"collection_name": "c", "name": "A"})
    b = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "B"}).json()
    r = client.put(f"/api/v1/playlists/{b['id']}", json={"name": "A"})
    assert r.status_code == 409


def test_update_with_no_fields_returns_400(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "X"}).json()
    r = client.put(f"/api/v1/playlists/{p['id']}", json={})
    assert r.status_code == 400


def test_delete_missing_returns_404(client):
    r = client.delete("/api/v1/playlists/9999999")
    assert r.status_code == 404


def test_add_duplicate_track_returns_409(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t1"})
    r = client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t1"})
    assert r.status_code == 409


def test_add_to_missing_playlist_returns_404(client):
    r = client.post("/api/v1/playlists/9999999/tracks", json={"track_id": "t1"})
    assert r.status_code == 404


def test_remove_missing_track_returns_404(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    r = client.delete(f"/api/v1/playlists/{p['id']}/tracks/no-such")
    assert r.status_code == 404


def test_reorder_with_set_mismatch_returns_400(client):
    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t1"})
    r = client.post(f"/api/v1/playlists/{p['id']}/reorder", json={"track_ids": ["t1", "nope"]})
    assert r.status_code == 400


def test_reorder_missing_playlist_returns_404(client):
    r = client.post("/api/v1/playlists/9999999/reorder", json={"track_ids": []})
    assert r.status_code == 404
