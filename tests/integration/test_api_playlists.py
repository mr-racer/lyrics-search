"""Integration tests for /api/v1/playlists (Plan 19).

Phase D (D-soft) adaptations
------------------------------
POST "" and GET "" now derive the collection from the JWT user (acct_<uid>).
Client-supplied ``collection_name`` is accepted but silently ignored.

Implications for the integration suite:
- Every playlist created via this client lands in the SAME derived collection
  (acct_<test-owner-uid>), regardless of the ``collection_name`` sent in the
  body / query-param.
- Tests that previously relied on separate client-supplied collection names to
  achieve isolation (test_cross_collection_list_isolation,
  test_create_same_name_different_collection_ok) are updated to reflect the
  new reality: isolation is now per-account, not per-client-supplied string.
- Assertions on the echoed ``collection_name`` now match the derived value
  (starts with "acct_") rather than the old client-supplied literals.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.resources.metadata_db import MetadataDB
from ._auth_helper import authenticate_test_client


@pytest.fixture()
def client():
    _app = create_app()
    with TestClient(_app) as c:
        authenticate_test_client(c, _app)
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
    # Phase D-soft: collection_name query param is ignored; server derives from JWT.
    # The echoed collection_name in the response is the derived "acct_<uid>".
    r = client.get("/api/v1/playlists", params={"collection_name": "fresh"})
    assert r.status_code == 200
    body = r.json()
    assert body["playlists"] == []
    # collection_name is now the derived acct_ value, not the client-supplied "fresh"
    assert body["collection_name"].startswith("acct_")


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
    # Phase D-soft: collection stored in DB is now the derived "acct_<uid>", not "c"
    assert body["collection_name"].startswith("acct_")


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
    # Phase D-soft: both requests go to the same derived collection — collision expected
    client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Mix"})
    r = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Mix"})
    assert r.status_code == 409


def test_create_same_name_in_different_supplied_collections_collides(client):
    """Phase D-soft: client-supplied collection_name is IGNORED; both requests
    land in the same derived collection → second create is a name collision (409)."""
    r1 = client.post("/api/v1/playlists", json={"collection_name": "a", "name": "Mix"})
    r2 = client.post("/api/v1/playlists", json={"collection_name": "b", "name": "Mix"})
    assert r1.status_code == 201
    # "b" is ignored — server uses the same acct_ collection — same name → conflict
    assert r2.status_code == 409


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


def test_list_contains_all_playlists_regardless_of_supplied_collection(client):
    """Phase D-soft: supplied collection_name is ignored → both playlists land in
    the same derived collection → both appear in any list request."""
    client.post("/api/v1/playlists", json={"collection_name": "a", "name": "X"})
    client.post("/api/v1/playlists", json={"collection_name": "b", "name": "Y"})
    # Even requesting "a" returns everything in the derived collection
    r = client.get("/api/v1/playlists", params={"collection_name": "a"}).json()
    names = {p["name"] for p in r["playlists"]}
    assert names == {"X", "Y"}


def test_orphan_tracks_partitioned_in_detail(client, monkeypatch):
    """When Qdrant only resolves SOME track_ids, the rest land in missing_track_ids."""
    import app.services.playlists_service as svc

    def fake_resolve_payloads(qdrant, collection_name: str, track_ids: list) -> dict:
        return {"t_real": {"title": "Real", "artist": "Artist"}}

    monkeypatch.setattr(svc, "_resolve_payloads", fake_resolve_payloads)

    p = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "M"}).json()
    client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t_real"})
    client.post(f"/api/v1/playlists/{p['id']}/tracks", json={"track_id": "t_orphan"})
    detail = client.get(f"/api/v1/playlists/{p['id']}").json()
    assert [t["track_id"] for t in detail["tracks"]] == ["t_real"]
    assert detail["missing_track_ids"] == ["t_orphan"]


def test_include_track_id_annotates_contains_track(client):
    p1 = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "Has"}).json()
    p2 = client.post("/api/v1/playlists", json={"collection_name": "c", "name": "HasNot"}).json()
    client.post(f"/api/v1/playlists/{p1['id']}/tracks", json={"track_id": "t-target"})

    r = client.get("/api/v1/playlists", params={"collection_name": "c", "include_track_id": "t-target"}).json()
    by_name = {p["name"]: p for p in r["playlists"]}
    assert by_name["Has"]["contains_track"] is True
    assert by_name["HasNot"]["contains_track"] is False


def test_without_include_track_id_contains_track_is_null(client):
    client.post("/api/v1/playlists", json={"collection_name": "c", "name": "X"})
    r = client.get("/api/v1/playlists", params={"collection_name": "c"}).json()
    assert r["playlists"][0]["contains_track"] is None


# ── Phase D-soft unit tests ────────────────────────────────────────────────────


def test_create_playlist_ignores_supplied_collection():
    """POST /playlists: server MUST pass derived acct_<uid> to the service,
    not the client-supplied collection_name."""
    from app.api.routes import playlists as pl_route
    from app.services import playlists_service
    from app.domain.models import PlaylistSummary
    from types import SimpleNamespace
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.api.main import create_app
    import datetime

    _fake_summary = PlaylistSummary(
        id=1, name="X", description=None, track_count=0,
        cover_track_ids=[], cover_art_paths=[],
        created_at=datetime.datetime.now().isoformat(),
        updated_at=datetime.datetime.now().isoformat(),
    )

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[pl_route.get_current_user] = lambda: fixed
    with patch.object(playlists_service, "create", return_value=_fake_summary) as p_create:
        with TestClient(app) as c:
            c.post("/api/v1/playlists", json={"collection_name": "acct_BAD", "name": "X"})
        assert p_create.call_args.args[0] == "acct_user-A"


def test_list_playlists_ignores_supplied_collection():
    """GET /playlists: server MUST scope to acct_<uid>, not the client query param."""
    from app.api.routes import playlists as pl_route
    from app.services import playlists_service
    from types import SimpleNamespace
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.api.main import create_app

    fixed = SimpleNamespace(id="user-A", email="a@x")
    app = create_app()
    app.dependency_overrides[pl_route.get_current_user] = lambda: fixed
    with patch.object(playlists_service, "list_playlists", return_value=[]) as p_list:
        with TestClient(app) as c:
            c.get("/api/v1/playlists", params={"collection_name": "acct_BAD"})
        assert p_list.call_args.args[0] == "acct_user-A"


def test_create_without_collection_name_accepted():
    """POST /playlists: omitting collection_name in body must NOT return 422
    (field is Optional after Phase D model relaxation)."""
    from app.api.routes import playlists as pl_route
    from app.services import playlists_service
    from app.domain.models import PlaylistSummary
    from types import SimpleNamespace
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.api.main import create_app
    import datetime

    _fake_summary = PlaylistSummary(
        id=2, name="No Collection Field", description=None, track_count=0,
        cover_track_ids=[], cover_art_paths=[],
        created_at=datetime.datetime.now().isoformat(),
        updated_at=datetime.datetime.now().isoformat(),
    )

    fixed = SimpleNamespace(id="user-B", email="b@x")
    app = create_app()
    app.dependency_overrides[pl_route.get_current_user] = lambda: fixed
    with patch.object(playlists_service, "create", return_value=_fake_summary):
        with TestClient(app) as c:
            r = c.post("/api/v1/playlists", json={"name": "No Collection Field"})
        # Must not 422 — collection_name is optional
        assert r.status_code == 201
