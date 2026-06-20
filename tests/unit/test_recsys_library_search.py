"""The recsys 'library_search' tool must route to the catalog tracks engine
(by name → songs), not the semantic hybrid search."""
import asyncio

import pytest

pytestmark = pytest.mark.unit

from app.services import recsys_ai_service, catalog_search_service


def test_library_search_uses_catalog_tracks(monkeypatch):
    captured = {}

    def fake_tracks(qdrant, collection, query, limit=20):
        captured["args"] = (collection, query, limit)
        return [{
            "track_id": "x1", "title": "Time", "artist": "Pink Floyd",
            "album": "DSOTM", "year": 1973, "genre": "rock", "duration": 412.0,
            "file_path": "/m/x1.mp3", "cover_art_path": None,
        }]

    monkeypatch.setattr(catalog_search_service, "search_catalog_tracks", fake_tracks)

    class _NoSemanticSearch:
        async def search(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("library_search must not call semantic search")

    action = {"tool": "library_search", "query": "pink floyd", "limit": 15}
    out = asyncio.run(recsys_ai_service._execute_action(
        action,
        search_service=_NoSemanticSearch(),
        qdrant_client=object(),
        collection_name="acct_x",
    ))

    assert captured["args"] == ("acct_x", "pink floyd", 15)
    assert out and out[0]["track_id"] == "x1"
    assert out[0]["file_path"] == "/m/x1.mp3"
    assert out[0]["tool"] == "library_search"
