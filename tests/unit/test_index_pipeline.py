"""Unit tests for IndexPipeline — the concurrent lyrics/CLAP/dense orchestrator.

IndexingService's encode primitives are mocked, so these tests run with no GPU,
no Qdrant and no network: they verify the orchestration contract (which tracks
get an online lyrics fetch, the CLAP→dense order, and that fetched lyrics land in
the canonical dicts the upsert reads).
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np

from app.services.index_pipeline import IndexPipeline


def _run(coro):
    return asyncio.run(coro)


def test_partitions_lyrics_and_orders_clap_before_dense(monkeypatch):
    # One track already has embedded lyrics, one is empty (needs an online fetch).
    tracks = {
        "A — 1": {"artist": "A", "title": "1", "file_path": "/x/1.flac",
                  "lyrics": "embedded", "duration": 100},
        "A — 2": {"artist": "A", "title": "2", "file_path": "/x/2.flac",
                  "lyrics": "", "duration": 100},
    }
    # prepare returns the SAME dict objects (a list) so in-place lyrics writes are
    # visible to encode_dense / upsert — exactly the real prepare_metadata contract
    # this pipeline relies on.
    filtered = list(tracks.values())

    order = []
    fake_indexing = MagicMock()
    fake_indexing.prepare.return_value = (filtered, ["/x/1.flac", "/x/2.flac"])

    def _clap(ft, progress_callback=None):
        order.append("clap")
        return {}, {}, {}

    def _dense(ft, progress_callback=None):
        order.append("dense")
        # dense must see lyrics already fetched into the dicts
        order.append(tuple(e["lyrics"] for e in ft))
        return np.zeros((len(ft), 4), dtype=np.float32)

    fake_indexing.encode_clap.side_effect = _clap
    fake_indexing.encode_dense.side_effect = _dense
    ctor_kwargs = {}

    def _fake_ctor(engine, **kw):
        ctor_kwargs.update(kw)
        return fake_indexing

    monkeypatch.setattr("app.services.index_pipeline.IndexingService", _fake_ctor)

    fetched = []

    def _fetch(entry, quality):
        fetched.append(entry["title"])
        return "ONLINE"

    monkeypatch.setattr("app.services.index_pipeline.fetch_online_lyrics", _fetch)

    engine = MagicMock()
    engine.collection_name = "saved"
    pipe = IndexPipeline(engine)
    n, ids = _run(pipe.run(tracks, "acct_x"))

    assert n == 2
    assert ids == {}                              # resolve_track_ids=False
    assert fetched == ["2"]                        # ONLY the empty-lyrics track fetched
    assert filtered[1]["lyrics"] == "ONLINE"       # fetched text written back in place
    assert filtered[0]["lyrics"] == "embedded"     # embedded untouched
    # CLAP runs before dense, and dense sees the fetched lyrics.
    assert order[0] == "clap"
    assert order[1] == "dense"
    assert order[2] == ("embedded", "ONLINE")
    fake_indexing.create_collection.assert_called_once()
    fake_indexing.upsert.assert_called_once()
    # Collection/model are passed to the per-run IndexingService instance;
    # the shared engine is never mutated (parallel-jobs isolation).
    assert ctor_kwargs == {"collection_name": "acct_x", "model_name": None}
    assert engine.collection_name == "saved"


def test_empty_batch_short_circuits(monkeypatch):
    fake_indexing = MagicMock()
    fake_indexing.prepare.return_value = ([], [])
    monkeypatch.setattr(
        "app.services.index_pipeline.IndexingService", lambda engine, **kw: fake_indexing,
    )
    engine = MagicMock()
    engine.collection_name = "saved"
    n, ids = _run(IndexPipeline(engine).run({}, "acct_x"))
    assert n == 0 and ids == {}
    fake_indexing.encode_clap.assert_not_called()
    fake_indexing.upsert.assert_not_called()
