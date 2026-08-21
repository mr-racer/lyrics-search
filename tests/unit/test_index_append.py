"""Unit tests for incremental ("add music") indexing.

The historic index path always dropped and recreated the target collection, so
a second run replaced the library instead of extending it. These tests pin the
append contract: an existing collection survives, the SQLite mirror is not
cleared, already-indexed files are skipped, and the axis norm stats are
recomputed over the whole collection rather than the incoming batch.

Everything external is faked — no Qdrant, no GPU, no network.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.index_pipeline import IndexPipeline
from app.services.indexing_service import IndexingService


def _run(coro):
    return asyncio.run(coro)


def _fake_pipeline_indexing(monkeypatch, filtered, clap_paths=None):
    """Install a MagicMock IndexingService into IndexPipeline; return it."""
    fake = MagicMock()
    fake.prepare.return_value = (filtered, clap_paths or [])
    fake.encode_clap.return_value = ({}, {}, {})
    fake.encode_dense.return_value = np.zeros((len(filtered), 4), dtype=np.float32)
    monkeypatch.setattr(
        "app.services.index_pipeline.IndexingService", lambda engine, **kw: fake,
    )
    monkeypatch.setattr(
        "app.services.index_pipeline.fetch_online_lyrics", lambda entry, quality: "",
    )
    return fake


# ── IndexPipeline: append vs replace ────────────────────────────────────────

def test_append_keeps_collection_and_mirror(monkeypatch):
    """append=True must never drop the collection or clear track_metadata."""
    filtered = [{"artist": "A", "title": "1", "file_path": "/x/1.flac", "lyrics": "L"}]
    fake = _fake_pipeline_indexing(monkeypatch, filtered)
    cleared = []
    monkeypatch.setattr(
        "app.services.index_pipeline.MetadataDB.clear_track_metadata",
        lambda coll: cleared.append(coll),
    )

    n, _ = _run(IndexPipeline(MagicMock()).run(
        {"A — 1": filtered[0]}, "acct_x", append=True,
    ))

    assert n == 1
    fake.create_collection.assert_not_called()      # the destructive one
    fake.ensure_collection.assert_called_once()     # the additive one
    fake.upsert.assert_called_once()
    assert cleared == []                            # mirror survives


def test_replace_still_recreates_collection(monkeypatch):
    """append=False keeps the historic drop-and-rebuild behaviour."""
    filtered = [{"artist": "A", "title": "1", "file_path": "/x/1.flac", "lyrics": "L"}]
    fake = _fake_pipeline_indexing(monkeypatch, filtered)
    cleared = []
    monkeypatch.setattr(
        "app.services.index_pipeline.MetadataDB.clear_track_metadata",
        lambda coll: cleared.append(coll),
    )

    _run(IndexPipeline(MagicMock()).run({"A — 1": filtered[0]}, "acct_x"))

    fake.create_collection.assert_called_once()
    fake.ensure_collection.assert_not_called()
    assert cleared == ["acct_x"]


def test_append_recomputes_norms_from_mirror_not_batch(monkeypatch):
    """Batch stats would overwrite the collection's with n=<batch size>."""
    filtered = [{"artist": "A", "title": "1", "file_path": "/x/1.flac", "lyrics": "L"}]
    fake = _fake_pipeline_indexing(monkeypatch, filtered)
    monkeypatch.setattr(
        "app.services.index_pipeline.MetadataDB.clear_track_metadata", lambda coll: None,
    )

    _run(IndexPipeline(MagicMock()).run(
        {"A — 1": filtered[0]}, "acct_x", append=True,
    ))

    fake.persist_axis_norm_stats_from_mirror.assert_called_once()
    # finalize_norms_and_prune bundles the batch-scoped stats write with a
    # full-collection prune scan — neither belongs to an append.
    fake.finalize_norms_and_prune.assert_not_called()


# ── IndexingService.ensure_collection ───────────────────────────────────────

def _service_with_client(client, *, vector_name="text", vector_dim=4):
    engine = MagicMock()
    engine.qdrant_client = client
    engine.vector_name = vector_name
    engine.vector_dim = vector_dim
    return IndexingService(engine, collection_name="acct_x")


def _named(name):
    """MagicMock(name=...) sets the mock's repr, not the attribute."""
    m = MagicMock()
    m.name = name
    return m


def test_ensure_collection_is_a_noop_when_present():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[_named("acct_x")])
    client.get_collection.return_value.config.params.vectors = {"text": {}, "clap": {}}
    _service_with_client(client).ensure_collection(["/x/1.flac"])

    client.delete_collection.assert_not_called()
    client.create_collection.assert_not_called()


def test_ensure_collection_creates_when_absent():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    _service_with_client(client).ensure_collection(["/x/1.flac"])

    client.delete_collection.assert_not_called()
    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "acct_x"
    assert set(kwargs["vectors_config"]) == {"text", "clap"}


# ── Dedupe: a file already in the library must not be indexed twice ─────────

def test_split_already_indexed_partitions_on_file_path(monkeypatch):
    from app.services import library_service as ls

    monkeypatch.setattr(
        ls.MetadataDB, "get_light_points",
        classmethod(lambda cls, coll: [("id1", {"file_path": "/m/a.flac"})]),
    )
    tracks = {
        "A — 1": {"artist": "A", "title": "1", "file_path": "/m/a.flac"},
        "A — 2": {"artist": "A", "title": "2", "file_path": "/m/b.flac"},
    }

    fresh, already = ls.LibraryService._split_already_indexed("acct_x", tracks)

    assert list(fresh) == ["A — 2"]
    assert already == {"A — 1": "id1"}   # key → the point already holding it


def test_split_already_indexed_passes_everything_through_on_empty_mirror(monkeypatch):
    from app.services import library_service as ls

    monkeypatch.setattr(
        ls.MetadataDB, "get_light_points", classmethod(lambda cls, coll: []),
    )
    tracks = {"A — 1": {"artist": "A", "title": "1", "file_path": "/m/a.flac"}}

    fresh, already = ls.LibraryService._split_already_indexed("acct_x", tracks)

    assert fresh == tracks and already == {}


def test_split_already_indexed_survives_a_mirror_failure(monkeypatch):
    """A dedupe lookup failure must degrade to "index everything", not raise."""
    from app.services import library_service as ls

    def _boom(cls, coll):
        raise RuntimeError("sqlite is down")

    monkeypatch.setattr(ls.MetadataDB, "get_light_points", classmethod(_boom))
    tracks = {"A — 1": {"artist": "A", "title": "1", "file_path": "/m/a.flac"}}

    fresh, already = ls.LibraryService._split_already_indexed("acct_x", tracks)

    assert fresh == tracks and already == {}


# ── Upload job wiring: dedupe short-circuit + honest completion counters ────

def _library_service_for_upload(monkeypatch):
    """A LibraryService with every external dependency stubbed out."""
    from app.services import library_service as ls

    monkeypatch.setattr(ls.ModelRegistry, "get_text_model", staticmethod(lambda: object()))
    monkeypatch.setattr(
        ls.MetadataDB, "get_pending_upload",
        classmethod(lambda cls, uid: {"upload_id": uid, "account_id": "acc1"}),
    )
    monkeypatch.setattr(
        ls.MetadataDB, "update_pending_upload_status",
        classmethod(lambda cls, *a, **kw: None),
    )
    svc = ls.LibraryService(db_client=MagicMock())
    return svc


def _frames(svc):
    """Capture every SSE payload the job publishes."""
    sent = []

    async def _notify(job, data):
        sent.append(data)

    svc._notify_progress = _notify
    return sent


def test_upload_job_short_circuits_when_everything_is_already_indexed(monkeypatch):
    from app.services.job_tracker import IndexStatus, JobTracker

    svc = _library_service_for_upload(monkeypatch)
    sent = _frames(svc)
    job = JobTracker().create_job("<uploads:2>", "acct_acc1")

    tracks = {
        "A — 1": {"artist": "A", "title": "1", "file_path": "/m/a.flac"},
        "A — 2": {"artist": "A", "title": "2", "file_path": "/m/b.flac"},
    }

    async def _tagread(account_id, rows, enrich_client, progress):
        return tracks, {"A — 1": "u1", "A — 2": "u2"}

    svc._tagread_upload_rows = _tagread
    monkeypatch.setattr(
        type(svc), "_split_already_indexed",
        staticmethod(lambda coll, t: ({}, {"A — 1": "p1", "A — 2": "p2"})),
    )
    stamped = {}
    svc._apply_upload_track_ids = lambda by_key, ids: stamped.update(ids)

    called = []
    monkeypatch.setattr(
        "app.services.index_pipeline.IndexPipeline.run",
        lambda *a, **kw: called.append(1),
    )

    _run(svc._run_upload_indexing_job(job, "acc1", ["u1", "u2"], "ru"))

    assert called == []                                  # no encode work at all
    assert job.overall_status == IndexStatus.COMPLETED
    final = sent[-1]
    assert final["tracks_added"] == 0
    assert final["tracks_skipped"] == 2
    # The uploads were flipped to 'indexing' by the tag-read — they must be
    # pointed back at the tracks that already hold them, not left hanging.
    assert stamped == {"A — 1": "p1", "A — 2": "p2"}
    # Nothing may be left RUNNING, or the client's progress UI spins forever.
    assert all(sp.status == IndexStatus.COMPLETED for sp in job.stages.values())


def test_upload_job_reports_added_and_skipped(monkeypatch):
    from app.services.job_tracker import IndexStatus, JobTracker

    svc = _library_service_for_upload(monkeypatch)
    sent = _frames(svc)
    job = JobTracker().create_job("<uploads:2>", "acct_acc1")

    fresh = {"A — 2": {"artist": "A", "title": "2", "file_path": "/m/b.flac"}}

    async def _tagread(account_id, rows, enrich_client, progress):
        return dict(fresh, **{"A — 1": {"file_path": "/m/a.flac"}}), {"A — 1": "u1", "A — 2": "u2"}

    svc._tagread_upload_rows = _tagread
    monkeypatch.setattr(
        type(svc), "_split_already_indexed",
        staticmethod(lambda coll, t: (fresh, {"A — 1": "p1"})),
    )
    svc._apply_upload_track_ids = lambda by_key, ids: None
    svc._apply_producer_label = lambda *a, **kw: None

    async def _facts(job_, coll, data, lang):
        return {}

    svc._fetch_facts_batch = _facts

    async def _ai(coll, n, lang, job=None):
        return None

    svc._run_ai_tasks = _ai

    seen = {}

    async def _fake_run(self, tracks, collection_name, **kw):
        seen["append"] = kw.get("append")
        seen["tracks"] = tracks
        return len(tracks), {}

    monkeypatch.setattr("app.services.index_pipeline.IndexPipeline.run", _fake_run)

    _run(svc._run_upload_indexing_job(job, "acc1", ["u1", "u2"], "ru"))

    assert job.overall_status == IndexStatus.COMPLETED
    # Uploads always append: on a first upload ensure_collection creates the
    # collection, on a later one the existing library is extended.
    assert seen["append"] is True
    assert list(seen["tracks"]) == ["A — 2"]     # the duplicate never reached the GPU
    final = sent[-1]
    assert final["tracks_added"] == 1
    assert final["tracks_skipped"] == 1


def test_a_rescan_with_nothing_new_never_warms_the_gpu(monkeypatch):
    """Loading the text model is tens of seconds of blocking work.

    A rescan whose path diff comes back empty — the usual outcome on a settled
    library — has to establish that FIRST and stop. Warming a GPU model it will
    never use is what turned "0 new files" into a long wait, and, when the
    model itself could not load, into a failed job for a no-op.
    """
    from app.services import library_service as ls
    from app.services.job_tracker import IndexStatus, JobTracker

    loaded: list = []
    monkeypatch.setattr(ls.ModelRegistry, "get_text_model",
                        staticmethod(lambda: loaded.append(1)))

    svc = ls.LibraryService(db_client=MagicMock())
    sent = _frames(svc)
    job = JobTracker().create_job("/music", "acct_acc1")
    monkeypatch.setattr(
        type(svc), "_files_to_index",
        lambda self, folder, coll, *, append: ([], 9088, 9088),
    )

    _run(svc._run_indexing_job(job, False, account_id="acc1", append=True))

    assert loaded == [], "the diff decides whether the model is needed at all"
    assert job.overall_status == IndexStatus.COMPLETED
    assert sent[-1]["tracks_skipped"] == 9088
    assert all(sp.status == IndexStatus.COMPLETED for sp in job.stages.values())
