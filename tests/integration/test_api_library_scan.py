"""POST /library/scan — the look-before-you-leap half of "rescan library".

Pressing rescan used to create an indexing job immediately: the progress UI
appeared before anything had been counted, and the job warmed the GPU text
model before it even knew whether there was anything to index. Scanning is now
its own endpoint — one directory walk, a live count, and a verdict — and it
touches neither the job tracker nor the models.

The gate is the same per-account grant as /library/index, and it is checked
before anything else, so these tests need no Qdrant.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.api.main import create_app
from app.resources.metadata_db import MetadataDB

JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIX_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "scan.db")
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    MetadataDB.set_instance_config(mode="server", created_at=1.0)
    yield
    MetadataDB._reset_for_tests()


@pytest.fixture
def known(monkeypatch):
    """Stub the SQLite mirror that answers 'what is already indexed'."""

    def _install(paths):
        monkeypatch.setattr(
            MetadataDB, "get_light_points",
            classmethod(lambda cls, collection: [
                (f"t{i}", {"file_path": p}) for i, p in enumerate(paths)
            ]),
        )

    _install([])
    return _install


def _as(user):
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: user
    return app


def _member(index_root=None, uid="member-1"):
    return SimpleNamespace(
        id=uid, email="m@x", role="member", created_at=0.0,
        last_login_at=None, premium=False, index_root=index_root,
    )


def _touch(root, name):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


def _scan(client, path):
    return client.post("/api/v1/library/scan", json={"folder_path": str(path)})


def _frames(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class TestGate:
    """Identical to /library/index — a scan reveals what is on the host disk."""

    def test_member_without_a_grant_is_refused(self, tmp_path, known):
        with TestClient(_as(_member())) as c:
            assert _scan(c, tmp_path).status_code == 403

    def test_the_refusal_names_no_path(self, tmp_path, known, monkeypatch):
        monkeypatch.setenv("MEMBER_INDEX_ROOT", "/music")
        with TestClient(_as(_member())) as c:
            body = _scan(c, tmp_path).json()["detail"]
        assert "/music" not in body

    def test_a_member_may_not_escape_its_grant(self, tmp_path, known):
        grant = tmp_path / "mine"
        grant.mkdir()
        (tmp_path / "theirs").mkdir()
        with TestClient(_as(_member(index_root=str(grant)))) as c:
            assert _scan(c, tmp_path / "theirs").status_code == 403

    def test_a_missing_folder_is_a_clean_400(self, tmp_path, known):
        root = tmp_path / "music"
        root.mkdir()
        with TestClient(_as(_member(index_root=str(root)))) as c:
            assert _scan(c, root / "gone").status_code == 400


class TestResult:
    def test_the_verdict_counts_new_files_against_the_whole_root(self, tmp_path, known):
        root = tmp_path / "music"
        old = _touch(root, "Album/01 - Old.mp3")
        _touch(root, "Album/02 - Fresh.mp3")
        _touch(root, "Album/03 - Fresh.flac")
        known([str(old)])

        with TestClient(_as(_member(index_root=str(root)))) as c:
            frames = _frames(_scan(c, root))

        assert frames[-1] == {"type": "done", "seen": 3, "new_count": 2, "rejected": {}}

    def test_nothing_new_is_a_result_not_an_error(self, tmp_path, known):
        root = tmp_path / "music"
        old = _touch(root, "Album/01 - Old.mp3")
        known([str(old)])

        with TestClient(_as(_member(index_root=str(root)))) as c:
            frames = _frames(_scan(c, root))

        assert frames[-1] == {"type": "done", "seen": 1, "new_count": 0, "rejected": {}}

    def test_the_count_streams_while_the_walk_runs(self, tmp_path, known, monkeypatch):
        monkeypatch.setattr("app.api.routes.library._SCAN_PROGRESS_EVERY", 2)
        root = tmp_path / "music"
        for i in range(5):
            _touch(root, f"Album/{i:02d} - Song.mp3")

        with TestClient(_as(_member(index_root=str(root)))) as c:
            frames = _frames(_scan(c, root))

        progress = [f["seen"] for f in frames if f["type"] == "progress"]
        assert progress == [2, 4, 5]
        assert frames[-1]["type"] == "done"

    def test_an_unmounted_library_refuses_instead_of_reporting_zero(self, tmp_path, known):
        """/mnt/data carries nofail, so a boot-order race leaves an empty
        directory where 220 GB of music should be — indistinguishable from
        'the user deleted everything'."""
        root = tmp_path / "music"
        root.mkdir()
        known(["/music/Music/still-indexed.mp3"])

        with TestClient(_as(_member(index_root=str(root)))) as c:
            frames = _frames(_scan(c, root))

        assert frames[-1] == {"type": "error", "code": "mount_empty"}


class TestCosts:
    def test_scanning_never_warms_the_text_model(self, tmp_path, known, monkeypatch):
        """The whole point of the separate step: find out whether there is any
        work BEFORE spending tens of seconds loading a model onto the GPU."""
        from app.resources.model_registry import ModelRegistry

        def boom(*a, **kw):
            raise AssertionError("the scan must not touch the text model")

        monkeypatch.setattr(ModelRegistry, "get_text_model", classmethod(boom))
        root = tmp_path / "music"
        _touch(root, "Album/01 - Song.mp3")

        with TestClient(_as(_member(index_root=str(root)))) as c:
            frames = _frames(_scan(c, root))

        assert frames[-1]["new_count"] == 1
