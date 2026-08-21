"""Appending to an existing library must not re-read every file's tags.

The old order was: walk the folder, open and tag-read ALL of it, then discard
what was already indexed. On a 220 GB by-reference library on a spinning disk
that is thousands of file opens to discover, usually, nothing — which is what
made "add music" feel like a full re-index. The diff now happens on paths.
"""
from pathlib import Path

import pytest

from app.resources.metadata_db import MetadataDB
from app.services.library_scan_service import MountLooksEmpty
from app.services.library_service import LibraryService


def _touch(root: Path, name: str) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


@pytest.fixture
def known(monkeypatch):
    """Stub the SQLite mirror that backs 'what is already indexed'."""

    def _install(paths):
        monkeypatch.setattr(
            MetadataDB,
            "get_light_points",
            classmethod(
                lambda cls, collection: [
                    (f"t{i}", {"file_path": p}) for i, p in enumerate(paths)
                ]
            ),
        )

    return _install


def test_append_hands_only_new_files_to_the_tag_reader(tmp_path, known):
    old = _touch(tmp_path, "Album/01 - Old.mp3")
    fresh = _touch(tmp_path, "Album/02 - Fresh.mp3")
    known([str(old)])

    files, seen, skipped = LibraryService()._files_to_index(
        str(tmp_path), "acct_x", append=True
    )

    assert [str(f) for f in files] == [str(fresh)]
    assert seen == 2
    assert skipped == 1


def test_append_with_nothing_new_hands_over_nothing(tmp_path, known):
    old = _touch(tmp_path, "Album/01 - Old.mp3")
    known([str(old)])

    files, seen, skipped = LibraryService()._files_to_index(
        str(tmp_path), "acct_x", append=True
    )

    assert files == []
    assert seen == 1
    assert skipped == 1


def test_a_full_rebuild_still_reads_everything(tmp_path, known):
    _touch(tmp_path, "Album/01 - Old.mp3")
    _touch(tmp_path, "Album/02 - Fresh.mp3")
    known([str(tmp_path / "Album" / "01 - Old.mp3")])

    files, seen, skipped = LibraryService()._files_to_index(
        str(tmp_path), "acct_x", append=False
    )

    assert len(files) == 2
    assert skipped == 0


def test_append_refuses_when_the_mount_vanished(tmp_path, known):
    """Empty folder + non-empty collection = the disk did not mount."""
    known(["/music/Music/still-indexed.mp3"])

    with pytest.raises(MountLooksEmpty):
        LibraryService()._files_to_index(str(tmp_path), "acct_x", append=True)


def test_lookup_failure_falls_back_to_indexing_everything(tmp_path, monkeypatch):
    """Losing the mirror must never silently swallow the user's new music."""
    _touch(tmp_path, "Album/01 - Song.mp3")

    def boom(cls, collection):
        raise RuntimeError("mirror unavailable")

    monkeypatch.setattr(MetadataDB, "get_light_points", classmethod(boom))

    files, seen, skipped = LibraryService()._files_to_index(
        str(tmp_path), "acct_x", append=True
    )

    assert len(files) == 1
    assert skipped == 0
