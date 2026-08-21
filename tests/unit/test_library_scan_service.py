"""Folder discovery behind the manual "rescan library" button.

The library is 220 GB on a spinning disk, so the contract that matters is that
a rescan costs one directory walk and nothing more: already-indexed files must
be dropped by PATH, before anyone opens them to read tags.
"""
from pathlib import Path

import pytest

from app.services.library_scan_service import (
    INDEXABLE_SUFFIXES,
    MountLooksEmpty,
    discover_new_files,
)


def _touch(root: Path, name: str) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


def test_finds_new_audio_file(tmp_path):
    track = _touch(tmp_path, "Album/01 - Song.mp3")

    result = discover_new_files(str(tmp_path), known_paths=set())

    assert result.new_files == [str(track)]


def test_ignores_extensions_the_indexer_cannot_handle(tmp_path):
    _touch(tmp_path, "Album/cover.jpg")
    _touch(tmp_path, "Album/notes.txt")
    _touch(tmp_path, "Album/rip.wav")
    keeper = _touch(tmp_path, "Album/02 - Song.flac")

    result = discover_new_files(str(tmp_path), known_paths=set())

    assert result.new_files == [str(keeper)]
    assert ".wav" not in INDEXABLE_SUFFIXES


def test_already_indexed_paths_are_not_reported(tmp_path):
    old = _touch(tmp_path, "Album/01 - Old.mp3")
    fresh = _touch(tmp_path, "Album/02 - Fresh.mp3")

    result = discover_new_files(str(tmp_path), known_paths={str(old)})

    assert result.new_files == [str(fresh)]


def test_walks_nested_directories(tmp_path):
    deep = _touch(tmp_path, "Artist/Album/Disc 1/03 - Deep.m4a")

    result = discover_new_files(str(tmp_path), known_paths=set())

    assert result.new_files == [str(deep)]


def test_reports_how_much_was_seen_not_just_what_is_new(tmp_path):
    old = _touch(tmp_path, "Album/01 - Old.mp3")
    _touch(tmp_path, "Album/02 - Fresh.mp3")

    result = discover_new_files(str(tmp_path), known_paths={str(old)})

    assert result.seen == 2
    assert len(result.new_files) == 1


def test_empty_root_with_a_non_empty_library_is_refused(tmp_path):
    """An unmounted disk looks exactly like 'the user deleted everything'."""
    with pytest.raises(MountLooksEmpty):
        discover_new_files(str(tmp_path), known_paths={"/music/Music/gone.mp3"})


def test_missing_root_with_a_non_empty_library_is_refused(tmp_path):
    with pytest.raises(MountLooksEmpty):
        discover_new_files(
            str(tmp_path / "nope"), known_paths={"/music/Music/gone.mp3"}
        )


def test_empty_root_with_an_empty_library_is_allowed(tmp_path):
    result = discover_new_files(str(tmp_path), known_paths=set())

    assert result.new_files == []
    assert result.seen == 0


def test_result_is_ordered_so_progress_is_stable(tmp_path):
    _touch(tmp_path, "B Album/01.mp3")
    _touch(tmp_path, "A Album/01.mp3")

    result = discover_new_files(str(tmp_path), known_paths=set())

    assert result.new_files == sorted(result.new_files)
