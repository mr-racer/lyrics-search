"""Discovery half of the manual "rescan library" action.

Rescanning a by-reference library means answering one question cheaply: which
files under the mounted root are not in the collection yet? The expensive part
of the old path was never the directory walk — it was that ``index_folder``
read tags out of every file it found and only then discarded the ones already
indexed. On a 220 GB library that is thousands of file opens on a spinning
disk to discover, typically, nothing.

So the diff happens here, on PATHS alone, and the caller only ever opens files
that are genuinely new.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

# What LibraryService.index_folder is actually able to ingest. Reporting
# anything else as "new" would make every rescan rediscover the same file
# forever, since indexing would silently drop it again.
INDEXABLE_SUFFIXES = (".flac", ".m4a", ".mp3")


class MountLooksEmpty(RuntimeError):
    """The root is missing or empty while the collection still holds tracks.

    On this deployment the library lives on a bind-mounted disk carrying
    ``nofail``, so a boot-order race leaves an empty directory where 220 GB of
    music should be. That is indistinguishable from "the user deleted
    everything", and the honest response is to refuse rather than to report a
    cheerful "0 new tracks".
    """


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one walk. ``seen`` counts every indexable file under the root,
    so the UI can say "9088 files, 3 new" instead of just "3"."""

    new_files: list[str]
    seen: int


def discover_new_files(
    root: str,
    known_paths: Iterable[str],
    *,
    suffixes: tuple[str, ...] = INDEXABLE_SUFFIXES,
) -> ScanResult:
    """Walk ``root`` and return the indexable files absent from ``known_paths``.

    ``known_paths`` holds container-side ``file_path`` values straight out of
    the library's SQLite mirror; comparison is exact-string, matching how
    ``_split_already_indexed`` dedupes downstream.
    """
    known = set(known_paths)

    if not os.path.isdir(root):
        if known:
            raise MountLooksEmpty(f"library root is not a directory: {root}")
        return ScanResult(new_files=[], seen=0)

    with os.scandir(root) as it:
        root_is_empty = next(it, None) is None
    if root_is_empty and known:
        raise MountLooksEmpty(f"library root is empty: {root}")

    new_files: list[str] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.lower().endswith(suffixes):
                continue
            seen += 1
            full = os.path.join(dirpath, name)
            if full not in known:
                new_files.append(full)

    new_files.sort()
    return ScanResult(new_files=new_files, seen=seen)
