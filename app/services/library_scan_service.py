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
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

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
    so the UI can say "9088 files, 3 new" instead of just "3".

    ``rejected`` maps a reason to a count for files that are new on disk but
    that indexing would drop anyway — reported so the user is not invited to
    add music that cannot arrive. It is empty unless the caller asked for the
    duration screen.
    """

    new_files: list[str]
    seen: int
    rejected: dict = field(default_factory=dict)


def screen_by_duration(paths: list[str]) -> tuple[list[str], dict]:
    """Split ``paths`` into (indexable, ``{reason: count}``) by playing time.

    ``prepare_metadata`` drops anything over ``MAX_DURATION`` and says nothing,
    so a file over the cap is new on disk forever: the path diff offers it,
    indexing discards it, the next rescan offers it again. On one real library
    that was 93 of the 94 files a rescan kept promising.

    Only the files the diff already called NEW are opened, so this costs one
    header read each on the rare occasion there is anything to read at all.
    A file whose duration cannot be read is kept — the indexer has its own
    opinion about those, and guessing here would hide real music.
    """
    from app.resources.qdrant_payload import MAX_DURATION

    try:
        from mutagen import File as MutagenFile
    except Exception:            # pragma: no cover - mutagen is a hard dep
        return list(paths), {}

    keep: list[str] = []
    rejected: dict[str, int] = {}
    for path in paths:
        try:
            audio = MutagenFile(path)
            duration = float(getattr(getattr(audio, "info", None), "length", 0) or 0)
        except Exception:
            keep.append(path)
            continue
        if duration > MAX_DURATION:
            rejected["too_long"] = rejected.get("too_long", 0) + 1
        else:
            keep.append(path)
    return keep, rejected


def discover_new_files(
    root: str,
    known_paths: Iterable[str],
    *,
    suffixes: tuple[str, ...] = INDEXABLE_SUFFIXES,
    on_progress: Optional[Callable[[int], None]] = None,
    progress_every: int = 200,
    screen_durations: bool = False,
) -> ScanResult:
    """Walk ``root`` and return the indexable files absent from ``known_paths``.

    ``known_paths`` holds container-side ``file_path`` values straight out of
    the library's SQLite mirror; comparison is exact-string, matching how
    ``_split_already_indexed`` dedupes downstream.

    ``screen_durations`` additionally opens each NEW file and drops the ones
    over the indexer's duration cap, so the count returned is what indexing
    can actually ingest rather than what the directory holds. Off by default:
    a first-time index tag-reads every file straight afterwards anyway, and
    doubling that I/O on a spinning disk buys nothing there.

    ``on_progress`` receives the running count of indexable files seen, every
    ``progress_every`` of them and once more at the end. The walk is the slow
    half of a rescan on a spinning disk, and a UI that can show "4213 files"
    instead of a spinner is the difference between "working" and "hung". The
    final call is skipped when it would only repeat the previous number.
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
    reported = -1
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if not name.lower().endswith(suffixes):
                continue
            seen += 1
            full = os.path.join(dirpath, name)
            if full not in known:
                new_files.append(full)
            if on_progress is not None and seen % progress_every == 0:
                on_progress(seen)
                reported = seen

    if on_progress is not None and seen != reported:
        on_progress(seen)

    new_files.sort()
    rejected: dict = {}
    if screen_durations and new_files:
        new_files, rejected = screen_by_duration(new_files)
    return ScanResult(new_files=new_files, seen=seen, rejected=rejected)
