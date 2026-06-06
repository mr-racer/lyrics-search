"""One-shot generator for tiny audio test fixtures.

Run manually:
    python tests/fixtures/audio/generate_fixtures.py

Requires ffmpeg on PATH. Produces three 1-second silent files with valid
ID3/Vorbis/MP4 tags so that the metadata readers return non-empty title/artist
(a hard requirement to enter the indexing pipeline). DO NOT run on every test
invocation — fixtures are committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

OUT = Path(__file__).parent

COMMON_TAGS = [
    "-metadata", "title=Tiny Test Song",
    "-metadata", "artist=Tiny Test Artist",
    "-metadata", "album=Tiny Test Album",
    "-metadata", "date=2020",
]


def _gen(out_path: Path, codec: str, ext_flags: list[str]) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "1", "-c:a", codec, *ext_flags, *COMMON_TAGS, str(out_path),
        ],
        check=True, capture_output=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _gen(OUT / "tiny.flac", "flac", [])
    _gen(OUT / "tiny.mp3", "libmp3lame", ["-b:a", "32k"])
    _gen(OUT / "tiny.m4a", "aac", ["-b:a", "32k"])
    print("Generated:", *sorted(str(p) for p in OUT.glob("tiny.*")))


if __name__ == "__main__":
    main()
