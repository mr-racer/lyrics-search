"""Audio metadata readers (Mutagen) + genre normalisation.

Extracted from legacy file_processor/utils.py during Refactor 3.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

logger = logging.getLogger(__name__)

_CURRENT_YEAR = datetime.datetime.now().year
_MIN_YEAR = 1900

genre_map = {
    # Pop
    'pop': 'Pop', 'поп': 'Pop', 'dance-pop': 'Pop', 'indie pop': 'Pop',
    'art pop': 'Pop', 'adult contemporary': 'Pop', 'bubblegum': 'Pop',
    'instrumental pop': 'Pop', 'pop/indie': 'Pop', 'alternative pop': 'Pop',

    # Rock
    'rock': 'Rock', 'classic rock': 'Rock', 'hard rock': 'Rock',
    'indie rock': 'Rock', 'alternative rock': 'Rock', 'alt. rock': 'Rock',
    'pop rock': 'Rock', 'blues rock': 'Rock', 'grunge': 'Rock',
    'post-grunge': 'Rock', 'post grunge': 'Rock', 'progressive rock': 'Rock',
    'album rock': 'Rock', 'soft rock': 'Rock', 'pop-rock': 'Rock',
    'pop/rock': 'Rock', 'blues/pop rock': 'Rock', 'dance-punk': 'Rock',
    'acid punk': 'Rock', 'stoner rock': 'Rock', 'alternative metal': 'Rock',
    'nu metal': 'Nu-Metal', 'goth rock': 'Rock',

    # Electronic
    'electronic': 'Electronic', 'synthpop': 'Electronic', 'synth-pop': 'Electronic',
    'electro': 'Electronic', 'electro house': 'Electronic', 'french house': 'Electronic',
    'house': 'Electronic', 'progressive house': 'Electronic', 'dubstep': 'Electronic',
    'brostep': 'Electronic', 'eurodance': 'Electronic', 'club': 'Electronic',
    'downtempo': 'Electronic', 'ambient': 'Electronic', 'nu-disco': 'Electronic',
    'disco': 'Electronic', 'trance': 'Electronic', 'dance & dj': 'Electronic',
    'club/dance': 'Electronic', 'dance': 'Electronic',

    # Hip-Hop
    'hip hop': 'Hip-Hop', 'rap': 'Hip-Hop', 'hip-hop': 'Hip-Hop',
    'east coast hip hop': 'Hip-Hop', 'alternative hip hop': 'Hip-Hop',

    # R&B / Soul
    'r&b': 'R&B/Soul', 'soul': 'R&B/Soul', 'contemporary r&b': 'R&B/Soul',
    'alternative r&b': 'R&B/Soul', 'funk': 'R&B/Soul', 'boogie': 'R&B/Soul',
    'r&b/soul': 'R&B/Soul', 'acid jazz': 'R&B/Soul', 'blues': 'R&B/Soul',

    # Alternative
    'alternative': 'Alternative', 'new wave': 'Alternative', 'indie': 'Alternative',
    'experimental': 'Alternative',
    'alternative & indie': 'Alternative',

    # Soundtrack
    'soundtrack': 'Soundtrack',
}

NOISE = {'none', 'miscellaneous', 'abstract', 'aggressive', '00s', '80s',
         '5+ wochen', 'adam levine', 'country & folk', 'электронная музыка',
         'pop, miscellaneous', 'pop soul r&b'}


def validate_year(raw: str | None) -> int | None:
    """Return year as int only if it looks sane (1900–present), else None."""
    if not raw:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(raw))
    if m:
        year = int(m.group(1))
        if _MIN_YEAR <= year <= _CURRENT_YEAR:
            return year
    return None


def parse_track_index(raw) -> int | None:
    """Parse a track- or disc-index tag value into a positive int, or None.

    Handles the shapes mutagen returns across containers:
      - FLAC / MP3 (EasyID3): a string like ``"5"`` or ``"5/12"`` (index / total)
      - MP4 (``trkn`` / ``disk``): an ``(index, total)`` tuple, or a bare int
    Returns None for missing, zero, or unparseable values.
    """
    if raw is None or isinstance(raw, bool):  # bool is an int subclass — exclude
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, (tuple, list)):
        return parse_track_index(raw[0]) if raw else None
    if isinstance(raw, str):
        head = raw.strip().split("/", 1)[0].strip()
        try:
            n = int(head)
        except ValueError:
            return None
        return n if n > 0 else None
    return None


def get_flac_metadata(filepath: str) -> dict:
    audio = FLAC(filepath)
    duration = audio.info.length

    # Compute actual FLAC bitrate from stream parameters (variable, lossless)
    info = audio.info
    bitrate_kbps: int | None = None
    if duration and getattr(info, "total_samples", None) and getattr(info, "bits_per_sample", None):
        bitrate_kbps = round(info.total_samples * info.bits_per_sample * info.channels / duration / 1000)

    raw_year = (audio.get("date") or audio.get("year") or [""])[0]
    return {
        "title":        (audio.get("title")  or [""])[0].strip(),
        "artist":       (audio.get("artist") or [""])[0].strip(),
        "album_artist": (audio.get("albumartist") or [""])[0].strip() or None,
        "album":        (audio.get("album")  or [""])[0].strip(),
        "year":         validate_year(raw_year),
        "genre":        (audio.get("genre")  or [""])[0].strip() or None,
        "duration":     round(duration),
        "bitrate_kbps": bitrate_kbps,
        "track_number": parse_track_index((audio.get("tracknumber") or [None])[0]),
        "disc_number":  parse_track_index((audio.get("discnumber")  or [None])[0]),
    }


def get_alac_metadata(filepath: str) -> dict:
    audio = MP4(filepath)
    duration = audio.info.length

    # mutagen reports M4A bitrate in bps; ALAC is typically 600–1400 kbps, AAC 128–320 kbps
    raw_bitrate = getattr(audio.info, "bitrate", None)
    bitrate_kbps = round(raw_bitrate / 1000) if raw_bitrate else None

    def _tag(key: str) -> str:
        val = audio.tags.get(key) or [""]
        return val[0].strip() if isinstance(val[0], str) else str(val[0]).strip()

    # mutagen reports the sample-entry codec: "alac" or "mp4a.40.2" (AAC).
    # Stored in the payload so the stream endpoint can decide ALAC→FLAC
    # transcoding without an ffprobe on the request hot path.
    codec = getattr(audio.info, "codec", None)

    raw_year = _tag("©day")
    return {
        "title":        _tag("©nam"),
        "artist":       _tag("©ART"),
        "album_artist": _tag("aART") or None,
        "album":        _tag("©alb"),
        "year":         validate_year(raw_year),
        "genre":        _tag("©gen") or None,
        "duration":     round(duration),
        "bitrate_kbps": bitrate_kbps,
        "audio_codec":  codec.lower() if isinstance(codec, str) else None,
        # MP4 stores track/disc as a list of (index, total) tuples.
        "track_number": parse_track_index((audio.tags.get("trkn") or [None])[0]),
        "disc_number":  parse_track_index((audio.tags.get("disk") or [None])[0]),
    }


def get_mp3_metadata(filepath: str) -> dict:
    try:
        audio = EasyID3(filepath)
    except ID3NoHeaderError:
        audio = {}

    # EasyID3 exposes 'length' in milliseconds as a string
    raw_len = (audio.get("length") or [""])[0]
    duration: int | None = None
    if raw_len:
        try:
            duration = round(int(raw_len) / 1000)
        except ValueError:
            pass

    # Fallback: read from mutagen.mp3 header (no decode needed, header-only)
    if duration is None:
        from mutagen.mp3 import MP3
        duration = round(MP3(filepath).info.length)

    raw_year = (audio.get("date") or audio.get("originaldate") or [""])[0]
    return {
        "title":        (audio.get("title")  or [""])[0].strip(),
        "artist":       (audio.get("artist") or [""])[0].strip(),
        "album_artist": (audio.get("albumartist") or [""])[0].strip() or None,
        "album":        (audio.get("album")  or [""])[0].strip(),
        "year":         validate_year(raw_year),
        "genre":        (audio.get("genre")  or [""])[0].strip() or None,
        "duration":     duration,
        "track_number": parse_track_index((audio.get("tracknumber") or [None])[0]),
        "disc_number":  parse_track_index((audio.get("discnumber")  or [None])[0]),
    }


def read_embedded_lyrics(filepath: Path) -> str | None:
    """Return lyrics embedded in the file's tags, or None.

    Lets Yandex-imported tracks (which carry lyrics written by the import
    downloader) skip the online lyrics fetch. Reads USLT (MP3), ``LYRICS``/
    ``lyrics`` (FLAC/Vorbis), and ``©lyr`` (MP4/M4A). Best-effort: any read
    error returns None so the caller falls back to the online fetchers.
    """
    suffix = filepath.suffix.lower()
    try:
        if suffix == ".flac":
            val = FLAC(str(filepath)).get("lyrics") or FLAC(str(filepath)).get("LYRICS")
            text = val[0] if val else None
        elif suffix == ".m4a":
            val = MP4(str(filepath)).tags.get("\xa9lyr") if MP4(str(filepath)).tags else None
            text = val[0] if val else None
        elif suffix == ".mp3":
            from mutagen.id3 import ID3, ID3NoHeaderError
            try:
                id3 = ID3(str(filepath))
            except ID3NoHeaderError:
                return None
            frames = id3.getall("USLT")
            text = frames[0].text if frames else None
        else:
            return None
    except Exception:
        return None
    text = (text or "").strip()
    return text or None


def get_metadata(filepath: Path) -> dict | None:
    """Dispatch to format-specific readers; return None on failure."""
    try:
        if filepath.suffix.lower() == ".flac":
            return get_flac_metadata(str(filepath))
        elif filepath.suffix.lower() == ".m4a":
            return get_alac_metadata(str(filepath))
        elif filepath.suffix.lower() == ".mp3":
            return get_mp3_metadata(str(filepath))
    except Exception as e:
        logger.warning("[metadata] failed to read %s: %s", filepath.name, e)
    return None


def normalize_genre(raw: str | None):
    if raw is None:
        return 'Other'

    raw_lower = raw.lower().strip()

    if raw_lower in NOISE:
        return 'Other'

    # Exact match
    if raw_lower in genre_map:
        return genre_map[raw_lower]

    # Multi-genre strings — take first
    first = re.split(r'[,/]', raw_lower)[0].strip()
    if first in genre_map:
        return genre_map[first]

    # Keyword search
    for keyword, genre in [
        ('hip hop', 'Hip-Hop'), ('hip-hop', 'Hip-Hop'), ('rap', 'Hip-Hop'),
        ('nu metal', 'Nu-Metal'), ('metal', 'Rock'),
        ('rock', 'Rock'), ('pop', 'Pop'),
        ('electronic', 'Electronic'), ('house', 'Electronic'),
        ('r&b', 'R&B/Soul'), ('soul', 'R&B/Soul'), ('funk', 'R&B/Soul'),
        ('blues', 'Blues'), ('indie', 'Alternative'),
        ('dance', 'Dance'), ('synth', 'Electronic'),
    ]:
        if keyword in raw_lower:
            return genre

    return 'Other'
