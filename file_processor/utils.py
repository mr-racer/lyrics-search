from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from tqdm.auto import tqdm
import datetime
import hashlib
import time
import re

import json
import requests
import syncedlyrics
from mutagen import File as MutagenFile
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError, APIC

PROVIDERS = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz"]
TIME_BETWEEN_REQUESTS_STANDARD = 0.15
TIME_BETWEEN_REQUESTS_OVH = 0.4
TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS = 3
MAX_DURATION = 420 # filtering quite long songs

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
    'nu metal': 'Nu-Metal',

    # Electronic
    'electronic': 'Electronic', 'synthpop': 'Electronic', 'synth-pop': 'Electronic',
    'electro': 'Electronic', 'electro house': 'Electronic', 'french house': 'Electronic',
    'house': 'Electronic', 'progressive house': 'Electronic', 'dubstep': 'Electronic',
    'brostep': 'Electronic', 'eurodance': 'Electronic', 'club': 'Electronic',
    'downtempo': 'Electronic', 'ambient': 'Electronic', 'nu-disco': 'Electronic',
    'disco': 'Electronic', 'trance': 'Electronic', 'dance & dj': 'Electronic',
    'club/dance': 'Electronic',

    # Hip-Hop
    'hip hop': 'Hip-Hop', 'rap': 'Hip-Hop', 'hip-hop': 'Hip-Hop',
    'east coast hip hop': 'Hip-Hop', 'alternative hip hop': 'Hip-Hop',

    # R&B / Soul
    'r&b': 'R&B/Soul', 'soul': 'R&B/Soul', 'contemporary r&b': 'R&B/Soul',
    'alternative r&b': 'R&B/Soul', 'funk': 'R&B/Soul', 'boogie': 'R&B/Soul',
    'r&b/soul': 'R&B/Soul', 'acid jazz': 'R&B/Soul',

    # Alternative
    'alternative': 'Alternative', 'new wave': 'Alternative', 'indie': 'Alternative',
    'goth rock': 'Alternative', 'experimental': 'Alternative',
    'alternative & indie': 'Alternative',

    # Dance (самостоятельный)
    'dance': 'Dance',

    # Blues
    'blues': 'Blues',

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

# Directory where extracted cover art is saved
COVERS_DIR = Path(__file__).parent.parent / "frontend" / "covers"

# FILE OPERATING FUNCTIONS

def extract_cover_art(filepath: str) -> tuple[bytes, str] | None:
    """Extract embedded cover art from an audio file.

    Returns (image_bytes, mime_extension) or None if no artwork is found.
    Supported formats: MP3 (APIC), FLAC (embedded picture), M4A (covr).
    """
    audio = MutagenFile(filepath)
    if audio is None:
        return None

    # MP3 — APIC frames
    if hasattr(audio, "tags") and audio.tags is not None:
        for tag in audio.tags.values():
            if isinstance(tag, APIC):
                ext = "jpg" if tag.mime == "image/jpeg" else "png"
                return tag.data, ext

    # FLAC — embedded pictures
    if hasattr(audio, "pictures") and audio.pictures:
        pic = audio.pictures[0]
        ext = "png" if "png" in pic.mime else "jpg"
        return pic.data, ext

    # M4A/MP4 — covr atom
    if hasattr(audio, "get") and "covr" in audio:
        covers = audio["covr"]
        if covers:
            raw = covers[0]
            # MP4Cover data: first 32 bytes are a header, then the image
            # Try to detect format from content
            ext = "png" if raw.startswith(b'\x89PNG') else "jpg"
            return raw, ext

    return None


def save_cover_art(filepath: str, track_id: str) -> str | None:
    """Extract cover art and save to disk (content-addressed, deduplicated by SHA256).

    Returns relative URL path like '/covers/a3f2c1...jpg' or None.
    If an identical cover already exists on disk, returns the existing path — no duplicate write.
    """
    result = extract_cover_art(filepath)
    if result is None:
        return None

    image_data, ext = result
    COVERS_DIR.mkdir(parents=True, exist_ok=True)

    # Content-addressed filename: first 16 hex chars of SHA256
    content_hash = hashlib.sha256(image_data).hexdigest()[:16]
    dest = COVERS_DIR / f"{content_hash}.{ext}"

    if not dest.exists():
        dest.write_bytes(image_data)

    return f"/covers/{content_hash}.{ext}"


def get_flac_metadata(filepath: str) -> dict:
    audio = FLAC(filepath)
    duration = audio.info.length

    raw_year = (audio.get("date") or audio.get("year") or [""])[0]
    return {
        "title":    (audio.get("title")  or [""])[0].strip(),
        "artist":   (audio.get("artist") or [""])[0].strip(),
        "album":    (audio.get("album")  or [""])[0].strip(),
        "year":     validate_year(raw_year),
        "genre":    (audio.get("genre")  or [""])[0].strip() or None,
        "duration": round(duration),
    }


def get_alac_metadata(filepath: str) -> dict:
    audio = MP4(filepath)
    duration = audio.info.length

    def _tag(key: str) -> str:
        val = audio.tags.get(key) or [""]
        return val[0].strip() if isinstance(val[0], str) else str(val[0]).strip()

    raw_year = _tag("©day")
    return {
        "title":    _tag("©nam"),
        "artist":   _tag("©ART"),
        "album":    _tag("©alb"),
        "year":     validate_year(raw_year),
        "genre":    _tag("©gen") or None,
        "duration": round(duration),
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
        "title":    (audio.get("title")  or [""])[0].strip(),
        "artist":   (audio.get("artist") or [""])[0].strip(),
        "album":    (audio.get("album")  or [""])[0].strip(),
        "year":     validate_year(raw_year),
        "genre":    (audio.get("genre")  or [""])[0].strip() or None,
        "duration": duration,
    }

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
        print(f" Ошибка метаданных {filepath.name}: {e}")
    return None

def normalize_genre(raw: str | None):
    if raw is None:
        return 'Other'
    
    raw_lower = raw.lower().strip()
    
    if raw_lower in NOISE:
        return 'Other'
    
    # Точное совпадение
    if raw_lower in genre_map:
        return genre_map[raw_lower]
    
    # Мультижанровые строки — берём первый
    first = re.split(r'[,/]', raw_lower)[0].strip()
    if first in genre_map:
        return genre_map[first]
    
    # Поиск по ключевым словам
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


# ONLINE REQUESTING FUNCTIONS


def fetch_lyrics_ovh(artist: str, title: str) -> str | None:
    """Fetch lyrics from lyrics.ovh API. Returns plain lyrics string or None."""
    try:
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        resp = requests.get(url, timeout=5)
        time.sleep(TIME_BETWEEN_REQUESTS_OVH)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            lyrics = data["lyrics"]
            # Convert literal \n escape sequences to real newlines
            lyrics = lyrics.replace("\\n", "\n")
            return lyrics.strip()
    except Exception:
        pass
    return None


def get_lyrics(title: str, artist: str, better_lyrics_quality: bool) -> str | None:
    # Primary: try lyrics.ovh
    lyrics = fetch_lyrics_ovh(artist, title)
    if lyrics:
        return lyrics

    # Fallback: syncedlyrics
    if better_lyrics_quality:
        providers = PROVIDERS
        time_to_sleep = TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS
    else:
        providers = [x for x in PROVIDERS if x != "Musixmatch"]
        time_to_sleep = TIME_BETWEEN_REQUESTS_STANDARD

    try:
        lyrics = syncedlyrics.search(
            f"{title} {artist}",
            providers=providers,
            plain_only=True,
        )
        time.sleep(time_to_sleep)

        if not lyrics:
            return None
        lyrics = re.sub(r'\[.*?\]', '', lyrics)

    except Exception:
        return None

    return lyrics

def research_with_musixmatch(song: str) -> str:
    try:
        info = song.split(' — ', maxsplit=1)
        if len(info) < 2:
            return None
        new_text = syncedlyrics.search(
        f"{info[0]} {info[1]}",
        providers=['Musixmatch'],
        plain_only=True,
        )
        time.sleep(TIME_BETWEEN_REQUESTS_ENHANCED_LYRICS)

        
        if new_text:
            new_text = re.sub(r'\[.*?\]', '', new_text)
            return new_text
    except Exception as e:
        print(f"Не удалось обновить текст песни. {e}")
        return None


# MAIN FUNCTIONS


def process_file(filepath: Path, better_lyrics_quality: bool) -> dict | None:
    """Load metadata, enrich online where needed, then fetch lyrics.

    Returns:
        Full metadata dict or None if the track should be skipped.
    """
    meta = get_metadata(filepath)
    if not meta or not meta["title"] or not meta["artist"]:
        print(f"  Пропуск: {filepath.name}")
        return None

    lyrics = get_lyrics(meta["title"], meta["artist"], better_lyrics_quality)
    if not lyrics:
        print(f"✗ {meta['artist']} — {meta['title']}")
        return None

    if meta.get('genre'):
        meta['genre'] = normalize_genre(meta['genre'])

    return {**meta, "lyrics": lyrics, "file_path":str(filepath)}


def fetch_lyrics_bulk(music_folder: str, workers: int = 8, better_lyrics_quality: bool = False, progress_callback=None):
    """Scan folder, enrich & fetch lyrics in parallel, return results keyed by artist/title.

    Args:
        music_folder: Path to scan for audio files
        workers: Number of parallel workers (1 if better_lyrics_quality)
        better_lyrics_quality: Use all providers including Musixmatch
        progress_callback: Optional callback(current, total, message) called after each file
    """

    if better_lyrics_quality:
        workers = 1

    audio_files = [
        p for p in Path(music_folder).rglob("*")
        if p.suffix.lower() in (".flac", ".m4a", ".mp3")
    ]
    print(f"Найдено файлов: {len(audio_files)}, воркеров: {workers}")

    results = {}
    processed_count = 0
    found_count = 0
    not_found_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, better_lyrics_quality): f for f in audio_files}
        start_time = time.time()

        for future in tqdm(as_completed(futures), total=len(audio_files)):
            meta = future.result()
            processed_count += 1

            if meta:
                found_count += 1
                key = f"{meta['artist']} — {meta['title']}"
                results.setdefault(key, meta)
            else:
                not_found_count += 1

            if progress_callback:
                fp = futures[future]

                elapsed = time.time() - start_time
                rate_per_min = None
                eta_seconds = None
                if elapsed > 1 and processed_count > 1:
                    rate_per_sec = processed_count / elapsed
                    rate_per_min = round(rate_per_sec * 60, 1)
                    remaining = len(audio_files) - processed_count
                    eta_seconds = max(0, int(remaining / rate_per_sec)) if rate_per_sec > 0 else None

                if meta:
                    last_track = f"{meta['artist']} — {meta['title']}"
                    status = f"[{processed_count}/{len(audio_files)}] ✓ {last_track}"
                    last_success = True
                else:
                    last_track = fp.stem
                    status = f"[{processed_count}/{len(audio_files)}] ✗ {fp.name}"
                    last_success = False

                details = {
                    "found":        found_count,
                    "not_found":    not_found_count,
                    "rate_per_min": rate_per_min,
                    "eta_seconds":  eta_seconds,
                    "last_track":   last_track,
                    "last_success": last_success,
                }
                progress_callback(processed_count, len(audio_files), status, details)

    print(f"\nГотово: {len(results)}/{len(audio_files)} текстов найдено")
    return results
