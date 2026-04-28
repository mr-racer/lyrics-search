from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from tqdm.auto import tqdm
import datetime
import time
import re

import syncedlyrics
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

PROVIDERS = ["Musixmatch", "Lrclib", "NetEase", "Megalobiz"]
TIME_BETWEEN_REQUESTS_STANDARD = 0.15
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

# FILE OPERATING FUNCTIONS

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


def get_lyrics(title: str, artist: str, better_lyrics_quality: bool) -> str | None:
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

    return {**meta, "lyrics": lyrics}


def fetch_lyrics_bulk(music_folder: str, workers: int = 8, better_lyrics_quality: bool = False):
    """Scan folder, enrich & fetch lyrics in parallel, return results keyed by artist/title."""
    
    if better_lyrics_quality:
        workers = 1

    audio_files = [
        p for p in Path(music_folder).rglob("*")
        if p.suffix.lower() in (".flac", ".m4a", ".mp3")
    ]
    print(f"Найдено файлов: {len(audio_files)}, воркеров: {workers}")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f, better_lyrics_quality): f for f in audio_files}
        for future in tqdm(as_completed(futures), total=len(audio_files)):
            meta = future.result()
            if meta:
                key = f"{meta['artist']} — {meta['title']}"
                results.setdefault(key, meta)

    print(f"\nГотово: {len(results)}/{len(audio_files)} текстов найдено")
    return results
