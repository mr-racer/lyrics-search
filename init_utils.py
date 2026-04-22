import json
import re
import time
import datetime
import syncedlyrics
import musicbrainzngs
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient, models
import uuid
from tqdm.auto import tqdm

MP4_TAG_MAP = {"title": "©nam", "artist": "©ART", "album": "©alb"}

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


# ── MusicBrainz setup ───────────────────────────────────────────────────────
musicbrainzngs.set_useragent("MusicMetadataFetcher", "1.0", "your@email.com")

_CURRENT_YEAR = datetime.datetime.now().year
_MIN_YEAR = 1900


def _validate_year(raw: str | None) -> int | None:
    """Return year as int only if it looks sane (1900–present), else None."""
    if not raw:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(raw))
    if m:
        year = int(m.group(1))
        if _MIN_YEAR <= year <= _CURRENT_YEAR:
            return year
    return None


# ── Format-specific local readers ───────────────────────────────────────────

def _get_flac_metadata(filepath: str) -> dict:
    audio = FLAC(filepath)
    duration = audio.info.length  # seconds, float — always present in FLAC header

    raw_year = (audio.get("date") or audio.get("year") or [""])[0]
    return {
        "title":    (audio.get("title")  or [""])[0].strip(),
        "artist":   (audio.get("artist") or [""])[0].strip(),
        "album":    (audio.get("album")  or [""])[0].strip(),
        "year":     _validate_year(raw_year),
        "genre":    (audio.get("genre")  or [""])[0].strip() or None,
        "duration": round(duration),
    }


def _get_alac_metadata(filepath: str) -> dict:
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
        "year":     _validate_year(raw_year),
        "genre":    _tag("©gen") or None,
        "duration": round(duration),
    }


def _get_mp3_metadata(filepath: str) -> dict:
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
        "year":     _validate_year(raw_year),
        "genre":    (audio.get("genre")  or [""])[0].strip() or None,
        "duration": duration,
    }


def _get_metadata(filepath: Path) -> dict | None:
    """Dispatch to format-specific readers; return None on failure."""
    try:
        if filepath.suffix.lower() == ".flac":
            return _get_flac_metadata(str(filepath))
        elif filepath.suffix.lower() == ".m4a":
            return _get_alac_metadata(str(filepath))
        elif filepath.suffix.lower() == ".mp3":
            return _get_mp3_metadata(str(filepath))
    except Exception as e:
        print(f" Ошибка метаданных {filepath.name}: {e}")
    return None


# ── MusicBrainz online enrichment ───────────────────────────────────────────

def _mb_enrich(title: str, artist: str, meta: dict) -> dict:
    """Fill missing fields (year, genre, producer) from MusicBrainz.

    Only queries if at least one field is still absent. Adds a 1-second polite delay
    (MB rate-limit is 1 req/s for non-authenticated clients).

    Args:
        title:  Track title.
        artist: Artist name.
        meta:   Existing metadata dict (modified in-place and returned).

    Returns:
        Enriched metadata dict.
    """
    needs_track = not meta.get("year") or not meta.get("genre") or not meta.get("producer")

    if not needs_track:
        return meta

    time.sleep(1)  # respect MB rate limit

    # ── Recording lookup (year, genre/tags, producer) ────────────────────
    if needs_track:
        try:
            result = musicbrainzngs.search_recordings(
                recording=title,
                artist=artist,
                limit=1,
            )
            recordings = result.get("recording-list", [])
            if recordings:
                rec = recordings[0]

                if not meta.get("year"):
                    meta["year"] = _validate_year(rec.get("first-release-date"))

                if not meta.get("genre"):
                    tags = rec.get("tag-list", [])
                    if tags:
                        # pick the tag with the highest vote count
                        best = max(tags, key=lambda t: int(t.get("count", 0)))
                        meta["genre"] = best.get("name")

                if not meta.get("producer"):
                    for rel in rec.get("relation-list", []):
                        if rel.get("target-type") == "artist":
                            for r in rel.get("relation", []):
                                if r.get("type") == "producer":
                                    meta["producer"] = (
                                        r.get("artist", {}).get("name")
                                    )
                                    break
        except Exception:
            pass  # MB lookup is best-effort

    return meta


# ── Per-file pipeline ────────────────────────────────────────────────────────

def _get_lyrics(title: str, artist: str) -> str | None:
    try:
        return syncedlyrics.search(
            f"{title} {artist}",
            providers=["Lrclib", "NetEase", "Megalobiz"],
            plain_only=True,
        )
    except Exception:
        return None


def _process_file(filepath: Path) -> dict | None:
    """Load metadata, enrich online where needed, then fetch lyrics.

    Returns:
        Full metadata dict or None if the track should be skipped.
    """
    meta = _get_metadata(filepath)
    if not meta or not meta["title"] or not meta["artist"]:
        print(f"  Пропуск: {filepath.name}")
        return None

    # Online enrichment — fills gaps left by local tags
    meta = _mb_enrich(meta["title"], meta["artist"], meta)

    lyrics = _get_lyrics(meta["title"], meta["artist"])
    if not lyrics:
        print(f"✗ {meta['artist']} — {meta['title']}")
        return None

    return {**meta, "lyrics": lyrics}


# ── Bulk processor ───────────────────────────────────────────────────────────

def _fetch_lyrics_bulk(music_folder: str, output_file: str = "lyrics.json", workers: int = 8):
    """Scan folder, enrich & fetch lyrics in parallel, return results keyed by artist/title."""
    audio_files = [
        p for p in Path(music_folder).rglob("*")
        if p.suffix.lower() in (".flac", ".m4a", ".mp3")
    ]
    print(f"Найдено файлов: {len(audio_files)}, воркеров: {workers}")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, f): f for f in audio_files}
        for future in tqdm(as_completed(futures), total=len(audio_files)):
            meta = future.result()
            if meta:
                key = f"{meta['artist']} — {meta['title']}"
                results.setdefault(key, meta)

    found = sum(1 for v in results.values() if v.get("lyrics"))
    print(f"\nГотово: {found}/{len(results)} текстов найдено")
    return results


def _normalize_genre(raw: str | None):
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

# ── High-level API ───────────────────────────────────────────────────────────

class FileProcessor:
    """Collects music file metadata and fetches lyrics for tracks in a folder."""

    def __init__(self):
        self.metadata = []

    def process_folder(self, music_folder: str):
        processed_files = _fetch_lyrics_bulk(music_folder)
        self.metadata = processed_files
        return processed_files

    def to_json(self, directory: Path = Path.cwd() / "metadata.json"):
        if self.metadata:
            try:
                path_directory = Path(directory)
                with open(str(path_directory), "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                    print(f"Файл с метаданными был успешно сохранен в {str(path_directory)}")
            except Exception as e:
                print("Ошибка сохранения результатов! Введен неверный путь. ", e)

    def info(self):
        print(f"Инициализировано {len(self.metadata)} музыкальных файлов")
        print(f"У ... файлов были метаданные, для ... они были получены онлайн.")
        print(f"Для ... файлов успешно были получены тексты")


# ── Qdrant / vector DB ───────────────────────────────────────────────────────

class LyricsDB:
    """Manage a Qdrant collection with hybrid dense (Jina) and sparse (BM25) lyric embeddings."""

    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            self.qdrant_client.get_collections()
        except Exception as e:
            raise ConnectionError(
                "Qdrant is down. Please, disable VPN or restart Docker"
            ) from e

    def _create_collection(self):
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "jina-v3": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        print(f"Collection {self.collection_name} has been successfully created")

    def _upsert_in_batches(self, data:list, batch_size=32):
        points = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "jina-v3": models.Document(
                        text=song_info["lyrics"],
                        model="jinaai/jina-embeddings-v2-small-en",
                    ),
                    "bm25": models.Document(
                        text=song_info["lyrics"],
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    "lyrics":    song_info["lyrics"],
                    "title":     song_info["title"],
                    "artist":    song_info["artist"],
                    "album":     song_info["album"],
                    "year":      song_info.get("year"),
                    "genre":     _normalize_genre(song_info.get("genre")),
                    "duration":  song_info.get("duration"),
                },
            )
            for song, song_info in data.items() if len(song_info["lyrics"].split()) < 1300
        ]
        for i in tqdm(range(0, len(points), batch_size), ):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    def fit(self, data:list):
        self._create_collection()
        self._upsert_in_batches(data)
        print("Data was fitted in DB successfully")

    def _build_filter(
        self,
        artist: str | None = None,
        album: str | None = None,
        title: str | None = None,
        genre: str | list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> models.Filter | None:
        conditions = []

        if artist:
            conditions.append(models.FieldCondition(key="artist", match=models.MatchValue(value=artist)))
        if album:
            conditions.append(models.FieldCondition(key="album", match=models.MatchValue(value=album)))
        if title:
            conditions.append(models.FieldCondition(key="title", match=models.MatchValue(value=title)))

        if genre:
            conditions.append(
                models.FieldCondition(
                    key="genre",
                    match=models.MatchAny(any=genre) if isinstance(genre, list) else models.MatchValue(value=genre),
                )
            )

        if year_from is not None or year_to is not None:
            conditions.append(
                models.FieldCondition(key="year", range=models.Range(gte=year_from, lte=year_to))
            )

        return models.Filter(must=conditions) if conditions else None


    def search(
        self,
        query: str,
        limit: int = 1,
        artist: str | None = None,
        album: str | None = None,
        title: str | None = None,
        genre: str | list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[models.ScoredPoint]:
        query_filter = self._build_filter(
            artist=artist,
            album=album,
            title=title,
            genre=genre,
            year_from=year_from,
            year_to=year_to,
        )

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="jinaai/jina-embeddings-v2-small-en",
                    ),
                    using="jina-v3",
                    limit=35,
                    filter=query_filter,   # ← фильтр на prefetch, а не только на финале
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="bm25",
                    limit=35,
                    filter=query_filter,   # ← то же самое
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return results.points