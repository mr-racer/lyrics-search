import numpy as np

import torch
import gc

import librosa
import laion_clap


from qdrant_client import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATA FUNCTIONS

MAX_DURATION = 420 # filtering quite long songs

def prepare_metadata(data: dict):
    data_prep = list(data.values())

    # Фильтрация
    filtered = [d for d in data_prep if d['duration'] <= MAX_DURATION and len(d['lyrics']) > 50]
    durations = np.array([d['duration'] for d in filtered])

    # Квартили
    p25 = np.percentile(durations, 25)
    p50 = np.percentile(durations, 50)
    p75 = np.percentile(durations, 75)
    iqr_custom = p50 - p25

    lower = p25 - 1.5 * iqr_custom
    upper = p75 + 1.5 * iqr_custom
    max_dur = durations.max()

    # Условия и метки бакетов
    conditions = [
        durations < lower,
        (durations >= lower) & (durations < p25),
        (durations >= p25) & (durations < p50),
        (durations >= p50) & (durations <= p75),
        (durations > p75) & (durations <= upper),
        durations > upper,
    ]
    labels = [
        f'0-{int(round(lower))}',
        f'{int(round(lower))}-{int(round(p25))}',
        f'{int(round(p25))}-{int(round(p50))}',
        f'{int(round(p50))}-{int(round(p75))}',
        f'{int(round(p75))}-{int(round(upper))}',
        f'{int(round(upper))}-{int(max_dur)}',
    ]
    buckets = np.select(conditions, labels, default='')
    buckets = list(map(lambda x: str(x), list(buckets)))

    # Собираем результат. `duration` остаётся числом (секунды) — Pydantic-модели
    # ниже по конвейеру (LikedSongTrack/RecentTrack/PlaylistTrack) ждут float.
    # `duration_range` — отдельный bucket-лейбл для фасета по длительности
    # (зеркалит пару year/year_range).
    result = [
        {**d, 'duration_range': bucket}
        for d, bucket in zip(filtered, buckets)
    ]

    for rec in result:
        rec['lyrics_chunked'] = tuple(set(rec['lyrics'].split('\n\n')))

    # добавление диапазонов лет
    for track in result:
        if track.get('year'):
            decade_start = (track['year'] // 10) * 10
            track['year_range'] = f"{decade_start}-{decade_start + 9}"

    return result


# CUSTOM EMBEDDINGS FUNCTION

from dataclasses import dataclass

@dataclass
class TrackFeatures:
    title: str
    artist: str
    vector_clap: list

def unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def get_clap_embedding_long(clap_model, y: np.ndarray, sr: int,
                             chunk_sec: int = 30, device=DEVICE) -> np.ndarray:
    """
    Делит длинное аудио на чанки по chunk_sec секунд,
    получает эмбеддинг каждого, усредняет.
    """
    
    chunk_len = sr * chunk_sec
    
    # Нарезаем на чанки
    chunks = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start : start + chunk_len]
        
        # Пропускаем слишком короткие хвосты (< 5 секунд)
        if len(chunk) < sr * 5:
            continue
        
        # Padding до фиксированной длины
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        
        chunks.append(chunk)
    
    if not chunks:
        print("Аудио слишком короткое")
        return None
    
    # Эмбеддинг каждого чанка
    batch = torch.from_numpy(np.stack(chunks)).to(device)  # (n_chunks, chunk_len)

    with torch.no_grad():
        embeddings = clap_model.get_audio_embedding_from_data(
            x=batch, use_tensor=True
        )  # (n_chunks, 512), tensor on GPU

    # Усредняем по чанкам, возвращаем numpy
    return embeddings.mean(dim=0).cpu().numpy()  # shape: (512,)

def extract_clap_features(path: str, model, duration: int = 300, device=DEVICE) -> np.ndarray:
    """CLAP отдельно — вызывается когда акустика уже готова."""
    y, sr = librosa.load(path, duration=duration, sr=48000, mono=True)
    clap_vec = get_clap_embedding_long(model, y, sr, chunk_sec=30, device=device)
    del y
    return unit_norm(clap_vec)


def _encode_clap(
    tracks: list[dict],
    model_clap=None,
    progress_callback=None,
) -> dict[tuple, np.ndarray]:
    """Encode audio files with CLAP.

    Args:
        tracks: list[dict] — уже готовые метаданные (из filtered).
                Каждый dict должен содержать 'file_path', 'artist', 'title'.
        model_clap: загруженная CLAP-модель (или None — загрузится сама).
        progress_callback: optional callable(current, total) called after each file.

    Returns:
        {(artist_lower, title_lower): np.ndarray} — маппинг ключей в CLAP-векторы.
    """
    import logging
    log = logging.getLogger(__name__)

    # Строим lookup: file_path → (artist_lower, title_lower) из готовых метаданных
    path_to_key = {}
    for t in tracks:
        fp = t.get("file_path")
        artist = (t.get("artist") or "").strip().lower()
        title = (t.get("title") or "").strip().lower()
        if fp and artist and title:
            path_to_key[fp] = (artist, title)

    if not path_to_key:
        log.warning("[CLAP] No tracks with valid file_path + artist + title — skipping")
        return {}

    if not model_clap:
        from app.resources.model_registry import ModelRegistry
        model_clap = ModelRegistry.load_clap()

    # Кодируем каждый файл отдельно (GPU — один поток), с защитой от ошибок
    clap_map: dict[tuple, np.ndarray] = {}
    total = len(path_to_key)
    for idx, (fp, key) in enumerate(path_to_key.items(), 1):
        try:
            vec = extract_clap_features(fp, model_clap, 300)
            if vec is not None:
                clap_map[key] = vec
        except Exception as e:
            log.warning("[CLAP] Failed to encode %s (%s — %s): %s", fp, *key, e)

        if progress_callback:
            progress_callback(idx, total)

        if idx % 50 == 0 or idx == total:
            log.info("[CLAP] Encoded %d / %d", idx, total)

    del model_clap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("[CLAP] Mapped %d / %d tracks", len(clap_map), total)
    return clap_map
# QDRANT FUNCTIONS

def build_text_for_embedding(track: dict) -> str:
    parts = []
    if track.get("title"):
        parts.append(f"title: {track['title']}")
    if track.get("artist"):
        parts.append(f"artist: {track['artist']}")
    if track.get("album"):
        parts.append(f"album: {track['album']}")
    if track.get("genre"):
        parts.append(f"genre: {track['genre']}")
    lyrics = track.get("lyrics", "").strip()
    if len(lyrics) > 20:
        parts.append(lyrics)
    return " | ".join(parts)


def build_filter(
    artist: str | None = None,
    album: str | None = None,
    title: str | None = None,
    genre: str | list[str] | None = None,
    year: int | None = None,
    year_ranges: list[str] | None = None,
    sonic_tags: list[str] | None = None,
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

    if year:
        conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))

    if year_ranges:
        conditions.append(models.FieldCondition(
            key="year_range", match=models.MatchAny(any=list(year_ranges)),
        ))

    if sonic_tags:
        for tag in sonic_tags:
            conditions.append(models.FieldCondition(
                key="sonic_tags", match=models.MatchValue(value=tag),
            ))

    return models.Filter(must=conditions) if conditions else None