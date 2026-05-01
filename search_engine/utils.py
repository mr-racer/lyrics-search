import numpy as np

import torch
import gc
from sentence_transformers import SentenceTransformer

import librosa
import laion_clap

from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from file_processor.utils import get_metadata

from qdrant_client import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLAP_WEIGHTS = 'weights/music_audioset_epoch_15_esc_90.14.pt'

# MODEL FUNCTIONS

def load_model(model: str, device=DEVICE):
    model_loaded = SentenceTransformer(model, device=device)
    vector_name = model.split('/')[-1].lower()
    vector_dim = model_loaded.get_embedding_dimension()
    return model_loaded, vector_name, vector_dim

def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    # Собираем результат
    result = [
        {**{k: v for k, v in d.items() if k != 'duration'}, 'duration': bucket}
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

def load_model_clap(device=DEVICE, clap_weights_path=CLAP_WEIGHTS):
    model_clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model_clap.load_ckpt(clap_weights_path)
    model_clap.eval()
    model_clap = model_clap.to(device)
    gc.collect()

    torch.manual_seed(0)
    np.random.seed(0)
    return model_clap

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


# def process_batch(paths: list[str], device=DEVICE) -> list[TrackFeatures]:
#     with ProcessPoolExecutor(max_workers=3) as ex:
#         metadatas = list(tqdm(ex.map(get_metadata, paths), total=len(paths), desc="Metadata"))

#     # CLAP последовательно (GPU — один поток)
#     model = load_model_clap(device=device)
#     claps = [
#         extract_clap_features(p, model, 300, device=device)
#         for p in tqdm(paths, desc="CLAP embeddings")
#     ]
    
#     return [
#             TrackFeatures(
#                 title=meta.get('title'),
#                 artist=meta.get('artist'),
#                 vector_clap=c,
#             )
#             for c, meta in zip(claps, metadatas)
#             ]


def _encode_clap(paths: list[str], model_clap = None) -> dict[tuple, np.ndarray]:
    with ProcessPoolExecutor(max_workers=3) as ex:
        metadatas = list(tqdm(ex.map(get_metadata, paths), total=len(paths), desc="CLAP metadata"))

    if not model_clap:
        model_clap = load_model_clap()
    clap_vecs = [
        extract_clap_features(p, model_clap, 300)
        for p in tqdm(paths, desc="CLAP embeddings")
    ]
    del model_clap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        (m.get("artist", "").lower(), m.get("title", "").lower()): v
        for m, v in zip(metadatas, clap_vecs)
        if m.get("artist") and m.get("title") and v is not None
    }
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
    year_range: str | None = None
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

    if year_range:
        conditions.append(models.FieldCondition(key="year_range", match=models.MatchValue(value=year_range)))

    return models.Filter(must=conditions) if conditions else None