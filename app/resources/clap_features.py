"""CLAP audio feature extraction.

Extracted from legacy search_engine/utils.py during Refactor 2.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import torch

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_DURATION = 420  # seconds — kept here so clap_features is self-contained


@dataclass
class TrackFeatures:
    title: str
    artist: str
    vector_clap: list


def unit_norm(v):
    """L2-normalise a numpy vector. Returns the input unchanged if its norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def get_clap_embedding_long(clap_model, y: np.ndarray, sr: int,
                             chunk_sec: int = 10, device=DEVICE
                             ) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    """Split long audio into chunks, embed each, return (mean, per-chunk list).

    Both outputs come from the same forward pass. The mean is NOT unit-norm yet
    (caller decides); per-chunk vectors are returned as-is from the model.
    Returns ``(None, None)`` when the audio is shorter than 5 seconds.
    """
    chunk_len = sr * chunk_sec

    chunks = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start: start + chunk_len]
        if len(chunk) < sr * 5:
            continue
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)

    if not chunks:
        logger.debug("[CLAP] audio too short — skipping")
        return None, None

    batch = torch.from_numpy(np.stack(chunks)).to(device)
    with torch.no_grad():
        embeddings = clap_model.get_audio_embedding_from_data(x=batch, use_tensor=True)

    emb_np = embeddings.cpu().numpy()                  # (N_chunks, 512)
    mean_vec = emb_np.mean(axis=0)                     # (512,)
    chunk_list = [emb_np[i] for i in range(emb_np.shape[0])]
    return mean_vec, chunk_list


def extract_clap_features(path: str, model, duration: int = 300, device=DEVICE
                          ) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    """Load audio file via librosa and return (unit-norm mean vector, unit-norm per-chunk list).

    Per-chunk vectors are L2-normalised individually so callers can dot-product
    them directly for cosine similarity without further preprocessing.
    """
    if librosa is None:
        raise RuntimeError("librosa not installed — required for CLAP feature extraction")
    y, sr = librosa.load(path, duration=duration, sr=48000, mono=True)
    mean_vec, chunk_list = get_clap_embedding_long(model, y, sr, chunk_sec=10, device=device)
    del y
    if mean_vec is None:
        return None, None
    return unit_norm(mean_vec), [unit_norm(c) for c in (chunk_list or [])]


def _encode_clap(
    tracks: list[dict],
    model_clap=None,
    progress_callback=None,
) -> tuple[dict[tuple, np.ndarray], dict[tuple, list[np.ndarray]]]:
    """Encode each track's audio with CLAP.

    Returns ``(means_map, chunks_map)``, both keyed by ``(artist_lower, title_lower)``.
    ``means_map[key]`` is the unit-norm pooled vector used for the named CLAP vector
    in Qdrant (search path). ``chunks_map[key]`` is the list of unit-norm per-chunk
    vectors used to populate the ``clap_chunks`` payload field (analysis path).

    Args:
        tracks: list of track dicts that include ``file_path``, ``artist``, ``title``.
        model_clap: pre-loaded CLAP module. If None, ModelRegistry.load_clap() is called.
        progress_callback: optional callable(current, total).
    """
    # Build lookup file_path → (artist_lower, title_lower)
    path_to_key = {}
    for t in tracks:
        fp = t.get("file_path")
        artist = (t.get("artist") or "").strip().lower()
        title = (t.get("title") or "").strip().lower()
        if fp and artist and title:
            path_to_key[fp] = (artist, title)

    if not path_to_key:
        logger.warning("[CLAP] No tracks with valid file_path + artist + title — skipping")
        return {}, {}

    if not model_clap:
        from app.resources.model_registry import ModelRegistry
        model_clap = ModelRegistry.load_clap()

    clap_map: dict[tuple, np.ndarray] = {}
    chunks_map: dict[tuple, list[np.ndarray]] = {}
    total = len(path_to_key)
    for idx, (fp, key) in enumerate(path_to_key.items(), 1):
        try:
            mean_vec, chunk_list = extract_clap_features(fp, model_clap, 300)
            if mean_vec is not None:
                clap_map[key] = mean_vec
                if chunk_list:
                    chunks_map[key] = chunk_list
        except Exception as e:
            logger.warning("[CLAP] Failed to encode %s (%s — %s): %s", fp, *key, e)

        if progress_callback:
            progress_callback(idx, total)

        if idx % 50 == 0 or idx == total:
            logger.info("[CLAP] Encoded %d / %d", idx, total)

    # Don't delete the model — it's owned by ModelRegistry now (cached singleton).
    # Only release the local reference + free GPU memory tied to this batch.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "[CLAP] Mapped %d / %d tracks (per-chunk lists captured for %d)",
        len(clap_map), total, len(chunks_map),
    )
    return clap_map, chunks_map
