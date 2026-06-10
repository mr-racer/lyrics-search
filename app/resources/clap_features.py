"""CLAP audio feature extraction + sonic axes (Stream RecSys).

Extracted from legacy search_engine/utils.py during Refactor 2.
Sonic axes ported from notebooks/similar_music.ipynb (cell 67) for the
stream recsys design (docs/2026-06-09-stream-recsys-design.md §8).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

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


# --------------------------------------------------------------------------
# Sonic axes — CLAP text prompts → interpretable per-track axes.
#
# Antonym pairs become differential axes (energetic − calm, …) because the
# difference of two cosines cancels the shared "music-ness" component and
# isolates the contrast; single prompts stay absolute.
# Prompt wording is tuned against listening tests in similar_music.ipynb —
# do not edit casually: any change bumps AXIS_VERSION and invalidates the
# reference norm file.
# --------------------------------------------------------------------------

AXIS_PROMPTS: dict[str, str] = {
    'calm': 'a calm, relaxing, peaceful and quiet song with soft, gentle, airy sound',
    'energetic': 'an energetic, powerful track — either intense and driving or heavy '
                 'and bass-driven with deep low-end and hard-hitting groovy drums',
    'vocal': 'a vocal-led song where singing or rap is the main focus, with strong '
             'prominent lead vocals carrying the track',
    'instrumental': 'an instrument-led track where music and instruments are the main '
                    'focus, with little or no singing, vocals in the background or absent',
    'spacious': 'a song with a spacious, wide, open sound — lush reverb, echo and delay '
                'effects, a deep sense of space and a broad immersive stereo image',
    'experimental': 'experimental track with erratic shifting rhythm, abrupt beat changes, '
                    'distorted gritty texture, chaotic and unpredictable',
    'stable': 'conventional track with steady predictable beat, smooth transitions, '
              'clean polished texture, orderly and consistent',
    'bright': 'a bright track with sparkling, crisp, airy high frequencies and lots of treble',
    'dark': 'a dark track with dull, muffled, subdued tone and rolled-off, recessed '
            'high frequencies',
    'acoustic': 'a track played on real live acoustic instruments: piano, acoustic guitar, '
                'strings, drums, brass and woodwinds',
    'synthetic': 'an electronic track built from synthesizers, drum machines, samplers '
                 'and digital programmed sounds',
}

# Stubs — prompts that do NOT form an axis yet. 'dense' is unoptimized; future
# iterations may add text-derived axes here (see design doc §11).
AXIS_PROMPT_STUBS: dict[str, str] = {
    'dense': 'a dense, rich, layered, busy and saturated production',
}

AXIS_NAMES: tuple[str, ...] = (
    'energy', 'vocal_lead', 'spacious', 'experimental', 'brightness', 'acousticness',
)

_PROMPT_IDX = {name: i for i, name in enumerate(AXIS_PROMPTS)}


def axis_version() -> str:
    """Hash of (active prompts + checkpoint name) — versions the axis space.

    Stored alongside norm stats so a prompt/checkpoint change invalidates
    stale reference files instead of silently mixing incompatible scales.
    """
    from app.resources.model_registry import CLAP_WEIGHTS_PATH
    blob = json.dumps(AXIS_PROMPTS, sort_keys=True) + CLAP_WEIGHTS_PATH.name
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def compute_axis_text_embeddings(clap_model) -> np.ndarray:
    """Encode AXIS_PROMPTS with the CLAP text tower → (n_prompts, 512), L2-normalised rows."""
    emb = clap_model.get_text_embedding(list(AXIS_PROMPTS.values()), use_tensor=False)
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def make_axes(scores: np.ndarray) -> np.ndarray:
    """Raw prompt cosines (N, len(AXIS_PROMPTS)) → axis values (N, len(AXIS_NAMES)).

    Column order follows AXIS_NAMES.
    """
    scores = np.atleast_2d(np.asarray(scores))
    if scores.shape[1] != len(AXIS_PROMPTS):
        raise ValueError(
            f"expected {len(AXIS_PROMPTS)} prompt scores per row, got {scores.shape[1]}"
        )
    i = _PROMPT_IDX
    cols = [
        scores[:, i['energetic']] - scores[:, i['calm']],        # energy
        scores[:, i['vocal']] - scores[:, i['instrumental']],    # vocal_lead
        scores[:, i['spacious']],                                # spacious
        scores[:, i['experimental']] - scores[:, i['stable']],   # experimental
        scores[:, i['bright']] - scores[:, i['dark']],           # brightness
        scores[:, i['acoustic']] - scores[:, i['synthetic']],    # acousticness
    ]
    return np.stack(cols, axis=1)


def axes_for_clap_vectors(clap_vectors: np.ndarray, text_emb: np.ndarray) -> list[dict[str, float]]:
    """Project unit-norm CLAP audio vectors onto the axis space.

    Returns one ``{axis_name: raw_score}`` dict per input vector — the exact
    shape stored in the Qdrant ``sonic_axes`` payload field (raw scores, NOT
    z-scores: normalisation happens at read time from collection stats).
    """
    clap_vectors = np.atleast_2d(np.asarray(clap_vectors, dtype=np.float32))
    scores = clap_vectors @ text_emb.T                # (N, n_prompts)
    axes = make_axes(scores)                          # (N, n_axes)
    return [
        {name: float(row[j]) for j, name in enumerate(AXIS_NAMES)}
        for row in axes
    ]


# --------------------------------------------------------------------------
# Axis normalisation: reference stats + shrinkage blending (design §8).
#
# A 5-track collection produces garbage mean/std → dead axes. Shrinkage pulls
# small collections toward a reference built from a large diverse library
# (scripts/build_axis_norm_reference.py); big collections trust themselves.
# --------------------------------------------------------------------------

AXIS_NORM_REFERENCE_PATH = Path(__file__).parent / "data" / "axis_norm_reference.json"
AXIS_SHRINKAGE_PSEUDO_N = 100   # λ = n / (n + PSEUDO_N): n=100 → 50/50 blend


def _usable_axis_stats(stats: dict | None) -> bool:
    """A stats dict counts only with mean+std AND a version matching the
    current axis space — stats from old prompts must not leak into z-scores."""
    return bool(
        stats
        and stats.get("mean")
        and stats.get("std")
        and stats.get("version") == axis_version()
    )


def load_axis_norm_reference(path: Path | None = None) -> dict | None:
    """Load the reference stats file; None when missing, corrupt, or stale."""
    p = path or AXIS_NORM_REFERENCE_PATH
    if not p.exists():
        return None
    try:
        ref = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[axes] unreadable axis_norm_reference at %s — ignoring", p)
        return None
    if not _usable_axis_stats(ref):
        logger.warning(
            "[axes] axis_norm_reference at %s is stale or malformed (version %r != %r) — "
            "rebuild via scripts/build_axis_norm_reference.py",
            p, ref.get("version"), axis_version(),
        )
        return None
    return ref


def blend_axis_stats(collection_stats: dict | None, reference: dict | None) -> dict | None:
    """Shrinkage blend: ``λ = n/(n+100)``, ``stats = λ·collection + (1−λ)·reference``.

    Either source may be missing or version-stale — it is then dropped. Returns
    None when nothing usable remains (callers must disable axis z-scoring).
    """
    coll_ok = _usable_axis_stats(collection_stats)
    ref_ok = _usable_axis_stats(reference)
    if not coll_ok and not ref_ok:
        return None
    if not coll_ok:
        return {"mean": dict(reference["mean"]), "std": dict(reference["std"]),
                "n": 0, "source": "reference"}
    if not ref_ok:
        return {"mean": dict(collection_stats["mean"]), "std": dict(collection_stats["std"]),
                "n": collection_stats.get("n", 0), "source": "collection"}

    n = collection_stats.get("n", 0)
    lam = n / (n + AXIS_SHRINKAGE_PSEUDO_N)
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for a in AXIS_NAMES:
        mean[a] = lam * collection_stats["mean"].get(a, 0.0) + (1 - lam) * reference["mean"].get(a, 0.0)
        std[a] = lam * collection_stats["std"].get(a, 0.0) + (1 - lam) * reference["std"].get(a, 0.0)
    return {"mean": mean, "std": std, "n": n, "source": "blend", "lambda": lam}


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
