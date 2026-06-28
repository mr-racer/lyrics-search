# MusiX application image.
#
# ── Stage 1: build the Vite frontend ────────────────────────────────────────
# The frontend is a Vite project (frontend/). We compile it to static assets
# here and copy only the built dist/ into the runtime image — Node and
# node_modules never reach the final image.
FROM node:24-alpine AS frontend-build
WORKDIR /build
# Lockfile first for layer caching: this layer only rebuilds when deps change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# → /build/dist

# ── Stage 2: the Python application image ───────────────────────────────────
# Base: PyTorch CUDA runtime already ships torch 2.6.0+cu124 (+torchvision /
# torchaudio), so `pip install -r requirements.txt` reuses those wheels instead
# of re-downloading ~2.5 GB from the PyTorch index.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System libs that were provided by the OS on Windows but must be installed
# explicitly on Debian:
#   ffmpeg      — ALAC -> FLAC transcode on stream (app/services/audio_streaming.py)
#   libmagic1   — MIME sniff for uploads (python-magic)
#   libsndfile1 — librosa / soundfile audio feature extraction
#   git         — a few pip packages build from git
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libmagic1 \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace cache -> mounted as a named volume so sentence-transformers
    # + CLAP checkpoints (hundreds of MB) survive container rebuilds.
    HF_HOME=/app/.hf-cache

# Dependencies first for layer caching: this layer only rebuilds when
# requirements.txt changes, not on every code edit.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Data dirs (media/, cache/, weights/, frontend/covers/) are
# volumes — intentionally NOT copied (also excluded via .dockerignore).
COPY app/ ./app/
# Frontend: only the compiled assets from the build stage. The source tree
# (src/, node_modules/) never reaches the runtime image. covers/ is a volume.
COPY --from=frontend-build /build/dist ./frontend/dist
COPY scripts/ ./scripts/
COPY logging.conf __init__.py ./

EXPOSE 8000

# Single worker on purpose: CLAP/embedding models live in process memory and
# the JobTracker that drives SSE indexing progress is in-process. Multiple
# workers would duplicate the models and split job state across processes.
# One async worker serves many concurrent clients fine.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]
