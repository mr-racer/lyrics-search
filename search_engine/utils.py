"""Backward-compatibility shim.

The actual implementations were moved to ``app/resources/`` in Refactor 2:
    qdrant_filters.py — build_filter
    qdrant_payload.py — build_text_for_embedding, prepare_metadata
    clap_features.py  — extract_clap_features, _encode_clap, ...

This shim will be deleted in Refactor 6 once all call sites migrate.
"""
from app.resources.qdrant_filters import build_filter  # noqa: F401
from app.resources.qdrant_payload import (              # noqa: F401
    MAX_DURATION,
    build_text_for_embedding,
    prepare_metadata,
)
from app.resources.clap_features import (               # noqa: F401
    DEVICE,
    TrackFeatures,
    _encode_clap,
    extract_clap_features,
    get_clap_embedding_long,
    unit_norm,
)
