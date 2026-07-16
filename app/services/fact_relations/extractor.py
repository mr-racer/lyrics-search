"""Lazy, thread-safe GLiNER2 singleton for producer/sample relation extraction.

``gliner2``/``torch`` are only imported inside :func:`get_extractor`, never at
module scope -- importing this module (or the ``fact_relations`` package) must
stay cheap, since it's imported eagerly by ``song_facts_service`` on every
process start. The model itself loads lazily, on first real use, exactly like
``app/resources/model_registry.py`` does for the text/CLAP models.
"""
import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_extractor = None


class GlinerRelationExtractor:
    """Thin wrapper pairing a loaded GLiNER2 model with its fixed schema."""

    def __init__(self, model, schema):
        self._model = model
        self._schema = schema

    def extract(self, fact: str) -> dict:
        """Run one GLiNER2 extraction call over ``fact``.

        Returns the raw GLiNER2 output dict (entities/relation_extraction/
        sample_source/sample_usage) -- the shape ``triage.py`` (Task 2)
        expects as ``gliner_out``.
        """
        return self._model.extract(fact, self._schema, include_confidence=True)


def _build_schema(model):
    schema = model.create_schema()
    schema.entities({
        "producer": "person credited as the music producer of this song or album",
        "sampled_song": "title of an older song that is sampled or interpolated in this song",
    })
    schema.relations({
        "produced_by": "the song or album (head) was produced by a person (tail)",
        "samples": "the song (head) contains a sample or interpolation of another, older song or artist (tail)",
        "sampled_by": "the song (head) was later sampled or reused by another artist or song (tail)",
    })
    source = schema.structure("sample_source")
    source.field("song", dtype="str",
                 description="title of the OLDER song that is sampled or interpolated inside this song")
    source.field("artist", dtype="str",
                 description="artist of the older song that is sampled inside this song")
    usage = schema.structure("sample_usage")
    usage.field("song", dtype="str",
                description="title of the NEWER song in which this song was sampled or reused")
    usage.field("artist", dtype="str",
                description="artist who sampled or reused this song in their own newer track")
    return schema


def get_extractor() -> GlinerRelationExtractor:
    """Return the process-wide lazy GLiNER2 extractor singleton.

    Double-checked locking: the common case (already loaded) never takes the
    lock. We deliberately do NOT call ``torch.set_num_threads`` here -- that
    knob is process-global and would permanently throttle the primary CLAP /
    dense-text models (which load concurrently during indexing) to a fraction
    of the cores. Per-fact extraction is short (~350 ms), so it can share the
    default thread pool without starving anything.
    """
    global _extractor
    if _extractor is not None:
        return _extractor
    with _lock:
        if _extractor is None:
            from gliner2 import GLiNER2

            logger.info("[fact_relations] Loading GLiNER2 model (fastino/gliner2-base-v1)...")
            model = GLiNER2.from_pretrained("fastino/gliner2-base-v1").cpu().eval()
            schema = _build_schema(model)
            _extractor = GlinerRelationExtractor(model, schema)
            logger.info("[fact_relations] GLiNER2 model loaded.")
    return _extractor
