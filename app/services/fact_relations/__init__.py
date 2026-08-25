"""Producer-credits pipeline (GLiNER2 triage + LLM confirmation).

``process_song_facts`` (the pipeline entry point) glues the pure-Python
triage layer (``triage.py``), the LLM confirmation layer (``llm_re.py``), and
the lazy GLiNER2 extractor (``extractor.py``) together, and writes the result
via ``MetadataDB.set_song_producers``.

The sampling half of this package (``gates.py``, the sample side of
``triage.py``) is no longer wired into production — facts_v2 owns sampling
links. It is still reachable through ``collect_claims(samples=True)`` for
``scripts/dry_run_relations.py``.
"""
from .service import process_song_facts, process_song_facts_async

__all__ = ["process_song_facts", "process_song_facts_async"]
