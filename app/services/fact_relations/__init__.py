"""Producer/samples fact-relations pipeline (GLiNER2 triage + LLM confirmation).

``process_song_facts`` (the pipeline entry point) glues the pure-Python
triage layer (``triage.py``), the LLM confirmation layer (``llm_re.py``), and
the lazy GLiNER2 extractor (``extractor.py``) together, and writes the result
via ``MetadataDB.set_song_relations``.
"""
from .service import process_song_facts, process_song_facts_async

__all__ = ["process_song_facts", "process_song_facts_async"]
