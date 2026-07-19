"""Lyric gems — rare, quotable facts mined from a track's lyrics.

Four kinds:
- ``capsule``   — a bygone-era item used literally (pager, payphone, MySpace)
- ``namedrop``  — the lyrics mention another artist from THIS library
- ``popculture``— a fictional character or movie reference (Tony Montana)
- ``songref``   — the lyrics name a song/album that exists in THIS library

Pipeline: preprocess (strip section headers) → GLiNER2 entity pass (shared
model from ``fact_relations.extractor``) → post-filter → gazetteer/catalog
canonicalization → optional LLM verification for leftovers (≤2 web searches,
verdicts cached forever). Silence over noise: only confident hits persist.
"""
