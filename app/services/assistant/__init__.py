"""Unified AI assistant — one entry point over the three existing AI stacks.

``router`` picks the intent — the LLM reads the sentence (``intent_llm``),
GLiNER2 pulls the spans and stands in as classifier whenever the LLM cannot —
``service`` orchestrates the dispatch and normalises both executors' progress
callbacks into one NDJSON envelope, ``humanize`` turns every stage into a
ready-to-render label, and ``facts_executor`` implements the one genuinely new
branch (including "explain THIS fact", which is grounded or silent).

Nothing here owns business logic that already exists: search delegates to
``chat_search_service``, playlists to ``recsys_ai_service.ai_playlist``.
"""
