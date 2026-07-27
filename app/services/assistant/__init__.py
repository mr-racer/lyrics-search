"""Unified AI assistant — one entry point over the three existing AI stacks.

``router`` picks the intent with GLiNER2 (never the LLM), ``service``
orchestrates the dispatch and normalises both executors' progress callbacks
into one NDJSON envelope, ``humanize`` turns every stage into a ready-to-render
label, and ``facts_executor`` implements the one genuinely new branch.

Nothing here owns business logic that already exists: search delegates to
``chat_search_service``, playlists to ``recsys_ai_service.ai_playlist``.
"""
