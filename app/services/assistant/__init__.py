"""The AI assistant — a deterministic agent over the library and the open web.

``planner`` reads the request once and hands over a plan every field of which
code has checked; ``web_sources`` + ``fetcher`` gather pages (hosts pinned for
Wikipedia / Apple / Fandom, the open web reranked by cross-encoder before
anything is downloaded); ``chunking`` and ``services/retrieval`` turn pages into
the ten passages that matter; ``tracklists`` pulls track titles — structure
first, the model only where there is no structure; ``library_catalog`` decides
what exists, so a title the model invented matches nothing and disappears;
``agent`` runs the branches and owns every stop/continue decision.

Four branches, one input field: ``branches/lyrics`` (find the song from its
words), ``branches/audio`` (find it by how it sounds), ``branches/playlist``
(songs to play, sourced from the web) and ``branches/general`` (a grounded prose
answer, including "explain THIS statement").

``service`` turns whichever result comes back into the route's payload and
``humanize`` gives every progress stage a ready-to-render caption.

Legacy, still in the tree and no longer called: ``router``, ``intent_llm`` and
``facts_executor`` — the GLiNER2 router and the facts executor this replaced.
"""
