"""One module per intent, plus the machinery the two web branches share.

Each branch owns its own control flow and nothing else: the orchestrator builds
the parts (catalog, facts, search, fetch, LLM) and hands them over, so a branch
can be read start to finish without knowing how the others work.
"""
