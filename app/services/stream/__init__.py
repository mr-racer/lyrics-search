"""Session RecSys internals («Поток»).

Split out of the 2k-line ``stream_service`` by the 2026-08-03 session-recsys
redesign. ``stream_service`` stays the orchestrator (``next_chunk``) and the
home of everything this redesign does NOT touch (long-term profile, vibes,
similar tracks, axis playlist); the four modules here own the new machinery:

  calibration — CLAP cosine → per-collection percentile
  baseline    — the listener's own skip/completion/reaction baseline → weights
  session     — reaction cutoff, skip forgiveness, ±clusters, carryover
  pools       — fresh/familiar candidate pools, slider quotas, chunk assembly

Design: docs/superpowers/specs/2026-08-03-session-recsys-design.md
"""
