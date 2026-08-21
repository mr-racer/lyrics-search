"""Wikipedia-first artist biographies.

Split from ``ai_tasks/artist_bio`` for the same reason the fact pipeline was:
that module is a job runner, this is the work.
"""

from app.services.bio_v2.pipeline import build          # noqa: F401

__all__ = ["build"]
