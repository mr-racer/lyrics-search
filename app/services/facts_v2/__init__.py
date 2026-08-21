"""The labelled fact pipeline: classify, then rewrite one fact at a time.

Split out of ``ai_tasks/refined_facts`` because the task module is a job
runner — iterate the collection, count progress, persist — while this is the
processing itself, and the two are tested and reasoned about separately.
"""

from app.services.facts_v2.pipeline import (      # noqa: F401
    ARTIST_LABELS, SONG_LABELS, classify_entity, process_entity, route,
)
