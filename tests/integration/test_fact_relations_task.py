"""The fact_relations backfill task must register itself via the side-effect
import of ``app.services.ai_tasks`` — same registry contract as the other AI
tasks (removing that import silently breaks POST /library/ai-index/{type})."""
import pytest


@pytest.mark.integration
def test_fact_relations_task_registered():
    from app.services import ai_tasks  # noqa: F401 — side-effect import populates registry
    from app.services import ai_indexing_service

    assert "fact_relations" in ai_indexing_service._registry
    assert callable(ai_indexing_service._registry["fact_relations"])
