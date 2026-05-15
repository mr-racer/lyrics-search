"""AI Indexing task implementations — importing this package registers them."""

from app.services.ai_tasks import sonic_vibe  # noqa: F401 — side effect: register
from app.services.ai_tasks import refined_facts  # noqa: F401 — side effect: register
