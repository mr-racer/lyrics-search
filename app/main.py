"""Entry point shim.

The real application lives in app.api.main.
Run with:
    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from app.api.main import app  # noqa: F401 — re-export for uvicorn app.main:app

__all__ = ["app"]
