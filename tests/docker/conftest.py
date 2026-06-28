"""Fixtures + guard for the live-stack suite (``tests/docker/``).

These tests are the only ones that exercise the REAL stack — a reachable Qdrant
and real CLAP / sentence-transformers models (the root ``tests/conftest.py``
stubs those for every other suite). They are meant to run INSIDE the running
container, where ``QDRANT_URL`` points at the ``qdrant`` service and the model
weights are cached on a volume.

Run them with ``scripts/run_docker_tests.sh`` (which exports MUSIX_LIVE_STACK=1).
Without that flag every test here is skipped, so a plain ``pytest`` on a dev box
never tries to reach Qdrant or download multi-GB model weights.
"""
import os

import pytest

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


def pytest_collection_modifyitems(config, items):
    """Skip the whole live-stack suite unless MUSIX_LIVE_STACK is set."""
    if os.environ.get("MUSIX_LIVE_STACK"):
        return
    skip = pytest.mark.skip(
        reason="live-stack suite — run via scripts/run_docker_tests.sh "
        "(sets MUSIX_LIVE_STACK=1) inside the container"
    )
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="module")
def qdrant():
    """A real QdrantClient against the compose-network ``qdrant`` service.

    Calls ``get_collections()`` eagerly so an unreachable Qdrant fails here with
    a clear error instead of deep inside a test.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    client.get_collections()
    yield client
    client.close()
