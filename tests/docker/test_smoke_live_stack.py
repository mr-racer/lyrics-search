"""End-to-end smoke for the live stack — runs ONLY inside the container.

The rest of the suite mocks Qdrant and stubs the ML libs, so it can never catch
a break in the parts that only exist at runtime: real dense embeddings, a real
Qdrant accepting + ranking them, and CLAP actually loading. This smoke does.

Enabled by ``MUSIX_LIVE_STACK=1`` (see ``tests/docker/conftest.py`` and
``scripts/run_docker_tests.sh``); skipped otherwise. Heavy imports live inside
the test bodies so a normal collection stays cheap.

First run downloads the embedding model (~1 GB) and, for the CLAP test, the
~2.3 GB checkpoint into the ``weights``/HF-cache volumes; subsequent runs reuse
them.
"""
import uuid


def test_dense_embeddings_round_trip_through_qdrant(qdrant):
    """Real SentenceTransformer vectors upsert into Qdrant and rank by meaning."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.resources.model_registry import ModelRegistry

    model, _vector_name, dim = ModelRegistry.get_text_model(
        "intfloat/multilingual-e5-base"
    )

    docs = {
        1: "a sad slow piano ballad about heartbreak and rain",
        2: "an upbeat summer dance-pop banger with bright synths",
        3: "aggressive heavy-metal guitars and shouted vocals",
    }
    points = [
        PointStruct(id=i, vector=model.encode(text).tolist(), payload={"text": text})
        for i, text in docs.items()
    ]

    coll = f"smoke_{uuid.uuid4().hex[:8]}"
    qdrant.create_collection(
        coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    try:
        qdrant.upsert(coll, points=points)
        query = model.encode("a melancholic piano song about lost love").tolist()
        hits = qdrant.query_points(coll, query=query, limit=3).points
        assert hits, "Qdrant returned no hits"
        assert hits[0].id == 1, (
            "expected the sad-piano doc (id=1) nearest, got order "
            f"{[h.id for h in hits]}"
        )
    finally:
        qdrant.delete_collection(coll)


def test_clap_loads_and_encodes_text():
    """CLAP loads its checkpoint and produces a 512-d shared-space embedding."""
    import pytest

    from app.resources.model_registry import ModelRegistry

    if not ModelRegistry.is_clap_available():
        pytest.skip("laion_clap not installed in this image")

    clap = ModelRegistry.load_clap()
    emb = clap.get_text_embedding(
        ["a calm acoustic guitar melody", "a loud distorted techno track"]
    )
    assert emb.shape[0] == 2, f"expected 2 embeddings, got shape {emb.shape}"
    assert emb.shape[1] == 512, f"expected 512-d CLAP space, got {emb.shape[1]}"
