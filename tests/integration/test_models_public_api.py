"""The model legs over HTTP — the surface other RAG services consume.

The point of these tests is the CONTRACT, not the models: every encode is
faked. What has to hold is that the wire shape is the one outside stacks
already speak, that the asymmetric model is addressed correctly, and that a
failure arrives as a status code rather than as a 200 with something plausible
inside it.
"""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import models_public as mp
from app.resources.model_registry import ModelRegistry
from app.resources.models import ModelEncodeFailed, ModelOOM, ModelUnavailable

pytestmark = pytest.mark.integration

TOKEN = "test-models-token"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(mp.TOKEN_ENV, TOKEN)
    app = create_app()
    with TestClient(app) as c:
        c.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield c


def _fake_dense(monkeypatch, dim=None):
    """Records the side each call asked for and returns deterministic rows."""
    calls: list = []
    width = dim or ModelRegistry.VECTOR_DIM

    def encode_documents(texts, *, progress=None, is_query=False):
        calls.append({"n": len(texts), "is_query": is_query})
        return np.arange(len(texts) * width, dtype=np.float32).reshape(
            len(texts), width)

    monkeypatch.setattr(ModelRegistry, "encode_documents",
                        staticmethod(encode_documents))
    monkeypatch.setattr(mp, "_token_stats", lambda texts, limit: (7, 0))
    return calls


class TestAuth:
    def test_without_a_configured_token_everything_is_refused(self, monkeypatch):
        """An open model endpoint is not a sensible default even on a
        loopback-only deployment. Refusing loudly beats a door that turns out to
        have been open."""
        monkeypatch.delenv(mp.TOKEN_ENV, raising=False)
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/models/v1/models")
        assert resp.status_code == 503
        assert mp.TOKEN_ENV in resp.json()["detail"]

    def test_a_wrong_token_is_rejected(self, client):
        resp = client.get("/api/v1/models/v1/models",
                          headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401

    def test_a_missing_header_is_rejected(self, client):
        resp = client.get("/api/v1/models/v1/models", headers={"Authorization": ""})
        assert resp.status_code == 401

    def test_health_needs_no_token(self, monkeypatch):
        """It is what a healthcheck and a consumer's own breaker read; putting a
        token in front of that only means the token ends up in a compose file."""
        monkeypatch.delenv(mp.TOKEN_ENV, raising=False)
        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/models/health")
        assert resp.status_code == 200
        assert resp.json()["auth_configured"] is False


class TestTheTwoNamesForOneModel:
    """Octen ships two prompts and ``/v1/embeddings`` has no field for them.

    Every OpenAI client can set ``model``, none can set ``prompt_name`` — so the
    two sides are two model names over the same resident weights. Getting this
    wrong is not an error, it is a quiet recall loss, which is exactly why it is
    pinned here.
    """

    def test_the_catalogue_offers_both_sides_of_one_model(self, client):
        data = {m["id"]: m for m in client.get(
            "/api/v1/models/v1/models").json()["data"]}
        assert data["octen-query"]["side"] == "query"
        assert data["octen-document"]["side"] == "document"
        assert (data["octen-query"]["backing_model"]
                == data["octen-document"]["backing_model"]
                == ModelRegistry.TEXT_MODEL_NAME)
        assert data["octen-query"]["dimensions"] == ModelRegistry.VECTOR_DIM

    def test_the_query_name_encodes_the_query_side(self, client, monkeypatch):
        calls = _fake_dense(monkeypatch)
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-query", "input": "who produced this"})
        assert resp.status_code == 200
        assert calls[0]["is_query"] is True
        assert resp.json()["musix"]["side"] == "query"

    def test_the_document_name_encodes_the_document_side(self, client, monkeypatch):
        calls = _fake_dense(monkeypatch)
        client.post("/api/v1/models/v1/embeddings",
                    json={"model": "octen-document", "input": ["a", "b"]})
        assert calls[0]["is_query"] is False

    def test_an_unknown_model_is_a_404_that_says_what_is_served(self, client):
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "text-embedding-3-small", "input": "x"})
        assert resp.status_code == 404
        assert "octen-query" in resp.json()["detail"]


class TestEmbeddings:
    def test_the_response_is_openai_shaped(self, client, monkeypatch):
        _fake_dense(monkeypatch)
        body = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document",
                                 "input": ["a", "b"]}).json()
        assert body["object"] == "list"
        assert [d["index"] for d in body["data"]] == [0, 1]
        assert all(d["object"] == "embedding" for d in body["data"])
        assert len(body["data"][0]["embedding"]) == ModelRegistry.VECTOR_DIM
        assert body["usage"]["total_tokens"] == 7

    def test_base64_round_trips_to_the_same_floats(self, client, monkeypatch):
        """1024 floats as JSON text is roughly eight times the bytes of the same
        numbers packed, and a corpus indexer sends hundreds per request."""
        _fake_dense(monkeypatch)
        payload = {"model": "octen-document", "input": ["a"]}
        plain = client.post("/api/v1/models/v1/embeddings", json=payload).json()
        packed = client.post("/api/v1/models/v1/embeddings",
                             json={**payload, "encoding_format": "base64"}).json()

        decoded = np.frombuffer(
            base64.b64decode(packed["data"][0]["embedding"]), dtype=np.float32)
        assert np.allclose(decoded, plain["data"][0]["embedding"])

    def test_a_matryoshka_width_is_refused_rather_than_truncated(self, client,
                                                                 monkeypatch):
        """Octen's card documents a fixed 1024 and says nothing about MRL.
        Truncating it would degrade the vector invisibly."""
        _fake_dense(monkeypatch)
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": "x",
                                 "dimensions": 256})
        assert resp.status_code == 400
        assert "Matryoshka" in resp.json()["detail"]

    def test_the_models_own_width_is_accepted(self, client, monkeypatch):
        _fake_dense(monkeypatch)
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": "x",
                                 "dimensions": ModelRegistry.VECTOR_DIM})
        assert resp.status_code == 200

    def test_truncation_is_reported_rather_than_silent(self, client, monkeypatch):
        """The request succeeded, so this is not an error — but a caller that
        sent 3000-token documents to a 2048-token model has to find out."""
        _fake_dense(monkeypatch)
        monkeypatch.setattr(mp, "_token_stats", lambda texts, limit: (9000, 2))
        body = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document",
                                 "input": ["a", "b"]}).json()
        assert body["musix"]["truncated"] == 2
        assert body["musix"]["max_tokens"] == mp.MAX_SEQ_LENGTH

    def test_an_oversized_batch_is_refused_not_attempted(self, client, monkeypatch):
        """A bound on what one request may cost. Without it the failure lands on
        the card instead of in the caller's error handler."""
        calls = _fake_dense(monkeypatch)
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document",
                                 "input": ["x"] * (mp.MAX_INPUTS + 1)})
        assert resp.status_code == 413
        assert not calls, "the model must not be touched by a refused request"

    def test_an_empty_input_is_a_422(self, client, monkeypatch):
        _fake_dense(monkeypatch)
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": []})
        assert resp.status_code == 422


class TestSparse:
    """No standard to borrow, so the response is what the model produces —
    which is also Qdrant's sparse-vector format."""

    @staticmethod
    def _install(monkeypatch, rows):
        """A stand-in for the coalesced COO tensor MILCO returns.

        Built by hand rather than with ``torch.sparse_coo_tensor`` because the
        unit-test conftest stubs ``torch`` — and that is not a workaround, it is
        the point: ``_sparse_rows`` needs only ``coalesce``/``indices``/
        ``values``, so nothing in the wire format depends on a GPU stack being
        installed. The rows are emitted in (row, column) order, which is what
        ``coalesce`` guarantees and what the row-boundary search relies on.
        """
        calls: list = []

        class _Array:
            def __init__(self, data):
                self._data = np.asarray(data)
            def cpu(self):
                return self
            def numpy(self):
                return self._data

        class _FakeCOO:
            def __init__(self, rows):
                cols, vals, row_ids = [], [], []
                for r, row in enumerate(rows):
                    for col, val in sorted(row):
                        row_ids.append(r)
                        cols.append(col)
                        vals.append(val)
                self._idx = _Array([row_ids, cols])
                self._vals = _Array(vals)
            def coalesce(self):
                return self
            def indices(self):
                return self._idx
            def values(self):
                return self._vals

        def encode_sparse(texts, *, is_query=False):
            calls.append({"n": len(texts), "is_query": is_query})
            return _FakeCOO(rows)

        monkeypatch.setattr(ModelRegistry, "encode_sparse",
                            staticmethod(encode_sparse))
        return calls

    def test_rows_keep_their_own_non_zeros(self, client, monkeypatch):
        """The row boundaries are the whole contract: a consumer addresses these
        by input position, so mixing two rows would score passages against the
        wrong document — silently."""
        self._install(monkeypatch, [[(5, 0.5), (30600, 1.5)], [(7, 0.25)]])
        body = client.post("/api/v1/models/v1/embeddings/sparse",
                           json={"input": ["a", "b"]}).json()

        assert body["dim"] == mp.SPARSE_DIM == 280524
        assert body["data"][0]["indices"] == [5, 30600]
        assert body["data"][0]["values"] == [0.5, 1.5]
        assert body["data"][1]["indices"] == [7]
        assert [d["index"] for d in body["data"]] == [0, 1]

    def test_a_row_with_no_terms_stays_present_and_empty(self, client, monkeypatch):
        """Dropping it would shift every later row by one."""
        self._install(monkeypatch, [[(5, 0.5)], [], [(9, 0.1)]])
        body = client.post("/api/v1/models/v1/embeddings/sparse",
                           json={"input": ["a", "b", "c"]}).json()
        assert len(body["data"]) == 3
        assert body["data"][1]["indices"] == []

    def test_the_query_side_is_asked_for_explicitly(self, client, monkeypatch):
        calls = self._install(monkeypatch, [[(1, 1.0)]])
        client.post("/api/v1/models/v1/embeddings/sparse",
                    json={"input": "a", "is_query": True})
        assert calls[0]["is_query"] is True


class TestRerank:
    @staticmethod
    def _install(monkeypatch, probs):
        seen: list = []

        def ce_probabilities(query, docs):
            seen.append((query, list(docs)))
            return probs[:len(docs)]

        monkeypatch.setattr(ModelRegistry, "ce_probabilities",
                            staticmethod(ce_probabilities))
        return seen

    def test_results_are_best_first_and_carry_the_original_index(self, client,
                                                                 monkeypatch):
        self._install(monkeypatch, [0.1, 0.9, 0.5])
        body = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q", "documents": ["a", "b", "c"]}).json()
        assert [r["index"] for r in body["results"]] == [1, 2, 0]
        assert body["results"][0]["relevance_score"] == pytest.approx(0.9)

    def test_top_n_cuts_after_sorting(self, client, monkeypatch):
        self._install(monkeypatch, [0.1, 0.9, 0.5])
        body = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q", "documents": ["a", "b", "c"],
                                 "top_n": 2}).json()
        assert [r["index"] for r in body["results"]] == [1, 2]

    def test_documents_may_be_objects(self, client, monkeypatch):
        """Both forms are in circulation among the stacks that speak this shape,
        and guessing wrong costs a 422 the caller cannot act on."""
        seen = self._install(monkeypatch, [0.4, 0.6])
        body = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q",
                                 "documents": [{"text": "a"}, {"text": "b"}]}).json()
        assert seen[0][1] == ["a", "b"]
        assert [r["index"] for r in body["results"]] == [1, 0]

    def test_documents_can_be_echoed_back(self, client, monkeypatch):
        self._install(monkeypatch, [0.4, 0.6])
        body = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q", "documents": ["a", "b"],
                                 "return_documents": True}).json()
        assert body["results"][0]["document"]["text"] == "b"

    def test_an_object_without_text_is_a_422(self, client, monkeypatch):
        self._install(monkeypatch, [0.4])
        resp = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q", "documents": [{"body": "a"}]})
        assert resp.status_code == 422


class TestFailuresArriveAsStatusCodes:
    """The whole point of the change underneath this router: a broken leg must
    not be a 200 with something plausible in it."""

    def _fail(self, monkeypatch, exc):
        def boom(texts, *, progress=None, is_query=False):
            raise exc
        monkeypatch.setattr(ModelRegistry, "encode_documents", staticmethod(boom))
        monkeypatch.setattr(mp, "_token_stats", lambda texts, limit: (1, 0))

    def test_a_missing_leg_is_a_503_a_client_can_retry(self, client, monkeypatch):
        self._fail(monkeypatch,
                   ModelUnavailable("dense", "load", "weights are missing"))
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": "x"})
        assert resp.status_code == 503
        assert resp.headers["Retry-After"]
        error = resp.json()["error"]
        assert error["type"] == "model_unavailable"
        assert error["leg"] == "dense"
        assert "weights are missing" in error["message"]

    def test_an_out_of_memory_names_itself(self, client, monkeypatch):
        """Distinct from a generic failure because it says something about the
        machine rather than the request, and can clear on its own."""
        self._fail(monkeypatch, ModelOOM("dense", "encode", "no room"))
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": "x"})
        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "model_out_of_memory"

    def test_a_broken_encode_is_a_500_not_an_empty_result(self, client, monkeypatch):
        self._fail(monkeypatch, ModelEncodeFailed("dense", "encode", "boom"))
        resp = client.post("/api/v1/models/v1/embeddings",
                           json={"model": "octen-document", "input": "x"})
        assert resp.status_code == 500
        assert "data" not in resp.json()

    def test_the_sparse_leg_maps_the_same_way(self, client, monkeypatch):
        def boom(texts, *, is_query=False):
            raise ModelUnavailable("sparse", "load", "milco is not here")
        monkeypatch.setattr(ModelRegistry, "encode_sparse", staticmethod(boom))
        resp = client.post("/api/v1/models/v1/embeddings/sparse",
                           json={"input": "x"})
        assert resp.status_code == 503
        assert resp.json()["error"]["leg"] == "sparse"

    def test_the_reranker_maps_the_same_way(self, client, monkeypatch):
        def boom(query, docs):
            raise ModelUnavailable("cross_encoder", "load", "no reranker")
        monkeypatch.setattr(ModelRegistry, "ce_probabilities", staticmethod(boom))
        resp = client.post("/api/v1/models/v1/rerank",
                           json={"query": "q", "documents": ["a"]})
        assert resp.status_code == 503
        assert resp.json()["error"]["leg"] == "cross_encoder"
