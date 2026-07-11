"""Unit tests for sonic features — consolidated.

Merged from four former modules (one class per source):
  - test_sonic_axes.py            -> TestMakeAxes / TestAxesForClapVectors / TestAxisVersion
  - test_sonic_descriptor_tags.py -> TestSonicDescriptorTags
  - test_sonic_facets_aggregate.py-> TestSonicFacetsAggregate
  - test_axis_norm_blend.py       -> TestBlendAxisStats / TestLoadAxisNormReference

Covers: the prompt->axis differential math, payload-dict projection, input
validation and axis-space versioning (no CLAP model required — text embeddings
are faked with small matrices); the SonicDescriptorService scaffold + prompt
vocabulary loading + tag computation; MetadataDB.get_sonic_facets aggregation;
and shrinkage blending + reference loading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub heavy optional deps that are not installed in the test env (Python 3.13).
# Must happen before importing SonicDescriptorService (pulls in laion_clap).
sys.modules.setdefault("laion_clap", MagicMock())

import numpy as np

from app.resources.clap_features import (
    AXIS_NAMES,
    AXIS_PROMPT_STUBS,
    AXIS_PROMPTS,
    axes_for_clap_vectors,
    axis_version,
    blend_axis_stats,
    load_axis_norm_reference,
    make_axes,
)
from app.resources.metadata_db import MetadataDB
from app.services.sonic_descriptor_service import SonicDescriptorService

N_PROMPTS = len(AXIS_PROMPTS)


# --------------------------------------------------------------------------- #
# Module-level helpers & fixtures (hoisted from the merged sources)
# --------------------------------------------------------------------------- #
def _scores_row(**overrides) -> np.ndarray:
    """One row of prompt scores, zeros except named prompts."""
    row = np.zeros(N_PROMPTS, dtype=np.float32)
    names = list(AXIS_PROMPTS)
    for name, value in overrides.items():
        row[names.index(name)] = value
    return row


def _stats(mean_val: float, std_val: float, n: int, version: str | None = None) -> dict:
    return {
        "version": version if version is not None else axis_version(),
        "n": n,
        "mean": {a: mean_val for a in AXIS_NAMES},
        "std": {a: std_val for a in AXIS_NAMES},
    }


@pytest.fixture
def sample_vocab_file(tmp_path):
    vocab = {
        "version": 1,
        "groups": {
            "energy": ["punchy", "ambient"],
            "valence": ["sad", "happy"],
        },
    }
    p = tmp_path / "sonic_prompts.json"
    p.write_text(json.dumps(vocab))
    return p


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr("app.resources.metadata_db.DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr("app.resources.metadata_db.DB_DIR", tmp_path)
    MetadataDB._reset_for_tests()
    MetadataDB.init()
    yield MetadataDB
    MetadataDB._reset_for_tests()


# --------------------------------------------------------------------------- #
# From test_sonic_axes.py
# --------------------------------------------------------------------------- #
class TestMakeAxes:
    def test_differential_axes_subtract_antonym(self):
        row = _scores_row(energetic=0.5, calm=0.2, bright=0.1, dark=0.4)
        axes = make_axes(row)

        by_name = dict(zip(AXIS_NAMES, axes[0]))
        assert by_name["energy"] == pytest.approx(0.3, abs=1e-6)
        assert by_name["brightness"] == pytest.approx(-0.3, abs=1e-6)

    def test_single_prompt_axis_passes_through(self):
        row = _scores_row(spacious=0.42)
        axes = make_axes(row)
        assert dict(zip(AXIS_NAMES, axes[0]))["spacious"] == pytest.approx(0.42, abs=1e-6)

    def test_all_six_axes_in_declared_order(self):
        axes = make_axes(np.zeros((3, N_PROMPTS)))
        assert axes.shape == (3, len(AXIS_NAMES))
        assert AXIS_NAMES == (
            "energy", "vocal_lead", "spacious", "experimental", "brightness", "acousticness",
        )

    def test_1d_input_treated_as_single_row(self):
        axes = make_axes(np.zeros(N_PROMPTS))
        assert axes.shape == (1, len(AXIS_NAMES))

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError, match="prompt scores"):
            make_axes(np.zeros((1, N_PROMPTS + 1)))

    def test_vocal_lead_and_acousticness_signs(self):
        row = _scores_row(vocal=0.6, instrumental=0.1, acoustic=0.1, synthetic=0.6)
        by_name = dict(zip(AXIS_NAMES, make_axes(row)[0]))
        assert by_name["vocal_lead"] > 0          # vocal-dominant
        assert by_name["acousticness"] < 0        # synthetic-dominant


class TestAxesForClapVectors:
    def test_projection_matches_manual_matmul(self):
        rng = np.random.default_rng(7)
        dim = 16
        text_emb = rng.normal(size=(N_PROMPTS, dim)).astype(np.float32)
        vec = rng.normal(size=(2, dim)).astype(np.float32)

        result = axes_for_clap_vectors(vec, text_emb)

        expected = make_axes(vec @ text_emb.T)
        assert len(result) == 2
        for i, d in enumerate(result):
            assert tuple(d) == AXIS_NAMES           # keys, in order
            for j, name in enumerate(AXIS_NAMES):
                assert d[name] == pytest.approx(float(expected[i, j]), abs=1e-5)

    def test_values_are_plain_floats(self):
        text_emb = np.eye(N_PROMPTS, 8, dtype=np.float32)
        result = axes_for_clap_vectors(np.ones((1, 8), dtype=np.float32), text_emb)
        assert all(type(v) is float for v in result[0].values())


class TestAxisVersion:
    def test_deterministic_12_hex_chars(self):
        v = axis_version()
        assert v == axis_version()
        assert len(v) == 12
        int(v, 16)  # raises if not hex

    def test_dense_is_stub_not_active(self):
        assert "dense" in AXIS_PROMPT_STUBS
        assert "dense" not in AXIS_PROMPTS


# --------------------------------------------------------------------------- #
# From test_sonic_descriptor_tags.py
# --------------------------------------------------------------------------- #
class TestSonicDescriptorTags:
    def test_service_loads_vocab(self, sample_vocab_file):
        svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file)
        prompts = svc.load_prompt_vocab()
        assert prompts == ["punchy", "ambient", "sad", "happy"]

    def test_missing_vocab_file_falls_back_to_builtin_default(self, tmp_path):
        """cache/ is wipeable — the service must work without the JSON file."""
        from app.services.sonic_descriptor_service import DEFAULT_VOCAB

        svc = SonicDescriptorService(prompt_vocab_path=tmp_path / "nope.json")
        prompts = svc.load_prompt_vocab()

        expected = [p for group in DEFAULT_VOCAB["groups"].values() for p in group]
        assert prompts == expected
        assert len(prompts) == 40  # 40 adjectives across 7 axes

    def test_vocab_file_overrides_builtin_default(self, sample_vocab_file):
        svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file)
        assert svc.load_prompt_vocab() == ["punchy", "ambient", "sad", "happy"]

    def test_service_default_top_k_is_five(self):
        svc = SonicDescriptorService()
        assert svc.top_k_tags == 5

    def test_service_uses_explicit_top_k(self, sample_vocab_file):
        svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, top_k_tags=3)
        assert svc.top_k_tags == 3

    def test_prompt_embeddings_computed_via_clap_when_cache_missing(self, sample_vocab_file, tmp_path, monkeypatch):
        embeddings_path = tmp_path / "embeds.npy"
        svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, embeddings_path=embeddings_path)

        fake_emb = np.random.rand(4, 512).astype(np.float32)
        fake_clap = MagicMock()
        fake_clap.get_text_embedding.return_value = fake_emb
        monkeypatch.setattr(
            "app.resources.model_registry.ModelRegistry.load_clap",
            lambda: fake_clap,
        )

        out = svc.load_or_compute_prompt_embeddings()
        assert out.shape == (4, 512)
        assert embeddings_path.exists()
        fake_clap.get_text_embedding.assert_called_once_with(["punchy", "ambient", "sad", "happy"], use_tensor=False)

    def test_prompt_embeddings_loaded_from_cache(self, sample_vocab_file, tmp_path, monkeypatch):
        cached = np.random.rand(4, 512).astype(np.float32)
        embeddings_path = tmp_path / "embeds.npy"
        np.save(embeddings_path, cached)

        svc = SonicDescriptorService(prompt_vocab_path=sample_vocab_file, embeddings_path=embeddings_path)

        fake_clap = MagicMock()
        monkeypatch.setattr(
            "app.resources.model_registry.ModelRegistry.load_clap",
            lambda: fake_clap,
        )

        out = svc.load_or_compute_prompt_embeddings()
        np.testing.assert_array_equal(out, cached)
        fake_clap.get_text_embedding.assert_not_called()

    def test_compute_tags_returns_top_k_sorted(self, sample_vocab_file, tmp_path, monkeypatch):
        # 4 prompts: ["punchy", "ambient", "sad", "happy"]
        # Construct prompt embeddings so cosine to a known audio vector is predictable.
        embeddings_path = tmp_path / "embeds.npy"
        prompt_embs = np.array([
            [1.0, 0.0, 0.0],   # punchy
            [0.0, 1.0, 0.0],   # ambient
            [0.0, 0.0, 1.0],   # sad
            [0.5, 0.5, 0.5],   # happy (normalized below)
        ], dtype=np.float32)
        # Normalize each row
        prompt_embs = prompt_embs / np.linalg.norm(prompt_embs, axis=1, keepdims=True)
        np.save(embeddings_path, prompt_embs)

        svc = SonicDescriptorService(
            prompt_vocab_path=sample_vocab_file,
            embeddings_path=embeddings_path,
            top_k_tags=3,
        )

        # Audio vector very close to "punchy" (1,0,0)
        audio = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        audio = audio / np.linalg.norm(audio)

        tags = svc.compute_tags(audio_vector=audio)
        assert len(tags) == 3
        # First tag is "punchy" with highest similarity
        assert tags[0]["tag"] == "punchy"
        assert tags[0]["score"] > tags[1]["score"] >= tags[2]["score"]
        # All scores in 0..1 range (cosine of unit vectors)
        assert all(0 <= t["score"] <= 1 for t in tags)

    def test_compute_tags_bulk_persists_per_track(self, sample_vocab_file, tmp_path, monkeypatch):
        # Pre-stage prompt embeddings
        embeddings_path = tmp_path / "embeds.npy"
        prompt_embs = np.eye(4, dtype=np.float32)  # 4 unit vectors
        np.save(embeddings_path, prompt_embs)

        svc = SonicDescriptorService(
            prompt_vocab_path=sample_vocab_file,
            embeddings_path=embeddings_path,
            top_k_tags=2,
        )

        # Mock Qdrant scroll: 3 points, each with a known vector + slug
        fake_points = []
        for i, vec in enumerate([
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
        ]):
            p = MagicMock()
            p.id = f"track-{i}"
            p.payload = {"slug": f"slug-{i}", "title": f"T{i}", "artist": f"A{i}"}
            p.vector = {"audio": vec.tolist()}
            fake_points.append(p)

        fake_qdrant = MagicMock()
        fake_qdrant.scroll.side_effect = [(fake_points, None)]

        persisted: list[tuple] = []
        monkeypatch.setattr(
            "app.resources.metadata_db.MetadataDB.upsert_sonic_descriptor",
            lambda song_slug, tags=None, sonic_class=None, confidence=None, audio_signature=None: persisted.append((song_slug, tags)),
        )

        svc.compute_tags_bulk(qdrant=fake_qdrant, collection="test_col", audio_vector_name="audio")
        assert len(persisted) == 3
        # First track should match "prompt 0" (eye matrix → prompt 0 is [1,0,0,0])
        assert persisted[0][0] == "slug-0"
        assert persisted[0][1][0]["tag"] == "punchy"  # first prompt in vocab


# --------------------------------------------------------------------------- #
# From test_sonic_facets_aggregate.py
# --------------------------------------------------------------------------- #
class TestSonicFacetsAggregate:
    def test_get_sonic_facets_returns_empty_for_empty_db(self, db):
        out = MetadataDB.get_sonic_facets()
        assert out == {"tags": []}

    def test_get_sonic_facets_aggregates_tags_from_json(self, db):
        # song row must exist before upsert_sonic_descriptor (UPDATE WHERE slug=?).
        db.upsert_artist("a", "A", "test_col")
        db.upsert_song("a-1", "T1", "a", "test_col")
        db.upsert_song("a-2", "T2", "a", "test_col")
        db.upsert_sonic_descriptor(
            song_slug="a-1",
            tags=[{"tag": "melancholic", "score": 0.8}, {"tag": "lo-fi", "score": 0.7}],
        )
        db.upsert_sonic_descriptor(
            song_slug="a-2",
            tags=[{"tag": "melancholic", "score": 0.9}],
        )
        out = MetadataDB.get_sonic_facets()
        tag_counts = {t["value"]: t["count"] for t in out["tags"]}
        assert tag_counts == {"melancholic": 2, "lo-fi": 1}

    def test_get_sonic_facets_sorts_tags_descending_by_count(self, db):
        db.upsert_artist("a", "A", "test_col")
        db.upsert_song("a-1", "T1", "a", "test_col")
        db.upsert_song("b-1", "T2", "a", "test_col")
        db.upsert_song("b-2", "T3", "a", "test_col")
        db.upsert_song("b-3", "T4", "a", "test_col")
        db.upsert_sonic_descriptor(
            song_slug="a-1",
            tags=[{"tag": "rare", "score": 0.5}],
        )
        for slug in ["b-1", "b-2", "b-3"]:
            db.upsert_sonic_descriptor(
                song_slug=slug,
                tags=[{"tag": "common", "score": 0.5}],
            )
        out = MetadataDB.get_sonic_facets()
        assert out["tags"][0]["value"] == "common"
        assert out["tags"][0]["count"] == 3
        assert out["tags"][1]["value"] == "rare"

    def test_get_sonic_facets_tolerates_malformed_tags_json(self, db):
        """A bad sonic_tags_json row should not crash the aggregate."""
        db.upsert_artist("a", "A", "test_col")
        db.upsert_song("a-1", "T1", "a", "test_col")
        # Directly write a malformed value via raw SQL since upsert_sonic_descriptor json.dumps's it.
        conn = db._connect()
        conn.execute("UPDATE songs SET sonic_tags_json = ? WHERE slug = ?", ("not-json", "a-1"))
        conn.commit()
        out = MetadataDB.get_sonic_facets()
        assert out["tags"] == []

    def test_get_sonic_facets_caps_to_top_k(self, db):
        db.upsert_artist("a", "A", "test_col")
        for i in range(50):
            slug = f"a-{i}"
            db.upsert_song(slug, f"T{i}", "a", "test_col")
            db.upsert_sonic_descriptor(
                song_slug=slug,
                tags=[{"tag": f"tag-{i}", "score": 0.5}],
            )
        out = MetadataDB.get_sonic_facets(top_k=10)
        assert len(out["tags"]) == 10


# --------------------------------------------------------------------------- #
# From test_axis_norm_blend.py
# --------------------------------------------------------------------------- #
class TestBlendAxisStats:
    def test_both_missing_returns_none(self):
        assert blend_axis_stats(None, None) is None

    def test_collection_only(self):
        out = blend_axis_stats(_stats(1.0, 2.0, n=50), None)
        assert out["source"] == "collection"
        assert out["mean"]["energy"] == 1.0
        assert out["n"] == 50

    def test_reference_only(self):
        out = blend_axis_stats(None, _stats(3.0, 4.0, n=5000))
        assert out["source"] == "reference"
        assert out["std"]["energy"] == 4.0

    def test_blend_lambda_at_n_100_is_half(self):
        """λ = n/(n+100): n=100 → exact 50/50 mix."""
        out = blend_axis_stats(_stats(0.0, 0.0, n=100), _stats(1.0, 1.0, n=5000))
        assert out["source"] == "blend"
        assert out["lambda"] == pytest.approx(0.5)
        assert out["mean"]["energy"] == pytest.approx(0.5)
        assert out["std"]["energy"] == pytest.approx(0.5)

    def test_tiny_collection_leans_on_reference(self):
        """n=5 → 95% reference weight (design §8 example)."""
        out = blend_axis_stats(_stats(1.0, 1.0, n=5), _stats(0.0, 0.0, n=5000))
        assert out["mean"]["energy"] == pytest.approx(5 / 105)

    def test_version_stale_collection_dropped(self):
        out = blend_axis_stats(_stats(1.0, 1.0, n=50, version="stale0000000"),
                               _stats(2.0, 2.0, n=5000))
        assert out["source"] == "reference"

    def test_version_stale_reference_dropped(self):
        out = blend_axis_stats(_stats(1.0, 1.0, n=50),
                               _stats(2.0, 2.0, n=5000, version="stale0000000"))
        assert out["source"] == "collection"

    def test_both_stale_returns_none(self):
        assert blend_axis_stats(_stats(1.0, 1.0, n=50, version="staleA000000"),
                                _stats(2.0, 2.0, n=10, version="staleB000000")) is None


class TestLoadAxisNormReference:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_axis_norm_reference(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        p = tmp_path / "ref.json"
        p.write_text("{broken", encoding="utf-8")
        assert load_axis_norm_reference(p) is None

    def test_stale_version_returns_none(self, tmp_path):
        p = tmp_path / "ref.json"
        p.write_text(json.dumps(_stats(0.0, 1.0, n=5000, version="cafebabe0000")),
                     encoding="utf-8")
        assert load_axis_norm_reference(p) is None

    def test_valid_file_loads(self, tmp_path):
        p = tmp_path / "ref.json"
        p.write_text(json.dumps(_stats(0.0, 1.0, n=5000)), encoding="utf-8")
        ref = load_axis_norm_reference(p)
        assert ref is not None
        assert ref["n"] == 5000
