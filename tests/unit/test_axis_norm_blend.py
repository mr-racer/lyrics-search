"""Unit tests for shrinkage blending + reference loading (clap_features)."""
import json

import pytest

from app.resources.clap_features import (
    AXIS_NAMES,
    axis_version,
    blend_axis_stats,
    load_axis_norm_reference,
)


def _stats(mean_val: float, std_val: float, n: int, version: str | None = None) -> dict:
    return {
        "version": version if version is not None else axis_version(),
        "n": n,
        "mean": {a: mean_val for a in AXIS_NAMES},
        "std": {a: std_val for a in AXIS_NAMES},
    }


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
