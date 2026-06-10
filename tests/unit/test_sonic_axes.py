"""Unit tests for sonic axes (clap_features) — notebook cell-67 port.

Covers the prompt→axis differential math, payload-dict projection, input
validation, and axis-space versioning. No CLAP model required: text
embeddings are faked with small matrices.
"""

import numpy as np
import pytest

from app.resources.clap_features import (
    AXIS_NAMES,
    AXIS_PROMPT_STUBS,
    AXIS_PROMPTS,
    axes_for_clap_vectors,
    axis_version,
    make_axes,
)

N_PROMPTS = len(AXIS_PROMPTS)


def _scores_row(**overrides) -> np.ndarray:
    """One row of prompt scores, zeros except named prompts."""
    row = np.zeros(N_PROMPTS, dtype=np.float32)
    names = list(AXIS_PROMPTS)
    for name, value in overrides.items():
        row[names.index(name)] = value
    return row


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
