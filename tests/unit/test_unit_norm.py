"""Tests for search_engine.utils.unit_norm()."""

import numpy as np
import pytest

from search_engine.utils import unit_norm


class TestUnitNorm:
    def test_unit_vector_returns_itself(self):
        v = np.array([1.0, 0.0, 0.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_zero_vector_returns_zero(self):
        v = np.array([0.0, 0.0, 0.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_random_vector_normalizes_to_unit_length(self):
        v = np.array([3.0, 4.0, 0.0])
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_1d_vector_works(self):
        v = np.array([5.0])
        result = unit_norm(v)
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_negative_values_handled(self):
        v = np.array([-3.0, 4.0, 0.0])
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        assert result[0] < 0

    def test_returns_numpy_array(self):
        v = np.array([1.0, 2.0, 3.0])
        result = unit_norm(v)
        assert isinstance(result, np.ndarray)

    def test_high_dim_vector(self):
        v = np.random.randn(512)
        result = unit_norm(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
