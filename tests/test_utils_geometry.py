#!/usr/bin/env python
"""Tests for Auto3D.foundation.utils.geometry module."""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401  (several tests below are parametrized helpers' home)

from Auto3D.foundation.utils.geometry import min_pairwise_distance


class TestMinPairwiseDistance:
    """Test the min_pairwise_distance function."""

    def test_simple_three_points(self):
        """Test with three simple points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_two_points(self):
        """Test with two points."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],  # Distance = 5
            ]
        )
        result = min_pairwise_distance(points)
        assert abs(result - 5.0) < 1e-5

    def test_collinear_points(self):
        """Test with collinear points."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_3d_points(self):
        """Test with points in 3D space."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],  # Distance = sqrt(3) ~ 1.732
                [5.0, 5.0, 5.0],
            ]
        )
        result = min_pairwise_distance(points)
        expected = np.sqrt(3)
        assert abs(result - expected) < 1e-5

    def test_input_type_conversion(self):
        """Test that integer input is properly converted."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.int32)
        result = min_pairwise_distance(points)
        assert abs(result - 1.0) < 1e-5

    def test_very_close_points(self):
        """Test with very close points."""
        points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = min_pairwise_distance(points)
        assert abs(result - 0.001) < 1e-6
