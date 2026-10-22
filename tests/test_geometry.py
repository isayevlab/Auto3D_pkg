#!/usr/bin/env python
"""Tests for Auto3D.domain.geometry module (formerly Auto3D.foundation.utils.geometry)."""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401  (several tests below are parametrized helpers' home)

from Auto3D.domain.geometry import min_distance_among, min_pairwise_distance


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


class TestMinPairwiseDistanceExclude:
    """``exclude`` masks index pairs out of the minimum directly,
    independent of the ``clash_relief`` caller that builds the pairs."""

    def test_excluding_the_closest_pair_falls_through_to_the_next_smallest(self):
        points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = min_pairwise_distance(points, exclude={frozenset((0, 1))})
        # Pairs (0,2) and (1,2) remain at 10.0 and 9.999; the smaller of the
        # two survives the exclusion of (0,1)'s 0.001.
        assert abs(result - 9.999) < 1e-3

    def test_excluding_every_pair_of_a_diatomic_returns_inf(self):
        points = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        result = min_pairwise_distance(points, exclude={frozenset((0, 1))})
        assert result == float("inf")

    def test_an_empty_exclude_set_changes_nothing(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        result = min_pairwise_distance(points, exclude=set())
        assert abs(result - 1.0) < 1e-5

    def test_an_out_of_range_index_raises_rather_than_silently_weakening_the_test(self):
        """A bad ``exclude`` pair must not be silently dropped.

        A pair naming an index beyond the point count is a caller error (a
        pair from a different molecule, or a conformer with a different atom
        count); dropping it out of the mask quietly would weaken the clash
        test it is supposed to strengthen, which is the unsafe direction for
        a safety check. It must raise instead.
        """
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        with pytest.raises(IndexError):
            min_pairwise_distance(points, exclude={frozenset((0, 99))})


class TestMinDistanceAmong:
    """The complement of ``exclude`` -- the minimum over only the
    named pairs, used by ``clash_relief`` to test bonded pairs specifically."""

    def test_minimum_among_the_named_pairs_only(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        # (0, 1) = 1.0, (0, 2) = 5.0; (1, 2) (not named) would be smaller
        # than neither, so this also confirms unnamed pairs are ignored.
        result = min_distance_among(points, {frozenset((0, 1)), frozenset((0, 2))})
        assert abs(result - 1.0) < 1e-5

    def test_empty_pairs_returns_inf(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert min_distance_among(points, set()) == float("inf")
