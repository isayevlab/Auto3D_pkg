#!/usr/bin/env python
"""Geometric measurements on a conformer's coordinates.

Distances only (RMSD lives in ``Auto3D.domain.filtering``, the only production
caller). Nothing here reads or writes a molecular property, consults a force
field, or decides whether a structure is acceptable -- those belong to
``utils/connectivity.py``, ``Auto3D.domain.clash_relief`` and
``Auto3D.domain.filtering`` respectively.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("auto3d")

__all__ = ["min_pairwise_distance"]


def min_pairwise_distance(points: np.ndarray) -> float:
    """Find the minimum pairwise distance among n points in 3D space.

    This function computes all pairwise distances between the provided points
    and returns the minimum distance. It uses vectorized NumPy operations
    for efficiency.

    Args:
        points: A (n, 3) array representing the coordinates of n points
            in 3D space.

    Returns:
        The minimum pairwise distance among the n points.

    Example:
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        >>> min_pairwise_distance(points)
        1.0
    """
    # Ensure input is a NumPy array with float32 type
    points = points.astype(np.float32)
    n = points.shape[0]

    # Guard for single atom or empty input
    if n < 2:
        # Single atom: no pairwise distance exists
        return float("inf")

    # Expand dimensions of points to enable broadcasting
    points_expanded = np.expand_dims(points, axis=1).repeat(n, axis=1)

    # Compute pairwise squared differences
    diff_squared = (points_expanded - points_expanded.transpose(1, 0, 2)) ** 2

    # Sum along the last dimension to get pairwise squared distances
    pairwise_squared_distances = np.sum(diff_squared, axis=-1)

    # Find the minimum squared distance from upper triangle
    upp_indices = np.triu_indices(n, 1)
    upp_values = pairwise_squared_distances[upp_indices]
    min_squared_distance = np.min(upp_values)

    # Return the square root of the minimum squared distance
    return float(np.sqrt(min_squared_distance))
