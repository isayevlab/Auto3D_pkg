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

__all__ = ["min_distance_among", "min_pairwise_distance"]


def min_pairwise_distance(points: np.ndarray, exclude: set[frozenset[int]] | None = None) -> float:
    """Find the minimum pairwise distance among n points in 3D space.

    This function computes all pairwise distances between the provided points
    and returns the minimum distance. It uses vectorized NumPy operations.

    Args:
        points: A (n, 3) array representing the coordinates of n points
            in 3D space.
        exclude: Index pairs to leave out of the minimum, each written as a
            two-element ``frozenset`` of the two points' positions in
            ``points``. A bonded pair is not a clash candidate (N-m1): a
            diatomic whose only pair is excluded has no pair left to compare,
            and returns ``inf``, the same as the single-atom guard below.
            This module stays RDKit-free, so the caller (``clash_relief``)
            is the one that knows which index pairs are bonds.

    Returns:
        The minimum pairwise distance among the n points, over every pair not
        named in ``exclude``.

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

    if exclude:
        # A boolean mask over the same upper-triangle pairs, built once from
        # `exclude` rather than testing membership per pair: `exclude` is
        # small (one entry per bond) and the upper triangle is O(n^2), so a
        # vectorized mask avoids a Python-level `frozenset in exclude` test
        # inside the loop that produced `upp_indices`.
        #
        # No bounds guard (M6): an index beyond `n` is a caller error -- a
        # pair from a different molecule, or a conformer with a different
        # atom count -- and silently dropping it out of the mask would
        # *weaken* the clash test rather than report the mistake, which is
        # the unsafe direction for a safety check. Let numpy's own
        # IndexError surface instead.
        mask = np.zeros((n, n), dtype=bool)
        for pair in exclude:
            i, j = pair
            mask[i, j] = True
            mask[j, i] = True
        keep = ~mask[upp_indices]
        upp_values = upp_values[keep]
        if upp_values.size == 0:
            # Every pair was excluded (e.g. a diatomic whose one pair is its
            # own bond): there is nothing left to call a clash, so this reads
            # the same as the n < 2 guard above.
            return float("inf")

    min_squared_distance = np.min(upp_values)

    # Return the square root of the minimum squared distance
    return float(np.sqrt(min_squared_distance))


def min_distance_among(points: np.ndarray, pairs: set[frozenset[int]]) -> float:
    """Find the minimum distance among exactly the given index pairs.

    The complement of :func:`min_pairwise_distance`'s ``exclude``: where that
    function finds the minimum over every pair NOT named, this finds it only
    over the pairs that ARE named. ``clash_relief`` uses it to ask about
    bonded pairs specifically -- the ones ``min_pairwise_distance`` excludes
    from its own, non-bonded test (C1): excluding a bonded pair from the
    clash test is right, but a bonded pair still needs its OWN floor, or a
    physically impossible bond length (a crushed embedding) passes through
    with no check at all.

    Args:
        points: A (n, 3) array representing the coordinates of n points
            in 3D space.
        pairs: Index pairs to measure, each a two-element ``frozenset`` of
            the two points' positions in ``points``.

    Returns:
        The minimum distance among the given pairs, or ``inf`` if ``pairs``
        is empty (e.g. a lone atom with no bonds at all).

    Example:
        >>> import numpy as np
        >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        >>> min_distance_among(points, {frozenset((0, 1))})
        1.0
    """
    if not pairs:
        return float("inf")
    points = points.astype(np.float32)
    distances = [
        float(np.linalg.norm(points[i] - points[j])) for i, j in (tuple(pair) for pair in pairs)
    ]
    return min(distances)
