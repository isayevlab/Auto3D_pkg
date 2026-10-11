#!/usr/bin/env python
"""Tests for the unit-conversion factors in Auto3D.domain.energy
(formerly Auto3D.foundation.utils.energy).

R17: the three legacy lowercase aliases (``hartree2ev``, ``hartree2kcalpermol``,
``ev2kcalpermol``) are gone. They were importable but never documented public
API; the two live callers (``entry.tautomer``, ``domain.ranking``) now import
the constants (``HARTREE_TO_KCAL_PER_MOL``, ``EV_TO_KCAL_PER_MOL``) directly.
"""

from __future__ import annotations

from Auto3D.domain import energy
from Auto3D.domain.energy import EV_TO_KCAL_PER_MOL, HARTREE_TO_EV, HARTREE_TO_KCAL_PER_MOL


class TestEnergyConversionConstants:
    """Test energy conversion constants."""

    def test_hartree_to_ev_value(self):
        """Test that HARTREE_TO_EV has the correct CODATA 2018 value."""
        assert abs(HARTREE_TO_EV - 27.211386245988) < 1e-10

    def test_hartree_to_kcal_per_mol_value(self):
        """Test that HARTREE_TO_KCAL_PER_MOL has the expected value."""
        assert abs(HARTREE_TO_KCAL_PER_MOL - 627.50947337481) < 1e-8

    def test_ev_to_kcal_per_mol_value(self):
        """Test that EV_TO_KCAL_PER_MOL has the expected value."""
        assert abs(EV_TO_KCAL_PER_MOL - 23.060547830619026) < 1e-10

    def test_the_legacy_lowercase_aliases_are_gone(self):
        """R17: hartree2ev, hartree2kcalpermol and ev2kcalpermol are removed."""
        assert not hasattr(energy, "hartree2ev")
        assert not hasattr(energy, "hartree2kcalpermol")
        assert not hasattr(energy, "ev2kcalpermol")

    def test_conversion_consistency(self):
        """Test that conversion factors are mathematically consistent."""
        # HARTREE_TO_KCAL_PER_MOL should approximately equal
        # HARTREE_TO_EV * EV_TO_KCAL_PER_MOL
        calculated = HARTREE_TO_EV * EV_TO_KCAL_PER_MOL
        # Allow some tolerance for floating point precision
        assert abs(calculated - HARTREE_TO_KCAL_PER_MOL) < 0.001


class TestModuleImports:
    """Test that all expected names are importable from their defining modules.

    There used to be a second test here importing the same names out of the
    ``Auto3D.foundation.utils`` package barrel, asserting that both paths worked. The
    barrel is gone (``tests/test_import_boundaries.py`` now forbids it), so the
    two-paths-for-one-name shape it pinned is the thing being prevented rather
    than checked.
    """

    def test_import_from_defining_modules(self):
        """Each name resolves at the module that now defines it."""
        from Auto3D.domain.connectivity import check_connectivity
        from Auto3D.domain.energy import HARTREE_TO_EV
        from Auto3D.domain.geometry import min_pairwise_distance

        assert HARTREE_TO_EV == 27.211386245988
        assert callable(min_pairwise_distance)
        assert callable(check_connectivity)
