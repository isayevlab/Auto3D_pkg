#!/usr/bin/env python
"""Tests for the unit-conversion factors and legacy aliases in Auto3D.utils.energy."""

from __future__ import annotations

from Auto3D.utils.energy import (
    EV_TO_KCAL_PER_MOL,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    ev2kcalpermol,
    hartree2ev,
    hartree2kcalpermol,
)


class TestEnergyConversionConstants:
    """Test energy conversion constants and aliases."""

    def test_hartree_to_ev_value(self):
        """Test that HARTREE_TO_EV has the correct CODATA 2018 value."""
        assert abs(HARTREE_TO_EV - 27.211386245988) < 1e-10

    def test_hartree_to_kcal_per_mol_value(self):
        """Test that HARTREE_TO_KCAL_PER_MOL has the expected value."""
        assert abs(HARTREE_TO_KCAL_PER_MOL - 627.50947337481) < 1e-8

    def test_ev_to_kcal_per_mol_value(self):
        """Test that EV_TO_KCAL_PER_MOL has the expected value."""
        assert abs(EV_TO_KCAL_PER_MOL - 23.060547830619026) < 1e-10

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases match constants."""
        assert hartree2ev == HARTREE_TO_EV
        assert hartree2kcalpermol == HARTREE_TO_KCAL_PER_MOL
        assert ev2kcalpermol == EV_TO_KCAL_PER_MOL

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
    ``Auto3D.utils`` package barrel, asserting that both paths worked. The
    barrel is gone (``tests/test_import_boundaries.py`` now forbids it), so the
    two-paths-for-one-name shape it pinned is the thing being prevented rather
    than checked.
    """

    def test_import_from_defining_modules(self):
        """Each name resolves at the module that now defines it."""
        from Auto3D.utils.connectivity import check_connectivity
        from Auto3D.utils.energy import HARTREE_TO_EV, hartree2ev
        from Auto3D.utils.geometry import get_rmsd, min_pairwise_distance
        from Auto3D.utils.molprops import get_mol_charge

        assert HARTREE_TO_EV == hartree2ev
        assert callable(get_mol_charge)
        assert callable(min_pairwise_distance)
        assert callable(get_rmsd)
        assert callable(check_connectivity)
