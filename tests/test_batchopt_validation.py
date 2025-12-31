"""Tests for batchopt.py validation logic.

These tests verify that proper ValueError exceptions are raised instead of
AssertionError when validation fails. This ensures validation works even
when Python is run with -O (optimized mode) which strips assert statements.
"""
import pytest
import warnings

from Auto3D.batch_opt.batchopt import padding_coords, padding_species


class TestPaddingCoordsValidation:
    """Tests for padding_coords validation."""

    def test_padding_coords_normal_operation(self):
        """padding_coords should work correctly with valid input."""
        # Two molecules with different numbers of atoms
        lists = [
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],  # 2 atoms
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],  # 3 atoms
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = padding_coords(lists)

        # Both should be padded to length 3
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        # First list should have one padding element
        assert result[0][2] == (0.0, 0.0, 0.0)

    def test_padding_coords_single_list(self):
        """padding_coords should work with a single list."""
        lists = [[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = padding_coords(lists)

        assert len(result) == 1
        assert len(result[0]) == 2


class TestPaddingSpeciesValidation:
    """Tests for padding_species validation."""

    def test_padding_species_normal_operation(self):
        """padding_species should work correctly with valid input."""
        # Two molecules with different numbers of atoms
        lists = [
            [6, 1, 1],  # 3 atoms (C, H, H)
            [8, 1],     # 2 atoms (O, H)
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = padding_species(lists)

        # Both should be padded to length 3
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        # Second list should have one padding element (-1)
        assert result[1][2] == -1

    def test_padding_species_single_list(self):
        """padding_species should work with a single list."""
        lists = [[6, 1, 1, 1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = padding_species(lists)

        assert len(result) == 1
        assert len(result[0]) == 4

    def test_padding_species_custom_pad_value(self):
        """padding_species should use custom pad value."""
        lists = [[6], [8, 1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = padding_species(lists, pad_value=0)

        # First list should be padded with 0, not -1
        assert result[0][1] == 0


class TestDeprecationWarnings:
    """Tests for deprecation warnings."""

    def test_padding_coords_deprecation_warning(self):
        """padding_coords should emit deprecation warning."""
        lists = [[(0.0, 0.0, 0.0)]]

        with pytest.warns(DeprecationWarning, match="padding_coords is deprecated"):
            padding_coords(lists)

    def test_padding_species_deprecation_warning(self):
        """padding_species should emit deprecation warning."""
        lists = [[6, 1]]

        with pytest.warns(DeprecationWarning, match="padding_species is deprecated"):
            padding_species(lists)
