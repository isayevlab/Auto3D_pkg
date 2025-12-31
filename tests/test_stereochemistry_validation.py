#!/usr/bin/env python
"""Tests for ValueError validation in stereochemistry.py.

These tests verify that assert statements have been replaced with proper
ValueError exceptions that provide informative error messages.
"""

import pytest
from Auto3D.utils.stereochemistry import enantiomer, no_enantiomer_helper


class TestEnantiomerValidation:
    """Tests for the enantiomer() function validation."""

    def test_enantiomer_mismatched_lengths_raises_valueerror(self):
        """enantiomer() should raise ValueError for mismatched list lengths."""
        l1 = [(0, 'R'), (1, 'S')]
        l2 = [(0, 'R')]  # Different length

        with pytest.raises(ValueError, match="length"):
            enantiomer(l1, l2)

    def test_enantiomer_mismatched_lengths_error_message(self):
        """Error message should contain actual lengths."""
        l1 = [(0, 'R'), (1, 'S'), (2, 'R')]
        l2 = [(0, 'R')]

        with pytest.raises(ValueError) as exc_info:
            enantiomer(l1, l2)

        error_msg = str(exc_info.value)
        assert "3" in error_msg
        assert "1" in error_msg

    def test_enantiomer_mismatched_indices_raises_valueerror(self):
        """enantiomer() should raise ValueError for mismatched stereo center indices."""
        l1 = [(0, 'R'), (1, 'S')]
        l2 = [(0, 'S'), (5, 'R')]  # Second index doesn't match

        with pytest.raises(ValueError, match="indices"):
            enantiomer(l1, l2)

    def test_enantiomer_mismatched_indices_error_message(self):
        """Error message should contain mismatched indices and position."""
        l1 = [(0, 'R'), (1, 'S')]
        l2 = [(0, 'S'), (5, 'R')]

        with pytest.raises(ValueError) as exc_info:
            enantiomer(l1, l2)

        error_msg = str(exc_info.value)
        assert "1" in error_msg  # idx1
        assert "5" in error_msg  # idx2
        assert "position" in error_msg.lower()

    def test_enantiomer_valid_enantiomers_returns_true(self):
        """Valid enantiomers should return True without raising."""
        l1 = [(0, 'R'), (1, 'S')]
        l2 = [(0, 'S'), (1, 'R')]

        result = enantiomer(l1, l2)
        assert result is True

    def test_enantiomer_non_enantiomers_returns_false(self):
        """Non-enantiomers (same stereo at any center) should return False."""
        l1 = [(0, 'R'), (1, 'S')]
        l2 = [(0, 'R'), (1, 'R')]  # First stereo matches

        result = enantiomer(l1, l2)
        assert result is False

    def test_enantiomer_empty_lists_returns_true(self):
        """Empty lists (no stereo centers) should return True."""
        l1: list[tuple[int, str]] = []
        l2: list[tuple[int, str]] = []

        result = enantiomer(l1, l2)
        assert result is True


class TestNoEnantiomerHelperValidation:
    """Tests for the no_enantiomer_helper() function validation."""

    def test_no_enantiomer_helper_mismatched_lengths_raises_valueerror(self):
        """no_enantiomer_helper() should raise ValueError for mismatched lengths."""
        info1 = ['@', '@@']
        info2 = ['@']  # Different length

        with pytest.raises(ValueError, match="length"):
            no_enantiomer_helper(info1, info2)

    def test_no_enantiomer_helper_mismatched_lengths_error_message(self):
        """Error message should contain actual lengths."""
        info1 = ['@', '@@', '@']
        info2 = ['@@']

        with pytest.raises(ValueError) as exc_info:
            no_enantiomer_helper(info1, info2)

        error_msg = str(exc_info.value)
        assert "3" in error_msg
        assert "1" in error_msg

    def test_no_enantiomer_helper_enantiomers_returns_true(self):
        """Enantiomeric stereo info should return True."""
        info1 = ['@', '@@']
        info2 = ['@@', '@']

        result = no_enantiomer_helper(info1, info2)
        assert result is True

    def test_no_enantiomer_helper_non_enantiomers_returns_false(self):
        """Non-enantiomeric stereo info should return False."""
        info1 = ['@', '@@']
        info2 = ['@', '@']  # First symbol matches

        result = no_enantiomer_helper(info1, info2)
        assert result is False

    def test_no_enantiomer_helper_empty_lists_returns_true(self):
        """Empty lists should return True."""
        info1: list[str] = []
        info2: list[str] = []

        result = no_enantiomer_helper(info1, info2)
        assert result is True


class TestNoAssertionError:
    """Tests to verify AssertionError is NOT raised (proper ValueError instead)."""

    def test_enantiomer_does_not_raise_assertion_error_for_length(self):
        """Should raise ValueError, not AssertionError, for length mismatch."""
        l1 = [(0, 'R')]
        l2 = [(0, 'S'), (1, 'R')]

        with pytest.raises(ValueError):
            enantiomer(l1, l2)

        # Explicitly verify it's not AssertionError
        try:
            enantiomer(l1, l2)
        except AssertionError:
            pytest.fail("enantiomer() raised AssertionError instead of ValueError")
        except ValueError:
            pass  # Expected

    def test_enantiomer_does_not_raise_assertion_error_for_indices(self):
        """Should raise ValueError, not AssertionError, for index mismatch."""
        l1 = [(0, 'R')]
        l2 = [(1, 'S')]

        with pytest.raises(ValueError):
            enantiomer(l1, l2)

        try:
            enantiomer(l1, l2)
        except AssertionError:
            pytest.fail("enantiomer() raised AssertionError instead of ValueError")
        except ValueError:
            pass  # Expected

    def test_no_enantiomer_helper_does_not_raise_assertion_error(self):
        """Should raise ValueError, not AssertionError, for length mismatch."""
        info1 = ['@']
        info2 = ['@', '@@']

        with pytest.raises(ValueError):
            no_enantiomer_helper(info1, info2)

        try:
            no_enantiomer_helper(info1, info2)
        except AssertionError:
            pytest.fail("no_enantiomer_helper() raised AssertionError instead of ValueError")
        except ValueError:
            pass  # Expected
