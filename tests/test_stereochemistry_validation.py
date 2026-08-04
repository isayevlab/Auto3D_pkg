#!/usr/bin/env python
"""Tests for ValueError validation in stereochemistry.py.

These tests verify that assert statements have been replaced with proper
ValueError exceptions that provide informative error messages.
"""

import pytest

from Auto3D.utils.stereochemistry import (
    create_enantiomer,
    enantiomer,
    no_enantiomer,
    no_enantiomer_helper,
)


class TestCreateEnantiomerStereoClasses:
    """create_enantiomer must not corrupt non-tetrahedral stereo descriptors."""

    @pytest.mark.parametrize(
        "smi",
        ["F[Po@SP1](Cl)(Br)I", "[C@TH1](F)(Cl)(Br)I"],
    )
    def test_nontetrahedral_tokens_stay_valid(self, smi):
        """@SP/@TH/... must be left intact, not spliced into invalid @@SP1.

        Treating a multi-letter stereo class as a bare '@' used to insert a
        second '@', producing an unparseable SMILES that aborted the whole
        molecule's stereo enumeration.
        """
        from rdkit import Chem
        out = create_enantiomer(smi)
        assert Chem.MolFromSmiles(out) is not None, out
        assert "@@SP" not in out and "@@TH" not in out

    def test_tetrahedral_inversion_still_works(self):
        """Plain tetrahedral centers must still be inverted."""
        assert create_enantiomer("C[C@H](O)F") == "C[C@@H](O)F"


class TestNoEnantiomerLengthMismatch:
    """no_enantiomer must tolerate isomers with differing stereo-marker counts."""

    def test_mismatched_marker_counts_does_not_raise(self):
        """Comparisons between different-length stereo infos are skipped, not
        raised, so amend_configuration no longer abandons the molecule."""
        # One marker vs two markers in the group: not enantiomers -> True.
        result = no_enantiomer(
            "C[C@H](O)CC", ["C[C@H](O)CC", "C[C@@H](O)[C@@H](F)Cl"]
        )
        assert result is True

    def test_true_enantiomer_still_detected(self):
        """A genuine enantiomer (same length, all inverted) still returns False."""
        result = no_enantiomer("C[C@H](O)F", ["C[C@H](O)F", "C[C@@H](O)F"])
        assert result is False


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

    def test_enantiomer_empty_lists_returns_false(self):
        """Two molecules with no stereo centers are not an enantiomeric pair.

        This asserted ``True`` until 3.0.0, matching the implementation's
        vacuous result: the comparison loop never executed and ``indicator``
        kept its ``True`` initial value. A molecule with no stereo centers is
        its own mirror image, so it has no enantiomer to be paired with, and
        the old behavior made ``enantiomer_helper`` discard distinct achiral
        compounds -- including one geometric isomer of every unspecified C=C.
        """
        l1: list[tuple[int, str]] = []
        l2: list[tuple[int, str]] = []

        result = enantiomer(l1, l2)
        assert result is False


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
