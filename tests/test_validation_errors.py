"""Tests for validation error handling in Auto3D.utils.validation module.

These tests verify that validation functions raise real exceptions (not
AssertionError) for invalid input data, ensuring reliable validation even with
Python's -O flag. The .smi reader raises the structured InputValidationError so
the CLI's `except Auto3DError` path produces an actionable hint.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rdkit import Chem

from Auto3D.exceptions import InputValidationError
from Auto3D.utils.validation import check_sdf_format, check_smi_format


class TestCheckSmiFormatErrors:
    """Tests for error handling in check_smi_format function."""

    def test_missing_id_raises_input_validation_error(self):
        """Missing ID (single token per line) should raise InputValidationError."""
        # Create file with only SMILES, no ID
        content = "CCO\n"  # Only SMILES, no ID

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write(content)
            f.flush()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should raise the structured InputValidationError (not AssertionError)
            with pytest.raises(InputValidationError):
                check_smi_format(args)

        Path(f.name).unlink()

    def test_only_whitespace_id_raises_input_validation_error(self):
        """Line with only whitespace after SMILES should raise InputValidationError."""
        content = "CCO   \n"  # SMILES with trailing whitespace only, no ID

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write(content)
            f.flush()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # split() drops the trailing whitespace, leaving a single token (no ID)
            with pytest.raises(InputValidationError):
                check_smi_format(args)

        Path(f.name).unlink()

    def test_empty_line_skipped(self):
        """Empty/whitespace-only lines should be skipped without error."""
        content = "   \n\nCCO ethanol\n  \n"  # Empty lines + valid line

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write(content)
            f.flush()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should not raise - empty lines are skipped
            ani, only_aimnet = check_smi_format(args)
            assert isinstance(ani, bool)
            assert isinstance(only_aimnet, list)

        Path(f.name).unlink()

    def test_valid_smi_format_no_error(self):
        """Valid SMI format should not raise any error."""
        content = "CCO ethanol\nCCC propane\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write(content)
            f.flush()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should not raise
            ani, only_aimnet = check_smi_format(args)
            assert isinstance(ani, bool)
            assert isinstance(only_aimnet, list)

        Path(f.name).unlink()

    def test_multiple_tokens_accepted(self):
        """Lines with extra tokens after SMILES and ID should be accepted.

        The chunk loader reads only the first two whitespace columns
        (usecols=[0, 1]), so validation must tolerate trailing columns rather
        than crash on 'too many values to unpack'.
        """
        content = "CCO ethanol extra_info more_data\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write(content)
            f.flush()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should NOT raise: only the first two columns (SMILES, ID) are used.
            ani, only_aimnet = check_smi_format(args)
            assert isinstance(ani, bool)
            assert isinstance(only_aimnet, list)

        Path(f.name).unlink()


class TestCheckSdfFormatErrors:
    """Tests for error handling in check_sdf_format function."""

    def test_empty_molecule_id_raises_value_error(self):
        """Empty molecule ID should raise ValueError, not AssertionError."""
        # Create a simple SDF with empty _Name property
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            # Set empty name
            mol.SetProp("_Name", "")
            writer.write(mol)
            writer.close()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            with pytest.raises(ValueError, match="[Ee]mpty.*ID|[Ee]mpty.*_Name"):
                check_sdf_format(args)

        Path(f.name).unlink()

    def test_valid_sdf_format_no_error(self):
        """Valid SDF format should not raise any error."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            mol.SetProp("_Name", "ethanol")
            writer.write(mol)
            writer.close()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should not raise
            ani, only_aimnet = check_sdf_format(args)
            assert isinstance(ani, bool)
            assert isinstance(only_aimnet, list)

        Path(f.name).unlink()

    def test_multiple_valid_molecules(self):
        """SDF with multiple valid molecules should not raise."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.MolFromSmiles("CCC")
        mol2 = Chem.AddHs(mol2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            mol1.SetProp("_Name", "ethanol")
            mol2.SetProp("_Name", "propane")
            writer.write(mol1)
            writer.write(mol2)
            writer.close()

            args = MagicMock()
            args.path = f.name
            args.enumerate_isomer = False

            # Should not raise
            ani, only_aimnet = check_sdf_format(args)
            assert isinstance(ani, bool)
            assert isinstance(only_aimnet, list)

        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
