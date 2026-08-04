"""Tests for Auto3D.utils.validation module."""
import os
import warnings

import pytest
from rdkit import Chem

from Auto3D.config import Auto3DOptions
from Auto3D.utils.chemistry import check_connectivity, filter_unique
from Auto3D.utils.validation import (
    check_input,
    check_sdf_format,
    check_smi_format,
    check_valid_configuration,
)

# Set up test file paths
folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_all_stereo = os.path.join(folder, "tests/files/all_stereo_centers_specified.smi")
path_unspecified = os.path.join(folder, "tests/files/contain_unspecified_centers.smi")
path_connectivity = os.path.join(folder, "tests/files/22057.sdf")
path_example_sdf = os.path.join(folder, "tests/files/example.sdf")
path_example_smi = os.path.join(folder, "tests/files/example.smi")


class TestCheckInput:
    """Tests for check_input function."""

    def test_check_input_all_specified_enumerate_true(self):
        """Test with all stereo centers specified and enumerate_isomer=True."""
        args = Auto3DOptions(path_all_stereo, k=1, enumerate_isomer=True, use_gpu=False)
        args["input_format"] = "smi"
        with warnings.catch_warnings(record=True) as warnings_list:
            check_input(args)
        assert len(warnings_list) == 0

    def test_check_input_unspecified_enumerate_true(self):
        """Test with unspecified centers and enumerate_isomer=True (no warnings)."""
        args = Auto3DOptions(path_unspecified, k=1, enumerate_isomer=True, use_gpu=False)
        args["input_format"] = "smi"
        with warnings.catch_warnings(record=True) as warnings_list:
            check_input(args)
        assert len(warnings_list) == 0

    def test_check_input_unspecified_enumerate_false_warns(self):
        """Test with unspecified centers and enumerate_isomer=False (should warn)."""
        args = Auto3DOptions(path_unspecified, k=1, use_gpu=False, enumerate_isomer=False)
        args["input_format"] = "smi"
        with warnings.catch_warnings(record=True) as warnings_list:
            warnings.simplefilter("always")
            check_input(args)
        assert len(warnings_list) >= 1

    def test_check_input_all_specified_enumerate_false(self):
        """Test with all stereo centers specified and enumerate_isomer=False (no warnings)."""
        args = Auto3DOptions(path_all_stereo, k=1, use_gpu=False, enumerate_isomer=False)
        args["input_format"] = "smi"
        with warnings.catch_warnings(record=True) as warnings_list:
            check_input(args)
        assert len(warnings_list) == 0


class TestCheckSmiFormat:
    """Tests for check_smi_format function."""

    def test_check_smi_format_returns_tuple(self):
        """Test that check_smi_format returns a tuple of (bool, list)."""
        args = Auto3DOptions(path_all_stereo, k=1, enumerate_isomer=True, use_gpu=False)
        result = check_smi_format(args)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_check_smi_format_ani_compatible(self):
        """Test that check_smi_format correctly identifies ANI-compatible molecules."""
        args = Auto3DOptions(path_example_smi, k=1, enumerate_isomer=True, use_gpu=False)
        ani_compatible, only_aimnet = check_smi_format(args)
        # Should be ANI compatible (organic molecules with H, C, N, O, F, S, Cl)
        assert ani_compatible is True
        assert len(only_aimnet) == 0

    def test_check_smi_format_tolerates_extra_columns(self, tmp_path):
        """Lines with >2 whitespace columns must be accepted.

        The chunk loader reads only the first two columns (usecols=[0, 1]), so
        a trailing comment column must not make validation reject input the rest
        of the pipeline happily ingests (previously raised 'too many values to
        unpack').
        """
        smi = tmp_path / "ragged.smi"
        smi.write_text("CCO ethanol inline_comment_column\nCCN amine\n")
        args = Auto3DOptions(str(smi), k=1, enumerate_isomer=True, use_gpu=False)
        ani, _ = check_smi_format(args)
        assert ani is True

    def test_check_smi_format_rejects_missing_id(self, tmp_path):
        """A non-blank line with only a SMILES (no ID) raises InputValidationError."""
        from Auto3D.exceptions import InputValidationError
        smi = tmp_path / "noid.smi"
        smi.write_text("CCO\n")
        args = Auto3DOptions(str(smi), k=1, enumerate_isomer=True, use_gpu=False)
        with pytest.raises(InputValidationError):
            check_smi_format(args)


class TestCheckSdfFormat:
    """Tests for check_sdf_format function."""

    def test_check_sdf_format_returns_tuple(self):
        """Test that check_sdf_format returns a tuple of (bool, list)."""
        args = Auto3DOptions(path_example_sdf, k=1, enumerate_isomer=False, use_gpu=False)
        result = check_sdf_format(args)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_check_sdf_format_warns_enumerate_isomer(self):
        """Test that check_sdf_format warns when enumerate_isomer=True for SDF."""
        args = Auto3DOptions(path_example_sdf, k=1, enumerate_isomer=True, use_gpu=False)
        with warnings.catch_warnings(record=True) as warnings_list:
            warnings.simplefilter("always")
            check_sdf_format(args)
        assert len(warnings_list) >= 1


class TestCheckConnectivity:
    """Tests for check_connectivity function."""

    def test_check_connectivity_broken_bond(self):
        """Test that check_connectivity detects broken bonds."""
        supp = Chem.SDMolSupplier(path_connectivity, removeHs=False)
        mol1 = supp[0]  # Molecule with broken bond
        assert check_connectivity(mol1) is False

    def test_check_connectivity_valid(self):
        """Test that check_connectivity accepts valid molecules."""
        supp = Chem.SDMolSupplier(path_connectivity, removeHs=False)
        mol2 = supp[1]  # Valid molecule
        assert check_connectivity(mol2) is True

    def test_check_connectivity_example_sdf(self):
        """Test check_connectivity with example.sdf molecules."""
        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        for mol in supp:
            if mol is not None:
                # All molecules in example.sdf should have valid connectivity
                assert check_connectivity(mol) is True


class TestFilterUnique:
    """Tests for filter_unique function."""

    def test_filter_unique_removes_unconverged(self):
        """Test that filter_unique removes unconverged structures."""
        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        mols = [mol for mol in supp if mol is not None]

        # Set all mols as unconverged
        for mol in mols:
            mol.SetProp("Converged", "False")

        result = filter_unique(mols)
        assert len(result) == 0

    def test_filter_unique_keeps_converged(self):
        """Test that filter_unique keeps converged structures."""
        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        mols = [mol for mol in supp if mol is not None]

        # Set all mols as converged
        for mol in mols:
            mol.SetProp("Converged", "True")

        result = filter_unique(mols)
        # Should keep at least one unique structure
        assert len(result) >= 1

    def test_filter_unique_removes_duplicates(self):
        """Test that filter_unique removes similar structures."""
        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        mols = [mol for mol in supp if mol is not None]

        if len(mols) > 0:
            # Duplicate the first molecule
            mol = mols[0]
            mol.SetProp("Converged", "True")
            duplicate = Chem.RWMol(mol)
            duplicate.SetProp("Converged", "True")

            result = filter_unique([mol, duplicate], crit=0.3)
            # Only one should remain
            assert len(result) == 1

    def test_filter_unique_keeps_records_with_no_converged_property(self):
        """Absence of the property is not a failed optimization.

        Only ``batchopt`` writes ``Converged``; an ``opt_geometry`` output, an
        ORCA/Gaussian export or a hand-built conformer set carries none.
        Treating that as "did not converge" deleted every record. The
        consequence asserted here is that such a file filters exactly the same
        as one whose records all say Converged=True.
        """
        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        flagged = [mol for mol in supp if mol is not None]
        for mol in flagged:
            mol.SetProp("Converged", "True")
        expected = len(filter_unique(flagged))
        assert expected >= 1, "test premise: the flagged file must keep something"

        supp = Chem.SDMolSupplier(path_example_sdf, removeHs=False)
        unflagged = [mol for mol in supp if mol is not None]
        for mol in unflagged:
            mol.ClearProp("Converged")
            assert not mol.HasProp("Converged")

        result = filter_unique(unflagged)
        assert len(result) == expected, (
            f"{len(unflagged)} record(s) with no 'Converged' property kept "
            f"{len(result)}, but the same records marked Converged=True keep "
            f"{expected}"
        )

    def test_filter_unique_custom_threshold(self):
        """A tighter RMSD threshold must keep MORE structures than a looser
        one -- not merely "at least as many", which passes even when the two
        thresholds produce identical results (as the fixture in
        ``path_example_sdf`` does: two molecules of different sizes, so
        ``species_key`` alone already keeps both regardless of ``crit``).

        Constructs two conformers of ONE molecule whose RMSD sits strictly
        between the two thresholds, so equality cannot pass silently.
        """
        mol1 = Chem.AddHs(Chem.MolFromSmiles("CCCCCCCC"))  # octane: flexible
        mol2 = Chem.Mol(mol1)
        from rdkit.Chem import AllChem, rdMolAlign
        AllChem.EmbedMolecule(mol1, randomSeed=1)
        AllChem.EmbedMolecule(mol2, randomSeed=99)
        AllChem.MMFFOptimizeMolecule(mol1)
        AllChem.MMFFOptimizeMolecule(mol2)
        mol1.SetProp("Converged", "True")
        mol2.SetProp("Converged", "True")

        rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(mol1), Chem.RemoveHs(mol2))
        assert rmsd > 0.05, "test premise: conformers must be geometrically distinct"

        crit_strict = rmsd / 2   # below the actual RMSD -> kept separate
        crit_lenient = rmsd * 2  # above the actual RMSD -> merged

        result_strict = filter_unique([mol1, mol2], crit=crit_strict)
        result_lenient = filter_unique([mol1, mol2], crit=crit_lenient)

        assert len(result_strict) == 2, "strict threshold must not merge distinct conformers"
        assert len(result_lenient) == 1, "lenient threshold must merge near-identical conformers"
        assert len(result_strict) > len(result_lenient)


class TestCheckValidConfiguration:
    """Tests for check_valid_configuration function."""

    def test_valid_configuration(self):
        """Test that valid configuration returns no errors."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            optimizing_engine="AIMNET",
            isomer_engine="rdkit",
            opt_steps=5000,
        )
        assert len(errors) == 0

    def test_missing_path(self):
        """Test that missing path returns error."""
        errors = check_valid_configuration(
            path=None,
            k=1,
            use_gpu=False,
        )
        assert any("path" in e.lower() for e in errors)

    def test_nonexistent_path(self):
        """Test that nonexistent path returns error."""
        errors = check_valid_configuration(
            path="/nonexistent/path.smi",
            k=1,
            use_gpu=False,
        )
        assert any("exist" in e.lower() for e in errors)

    def test_missing_k_and_window(self):
        """Test that missing both k and window returns error."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=False,
            window=False,
            use_gpu=False,
        )
        assert any("k" in e.lower() or "window" in e.lower() for e in errors)

    def test_window_specified(self):
        """Test that window alone is sufficient."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=False,
            window=5.0,
            use_gpu=False,
        )
        assert not any("k" in e.lower() or "window" in e.lower() for e in errors)

    def test_invalid_optimizing_engine(self):
        """Test that invalid optimizing_engine returns error."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            optimizing_engine="INVALID",
        )
        assert any("optimizing_engine" in e.lower() for e in errors)

    def test_accepts_aimnet_registry_names(self):
        """Registry engine names must validate, matching model_factory/CLI schema."""
        for name in ("aimnet2", "aimnet2-2025", "aimnet2-nse", "aimnet2-pd"):
            errors = check_valid_configuration(
                path=path_example_smi,
                k=1,
                use_gpu=False,
                optimizing_engine=name,
            )
            assert not any("optimizing_engine" in e.lower() for e in errors), (name, errors)

    def test_invalid_isomer_engine(self):
        """Test that invalid isomer_engine returns error."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            isomer_engine="invalid",
        )
        assert any("isomer_engine" in e.lower() for e in errors)

    def test_opt_steps_too_small(self):
        """Test that opt_steps < 10 returns error."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            opt_steps=5,
        )
        assert any("opt_steps" in e.lower() for e in errors)

    def test_invalid_tauto_engine(self):
        """Test that invalid tauto_engine with enumerate_tautomer=True returns error."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            enumerate_tautomer=True,
            tauto_engine="invalid",
        )
        assert any("tauto_engine" in e.lower() for e in errors)

    def test_valid_tauto_configuration(self):
        """Test valid tautomer configuration."""
        errors = check_valid_configuration(
            path=path_example_smi,
            k=1,
            use_gpu=False,
            enumerate_tautomer=True,
            tauto_engine="rdkit",
        )
        # Should not have tauto_engine errors
        assert not any("tauto_engine" in e.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
