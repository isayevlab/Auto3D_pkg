"""Tests for Auto3D.utils.validation module."""

import os
import warnings

import pytest
from rdkit import Chem

from Auto3D.config import Auto3DOptions
from Auto3D.utils.connectivity import check_connectivity
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


def _options(**overrides):
    """Build the ``Auto3DOptions`` ``check_valid_configuration`` now takes.

    The function used to take ten keyword arguments mirroring
    ``Auto3DOptions``'s field names *and carrying their own defaults* -- a third
    configuration schema. It now takes the object, so these tests hand it one.
    ``use_gpu=False`` by default: this box's CUDA availability must not decide
    whether an assertion about paths or engines holds.
    """
    from Auto3D.config import Auto3DOptions

    params = {"path": path_example_smi, "k": 1, "use_gpu": False}
    params.update(overrides)
    return Auto3DOptions(**params)


class TestCheckValidConfiguration:
    """Tests for check_valid_configuration function."""

    def test_valid_configuration(self):
        """Test that valid configuration returns no errors."""
        errors = check_valid_configuration(
            _options(optimizing_engine="AIMNET", isomer_engine="rdkit", opt_steps=5000)
        )
        assert len(errors) == 0

    def test_missing_path(self):
        """Test that missing path returns error."""
        errors = check_valid_configuration(_options(path=None))
        assert any("path" in e.lower() for e in errors)

    def test_nonexistent_path(self):
        """Test that nonexistent path returns error."""
        errors = check_valid_configuration(_options(path="/nonexistent/path.smi"))
        assert any("exist" in e.lower() for e in errors)

    def test_missing_k_and_window(self):
        """Test that missing both k and window returns error."""
        errors = check_valid_configuration(_options(k=False, window=False))
        assert any("k" in e.lower() or "window" in e.lower() for e in errors)

    def test_window_specified(self):
        """Test that window alone is sufficient."""
        errors = check_valid_configuration(_options(k=False, window=5.0))
        assert not any("k" in e.lower() or "window" in e.lower() for e in errors)

    def test_invalid_optimizing_engine(self):
        """Test that invalid optimizing_engine returns error.

        Still checked here, unlike isomer_engine/tauto_engine below: an engine
        name may be a registry entry or a path to a custom model, so it is not
        an enumerable choice ``Auto3DOptions`` could validate from the value
        alone -- it needs the registry lookup.
        """
        errors = check_valid_configuration(_options(optimizing_engine="INVALID"))
        assert any("optimizing_engine" in e.lower() for e in errors)

    def test_accepts_aimnet_registry_names(self):
        """Registry engine names must validate, matching model_factory/CLI schema."""
        for name in ("aimnet2", "aimnet2-2025", "aimnet2-nse", "aimnet2-pd"):
            errors = check_valid_configuration(_options(optimizing_engine=name))
            assert not any("optimizing_engine" in e.lower() for e in errors), (name, errors)

    def test_invalid_isomer_engine_refused_at_construction(self):
        """An unrecognized isomer_engine is refused before this function runs.

        The whitelist moved to ``Auto3D.config.ENGINE_CHOICES`` and is enforced
        by ``Auto3DOptions.__post_init__``, so ``check_valid_configuration`` can
        no longer be reached with a bad value -- which is why it no longer
        carries its own copy of the set. The rejection did not disappear; it
        moved earlier, and to every entry point at once.
        """
        from Auto3D.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="isomer_engine"):
            _options(isomer_engine="invalid")

    def test_opt_steps_too_small_refused_at_construction(self):
        """opt_steps < 10 is refused at construction, not at run start.

        ``FIELD_BOUNDS["opt_steps"]`` is now ``("ge", 10)`` -- the single
        declaration of that floor -- so the two hand-written ``< 10`` checks
        that used to live in this module are gone. Same ``ConfigurationError``,
        raised before the banner instead of after it.
        """
        from Auto3D.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="opt_steps"):
            _options(opt_steps=5)

    def test_invalid_tauto_engine_refused_at_construction(self):
        """An unrecognized tauto_engine is refused at construction too.

        Unconditionally, where the old check only looked when
        ``enumerate_tautomer`` was true -- ``CLIConfig``'s
        ``Literal["rdkit", "oechem"]`` always rejected it, so the gated check
        was an entry-point divergence.
        """
        from Auto3D.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="tauto_engine"):
            _options(enumerate_tautomer=True, tauto_engine="invalid")
        with pytest.raises(ConfigurationError, match="tauto_engine"):
            _options(enumerate_tautomer=False, tauto_engine="invalid")

    def test_valid_tauto_configuration(self):
        """Test valid tautomer configuration."""
        errors = check_valid_configuration(_options(enumerate_tautomer=True, tauto_engine="rdkit"))
        # Should not have tauto_engine errors
        assert not any("tauto_engine" in e.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
