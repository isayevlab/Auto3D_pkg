"""Tests for validation module - exception-based error handling.

This module tests that check_input raises proper exceptions instead of sys.exit().
"""

import pytest
from unittest.mock import patch, MagicMock
from Auto3D.utils.validation import check_input
from Auto3D.exceptions import GPUError, DependencyError, ConfigurationError, ModelLoadError


class TestCheckInputExceptions:
    """Test that check_input raises exceptions instead of sys.exit."""

    def test_gpu_not_available_raises_gpu_error(self):
        """Should raise GPUError when GPU requested but not available."""
        args = MagicMock()
        args.use_gpu = True
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        with patch('Auto3D.utils.validation.torch.cuda.is_available', return_value=False):
            with pytest.raises(GPUError, match="No cuda device"):
                check_input(args)

    def test_omega_without_license_raises_dependency_error(self):
        """Should raise DependencyError when omega used without OE_LICENSE."""
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "omega"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(DependencyError, match="OE_LICENSE"):
                check_input(args)

    def test_omega_without_openeye_raises_dependency_error(self):
        """Should raise DependencyError when omega used but openeye not installed."""
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "omega"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        # Simulate OE_LICENSE present but openeye not installed
        with patch.dict('os.environ', {'OE_LICENSE': '/path/to/license'}):
            # Import of openeye should fail
            with patch.dict('sys.modules', {'openeye': None, 'openeye.oechem': None}):
                with patch('builtins.__import__', side_effect=ImportError("No module named 'openeye'")):
                    with pytest.raises(DependencyError, match="openeye"):
                        check_input(args)

    def test_ani2x_without_torchani_raises_dependency_error(self):
        """Should raise DependencyError when ANI2x used but torchani not installed."""
        import builtins
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "ANI2x"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        # Mock that torchani import fails
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'torchani':
                raise ImportError("No module named 'torchani'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(DependencyError, match="TorchANI"):
                check_input(args)

    def test_custom_nnp_load_failure_raises_model_load_error(self, tmp_path):
        """Should raise ModelLoadError when custom NNP cannot be loaded."""
        # Create a dummy file that exists but isn't a valid model
        fake_model_path = tmp_path / "fake_model.pt"
        fake_model_path.write_text("not a valid model")

        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = str(fake_model_path)
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        with pytest.raises(ModelLoadError, match="cannot be loaded"):
            check_input(args)

    def test_opt_steps_too_small_raises_configuration_error(self):
        """Should raise ConfigurationError when opt_steps < 10."""
        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "AIMNET"
        args.opt_steps = 5
        args.input_format = "smi"
        args.path = "/fake/path.smi"

        with pytest.raises(ConfigurationError, match="smaller than 10"):
            check_input(args)

    def test_only_aimnet_molecules_with_ani2x_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when molecules require AIMNET but ANI2x selected."""
        # Create a SMILES file with a charged molecule (requires AIMNET)
        smi_file = tmp_path / "charged.smi"
        smi_file.write_text("[NH4+] ammonium\n")

        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "ANI2x"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = str(smi_file)
        args.enumerate_isomer = False

        # ANI2x can't handle charged molecules, only AIMNET can
        with pytest.raises(ConfigurationError, match="Only AIMNET can handle"):
            check_input(args)

    def test_only_aimnet_molecules_with_ani2xt_raises_configuration_error(self, tmp_path):
        """Should raise ConfigurationError when molecules require AIMNET but ANI2xt selected."""
        # Create a SMILES file with a molecule containing non-ANI element (e.g., Br)
        smi_file = tmp_path / "bromine.smi"
        smi_file.write_text("CBr methyl_bromide\n")

        args = MagicMock()
        args.use_gpu = False
        args.isomer_engine = "rdkit"
        args.optimizing_engine = "ANI2xt"
        args.opt_steps = 100
        args.input_format = "smi"
        args.path = str(smi_file)
        args.enumerate_isomer = False

        # ANI2xt can't handle Br, only AIMNET can
        with pytest.raises(ConfigurationError, match="Only AIMNET can handle"):
            check_input(args)

    def test_check_input_accepts_registry_engine(self, tmp_path):
        """A registry engine name (not a path, not ANI) must pass validation.

        Registry names like "aimnet2-2025" are not file paths, so the custom-path
        torch.jit.load block must be skipped, and the ANI-element gate must not
        reject them (they are AIMNet engines, not ANI2x/ANI2xt).
        """
        from types import SimpleNamespace
        from Auto3D.utils.validation import check_input
        smi = tmp_path / "in.smi"
        smi.write_text("CCO mol1\n")
        args = SimpleNamespace(
            path=str(smi), input_format="smi", optimizing_engine="aimnet2-2025",
            enumerate_isomer=True, opt_steps=10, k=1, window=False, use_gpu=False,
            isomer_engine="rdkit", verbose=False,
        )
        check_input(args)  # must not raise for a valid registry engine
