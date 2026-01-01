"""Tests for CLI exception handling.

This module tests that the CLI properly handles Auto3DError exceptions
and returns appropriate exit codes.
"""

import pytest
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

from Auto3D.auto3Dcli import cli
from Auto3D.exceptions import (
    Auto3DError,
    ConfigurationError,
    GPUError,
    DependencyError,
    FileFormatError,
    OptimizationError,
)


class TestCLIExceptionHandling:
    """Test that CLI wraps exceptions properly."""

    def test_cli_catches_auto3d_error_and_exits_with_code_1(self):
        """CLI should catch Auto3DError and exit with code 1."""
        # Mock main() to raise Auto3DError
        with patch('Auto3D.auto3Dcli.main') as mock_main:
            mock_main.side_effect = ConfigurationError("Test configuration error")

            # Mock sys.argv to provide minimal arguments
            with patch.object(sys, 'argv', ['auto3d', 'config.yaml']):
                # Mock yaml.load to return valid config
                with patch('Auto3D.auto3Dcli.yaml.load', return_value={
                    'path': '/fake/path.smi',
                    'k': 1,
                    'window': None,
                    'memory': None,
                    'capacity': 40,
                    'enumerate_tautomer': False,
                    'tauto_engine': 'rdkit',
                    'pKaNorm': True,
                    'isomer_engine': 'rdkit',
                    'max_confs': None,
                    'enumerate_isomer': True,
                    'mode_oe': 'classic',
                    'mpi_np': 4,
                    'optimizing_engine': 'AIMNET',
                    'use_gpu': False,
                    'gpu_idx': 0,
                    'opt_steps': 2000,
                    'convergence_threshold': 0.01,
                    'patience': 250,
                    'threshold': 0.3,
                    'verbose': False,
                    'job_name': '',
                }):
                    with patch('builtins.open', MagicMock()):
                        # Capture stderr
                        captured_stderr = StringIO()
                        with patch.object(sys, 'stderr', captured_stderr):
                            with pytest.raises(SystemExit) as exc_info:
                                cli()

                            # Should exit with code 1
                            assert exc_info.value.code == 1

                            # Should print error message to stderr
                            stderr_output = captured_stderr.getvalue()
                            assert "Error:" in stderr_output
                            assert "Test configuration error" in stderr_output

    def test_cli_catches_gpu_error_and_exits_with_code_1(self):
        """CLI should catch GPUError and exit with code 1."""
        with patch('Auto3D.auto3Dcli.main') as mock_main:
            mock_main.side_effect = GPUError("No CUDA device available")

            with patch.object(sys, 'argv', ['auto3d', 'config.yaml']):
                with patch('Auto3D.auto3Dcli.yaml.load', return_value={
                    'path': '/fake/path.smi',
                    'k': 1,
                    'window': None,
                    'memory': None,
                    'capacity': 40,
                    'enumerate_tautomer': False,
                    'tauto_engine': 'rdkit',
                    'pKaNorm': True,
                    'isomer_engine': 'rdkit',
                    'max_confs': None,
                    'enumerate_isomer': True,
                    'mode_oe': 'classic',
                    'mpi_np': 4,
                    'optimizing_engine': 'AIMNET',
                    'use_gpu': True,
                    'gpu_idx': 0,
                    'opt_steps': 2000,
                    'convergence_threshold': 0.01,
                    'patience': 250,
                    'threshold': 0.3,
                    'verbose': False,
                    'job_name': '',
                }):
                    with patch('builtins.open', MagicMock()):
                        captured_stderr = StringIO()
                        with patch.object(sys, 'stderr', captured_stderr):
                            with pytest.raises(SystemExit) as exc_info:
                                cli()

                            assert exc_info.value.code == 1
                            stderr_output = captured_stderr.getvalue()
                            assert "No CUDA device available" in stderr_output

    def test_cli_catches_optimization_error_and_exits_with_code_1(self):
        """CLI should catch OptimizationError and exit with code 1."""
        with patch('Auto3D.auto3Dcli.main') as mock_main:
            mock_main.side_effect = OptimizationError("No structures converged")

            with patch.object(sys, 'argv', ['auto3d', 'config.yaml']):
                with patch('Auto3D.auto3Dcli.yaml.load', return_value={
                    'path': '/fake/path.smi',
                    'k': 1,
                    'window': None,
                    'memory': None,
                    'capacity': 40,
                    'enumerate_tautomer': False,
                    'tauto_engine': 'rdkit',
                    'pKaNorm': True,
                    'isomer_engine': 'rdkit',
                    'max_confs': None,
                    'enumerate_isomer': True,
                    'mode_oe': 'classic',
                    'mpi_np': 4,
                    'optimizing_engine': 'AIMNET',
                    'use_gpu': False,
                    'gpu_idx': 0,
                    'opt_steps': 2000,
                    'convergence_threshold': 0.01,
                    'patience': 250,
                    'threshold': 0.3,
                    'verbose': False,
                    'job_name': '',
                }):
                    with patch('builtins.open', MagicMock()):
                        captured_stderr = StringIO()
                        with patch.object(sys, 'stderr', captured_stderr):
                            with pytest.raises(SystemExit) as exc_info:
                                cli()

                            assert exc_info.value.code == 1
                            stderr_output = captured_stderr.getvalue()
                            assert "No structures converged" in stderr_output


class TestSmiles2MolsExceptionHandling:
    """Test that smiles2mols raises ConfigurationError instead of sys.exit."""

    def test_smiles2mols_raises_configuration_error_without_k_or_window(self):
        """smiles2mols should raise ConfigurationError when neither k nor window specified."""
        from Auto3D.auto3D import smiles2mols
        from Auto3D.config import Auto3DOptions

        # Create options without k or window (both set to False/default)
        args = Auto3DOptions(
            path=None,  # Will be set internally
            k=False,
            window=False,
            use_gpu=False,
        )

        # Should raise ConfigurationError, not call sys.exit
        with pytest.raises(ConfigurationError, match="Either k or window"):
            smiles2mols(["CCO"], args)
