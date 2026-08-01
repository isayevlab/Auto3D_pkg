"""Tests for Auto3D CLI.

This module tests the CLI interface including:
- The _is_yaml_file helper function
- The new Typer CLI interface
- CLI module imports
- Exception handling
"""

import pytest
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from Auto3D.auto3Dcli import cli, _is_yaml_file
from Auto3D.exceptions import (
    Auto3DError,
    ConfigurationError,
    GPUError,
    DependencyError,
    FileFormatError,
    OptimizationError,
)


# =============================================================================
# Tests for _is_yaml_file helper function
# =============================================================================


def test_is_yaml_file_yaml_extension():
    """_is_yaml_file should detect .yaml files."""
    assert _is_yaml_file("config.yaml") is True


def test_is_yaml_file_yml_extension():
    """_is_yaml_file should detect .yml files."""
    assert _is_yaml_file("config.yml") is True


def test_is_yaml_file_help_flag():
    """_is_yaml_file should return False for --help flag."""
    assert _is_yaml_file("--help") is False


def test_is_yaml_file_smi_file():
    """_is_yaml_file should return False for .smi files."""
    assert _is_yaml_file("input.smi") is False


def test_is_yaml_file_short_flag():
    """_is_yaml_file should return False for short flags."""
    assert _is_yaml_file("-v") is False


def test_is_yaml_file_uppercase():
    """_is_yaml_file should handle uppercase extensions."""
    assert _is_yaml_file("CONFIG.YAML") is True
    assert _is_yaml_file("config.YML") is True


def test_is_yaml_file_with_path():
    """_is_yaml_file should handle paths with directories."""
    assert _is_yaml_file("/path/to/config.yaml") is True
    assert _is_yaml_file("./config.yml") is True
    assert _is_yaml_file("../configs/test.yaml") is True


# =============================================================================
# Tests for new Typer CLI interface
# =============================================================================


def test_new_cli_help():
    """New CLI should show help."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "config" in result.stdout
    assert "models" in result.stdout
    assert "validate" in result.stdout


def test_new_cli_version():
    """New CLI should show version."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app
    import Auto3D

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert Auto3D.__version__ in result.stdout


def test_run_subcommand_help():
    """run --help should show all options."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    # Strip ANSI: rich colorizes the help in a TTY/CI (FORCE_COLOR), styling the
    # '--' prefix separately from the option name, which splits the literal
    # '--engine' across escape codes. Stripping rejoins the text.
    import re
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert "--config" in out or "-c" in out
    assert "--engine" in out
    assert "--gpu" in out
    assert "--json" in out


def test_config_subcommand_help():
    """config --help should show subcommands."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["config", "--help"])

    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "show" in result.stdout
    assert "validate" in result.stdout


def test_models_subcommand_help():
    """models --help should show subcommands."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["models", "--help"])

    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "info" in result.stdout


def test_validate_subcommand_help():
    """validate --help should show options."""
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--help"])

    assert result.exit_code == 0


# =============================================================================
# Tests for CLI module imports
# =============================================================================


def test_cli_module_imports():
    """CLI module should export key components."""
    from Auto3D.cli import app, console, print_success, print_error, print_warning

    assert app is not None
    assert console is not None
    assert callable(print_success)
    assert callable(print_error)
    assert callable(print_warning)


def test_cli_module_app_is_typer():
    """CLI app should be a Typer instance."""
    from Auto3D.cli import app
    import typer

    assert isinstance(app, typer.Typer)


def test_cli_module_console_is_rich():
    """CLI console should be a Rich Console instance."""
    from Auto3D.cli import console
    from rich.console import Console

    assert isinstance(console, Console)


# =============================================================================
# Tests for CLI exception handling
# =============================================================================


class TestCLIExceptionHandling:
    """Test that CLI wraps exceptions properly."""

    # Legacy YAML path now (a) checks the file exists and (b) maps exception
    # types to differentiated exit codes (see Auto3D.cli.errors.EXIT_CODES):
    # ConfigurationError -> 2, GPUError -> 4, generic Auto3DError/Optimization -> 1.
    #
    # optimizing_engine is 'ANI2xt' (a built-in name), not 'AIMNET', because
    # this legacy path now runs through CLIConfig (Task 1, C10/M27 parity),
    # whose engine validator resolves an aimnet-family name against the real
    # `aimnet` package registry. 'ANI2xt' short-circuits that resolution
    # (Auto3D.models.preflight.resolve_engine_name's first branch) with no
    # import of `aimnet` at all, keeping this exception-mapping test from
    # depending on that heavy optional dependency importing cleanly.
    _LEGACY_YAML = {
        'path': '/fake/path.smi', 'k': 1, 'window': None, 'memory': None,
        'capacity': 40, 'enumerate_tautomer': False, 'tauto_engine': 'rdkit',
        'pKaNorm': True, 'isomer_engine': 'rdkit', 'max_confs': None,
        'enumerate_isomer': True, 'mode_oe': 'classic', 'mpi_np': 4,
        'optimizing_engine': 'ANI2xt', 'use_gpu': False, 'gpu_idx': 0,
        'opt_steps': 2000, 'convergence_threshold': 0.01, 'patience': 250,
        'threshold': 0.3, 'verbose': False, 'job_name': '',
    }

    def _run_legacy_with_error(self, error):
        """Drive the legacy YAML path with main() raising `error`; return exit code."""
        with patch('Auto3D.auto3D.main') as mock_main:
            mock_main.side_effect = error
            with patch.object(sys, 'argv', ['auto3d', 'config.yaml']):
                with patch('yaml.safe_load', return_value=dict(self._LEGACY_YAML)):
                    with patch('builtins.open', MagicMock()), \
                         patch('pathlib.Path.is_file', return_value=True):
                        with pytest.raises(SystemExit) as exc_info:
                            cli()
                        return exc_info.value.code

    def test_cli_configuration_error_exits_2(self):
        """ConfigurationError -> exit code 2 (legacy YAML mode)."""
        assert self._run_legacy_with_error(ConfigurationError("bad config")) == 2

    def test_cli_gpu_error_exits_4(self):
        """GPUError -> exit code 4 (legacy YAML mode)."""
        assert self._run_legacy_with_error(GPUError("No CUDA device available")) == 4

    def test_cli_optimization_error_exits_1(self):
        """OptimizationError -> generic exit code 1 (legacy YAML mode)."""
        assert self._run_legacy_with_error(OptimizationError("No structures converged")) == 1


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
