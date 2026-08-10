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
    """New CLI should show help.

    A bare ``"run" in result.stdout`` also matches "running" inside
    ``validate``'s own description ("...without running optimization."), so
    deleting the entire ``run`` command would still leave this green. Anchor
    on the word boundary so only the standalone command name counts.
    """
    import re

    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    assert result.exit_code == 0
    assert re.search(r"\brun\b", out), "the 'run' command must be listed"
    assert "config" in out
    assert "models" in out
    assert "validate" in out


def test_new_cli_version():
    """New CLI should show the actual installed package version.

    Comparing against ``Auto3D.__version__`` -- the very attribute the CLI's
    own ``version_callback`` prints -- is tautological: hardcoding that
    attribute to anything would still make the CLI and this test agree.
    ``importlib.metadata.version("Auto3D")`` reads the installed
    distribution's metadata directly, independent of whatever
    ``Auto3D/__init__.py`` happens to expose.
    """
    from importlib.metadata import version as installed_version

    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert installed_version("Auto3D") in result.stdout


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
    # "-c" alone is trivially satisfied by "--max-confs" (which contains the
    # substring "-c") even if the -c/--config option were deleted outright.
    # "--config" is not a substring of any other option, so anchor on it.
    assert "--config" in out
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
    """models --help should show subcommands, and each one must be real.

    ``"info" in result.stdout`` is also satisfied by the group's own help
    text ("Neural network model **info**rmation."), so deleting the ``info``
    subcommand entirely would still leave this green. Actually invoking each
    subcommand's own ``--help`` proves it exists as a real command, not just
    that its name appears somewhere in the parent's help text.
    """
    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0

    for sub in ("list", "info", "test"):
        sub_result = runner.invoke(app, ["models", sub, "--help"])
        assert sub_result.exit_code == 0, (
            f"'models {sub} --help' exited {sub_result.exit_code}; the "
            f"group's --help text alone cannot prove the subcommand exists"
        )


def test_validate_subcommand_help():
    """validate --help should show its actual options, not just exit 0."""
    import re

    from typer.testing import CliRunner
    from Auto3D.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--help"])
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)

    assert result.exit_code == 0
    assert "--json" in out
    assert "--verbose" in out


# =============================================================================
# Tests for CLI module imports
#
# Each name is imported from the module that defines it. `Auto3D.cli` used to
# re-export all five; it re-exports nothing now, and `app`/`console` there are
# the *modules*, not the Typer application and the Rich console. See
# tests/test_import_boundaries.py::test_cli_app_and_console_names_resolve_to_their_modules.
# =============================================================================


def test_cli_module_imports():
    """The CLI's key components exist at their defining module paths."""
    from Auto3D.cli.app import app
    from Auto3D.cli.console import console, print_error, print_success, print_warning

    assert app is not None
    assert console is not None
    assert callable(print_success)
    assert callable(print_error)
    assert callable(print_warning)


def test_cli_module_app_is_typer():
    """CLI app should be a Typer instance."""
    import typer

    from Auto3D.cli.app import app

    assert isinstance(app, typer.Typer)


def test_cli_module_console_is_rich():
    """CLI console should be a Rich Console instance."""
    from rich.console import Console

    from Auto3D.cli.console import console

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
        "path": "/fake/path.smi",
        "k": 1,
        "window": None,
        "memory": None,
        "capacity": 40,
        "enumerate_tautomer": False,
        "tauto_engine": "rdkit",
        "pKaNorm": True,
        "isomer_engine": "rdkit",
        "max_confs": None,
        "enumerate_isomer": True,
        "mode_oe": "classic",
        "mpi_np": 4,
        "optimizing_engine": "ANI2xt",
        "use_gpu": False,
        "gpu_idx": 0,
        "opt_steps": 2000,
        "convergence_threshold": 0.01,
        "patience": 250,
        "threshold": 0.3,
        "verbose": False,
        "job_name": "",
    }

    def _run_legacy_with_error(self, error):
        """Drive the legacy YAML path with main() raising `error`; return exit code."""
        with patch("Auto3D.auto3D.main") as mock_main:
            mock_main.side_effect = error
            with patch.object(sys, "argv", ["auto3d", "config.yaml"]):
                with patch("yaml.safe_load", return_value=dict(self._LEGACY_YAML)):
                    with (
                        patch("builtins.open", MagicMock()),
                        patch("pathlib.Path.is_file", return_value=True),
                    ):
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


def _squash(text):
    """Strip ANSI and box-drawing characters, then remove *all* whitespace.

    Every part earns its place against a real way these assertions go wrong:

    * ANSI -- Rich styles adjacent cells separately (``[red]1[/red] failed``),
      so an escape sequence lands between the two words being looked for. It
      colorizes whenever it thinks it is on a terminal, which ``FORCE_COLOR``
      makes true in CI and false locally.
    * Box drawing -- Rich folds a long token at the panel width, so a job
      directory renders as ``...inputs_kes │`` / ``│ trel`` and the border
      character sits *inside* the word.
    * Whitespace -- removing it (rather than collapsing it) rejoins that fold.
    """
    import re

    plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    return "".join(re.sub(r"[─-╿]", "", plain).split())


class TestLegacyYamlReconciliation:
    """`auto3d params.yaml` must report lost molecules, like `auto3d run` does.

    The reconciliation data (``WorkflowResult.failures``) has been correct since
    Phase 4 and ``execute_run`` has printed it and exited 6 since then. This
    entry point printed an unconditional green tick and returned 0 on the very
    same result -- so the two supported ways of running the same config
    disagreed about whether the run had succeeded.

    ``main`` is stubbed in every test here: no potential is loaded and no model
    is downloaded. ``optimizing_engine`` is ``ANI2xt`` because that name
    short-circuits engine resolution without importing ``aimnet``.
    """

    @staticmethod
    def _write_yaml(tmp_path, **overrides):
        import yaml as yaml_mod

        smi = tmp_path / "inputs.smi"
        smi.write_text("CCO alpha\nCCC beta\n")
        params = {
            "path": str(smi),
            "k": 1,
            "window": "None",
            "optimizing_engine": "ANI2xt",
            "use_gpu": False,
            "gpu_idx": 0,
            "verbose": False,
        }
        params.update(overrides)
        yaml_path = tmp_path / "params.yaml"
        yaml_path.write_text(yaml_mod.dump(params))
        return yaml_path

    @staticmethod
    def _drive(tmp_path, monkeypatch, fake_main, **yaml_overrides):
        """Run the real `cli()` in legacy mode with `main` replaced."""
        import Auto3D.auto3D as a3d

        yaml_path = TestLegacyYamlReconciliation._write_yaml(tmp_path, **yaml_overrides)
        monkeypatch.setattr(a3d, "main", fake_main)
        monkeypatch.setattr(sys, "argv", ["auto3d", str(yaml_path)])
        cli()

    def test_lost_molecules_are_named_in_the_output_and_exit_partial_success(
        self, tmp_path, monkeypatch, capsys
    ):
        """The failure *list* has to reach the user, not just a non-zero code.

        Asserting the integer alone would prove very little here: several guards
        in this CLI exit with the same code, and an earlier exit-code-only test
        on this branch passed with its fix reverted. The molecule name is what
        only the reconciliation path can put on screen, and "chlorpromazine"
        cannot appear incidentally -- notably not in ``tmp_path``, which pytest
        names after this test function.
        """
        from Auto3D.cli.commands.run import EXIT_PARTIAL_SUCCESS
        from Auto3D.results import WorkflowResult

        # A path that is never created: count_output reports (0, 0) for it, so
        # this test is about the failure list and not about parsing an SDF.
        missing_output = tmp_path / "combined.sdf"

        with pytest.raises(SystemExit) as exc_info:
            self._drive(
                tmp_path,
                monkeypatch,
                lambda options, **kw: WorkflowResult(
                    str(missing_output), failures=["chlorpromazine"]
                ),
            )

        assert exc_info.value.code == EXIT_PARTIAL_SUCCESS
        out = _squash(capsys.readouterr().out)
        assert "chlorpromazine" in out, "the lost molecule was never named to the user"
        assert "1failed" in out

    def test_a_clean_run_still_reports_success_and_exits_zero(self, tmp_path, monkeypatch, capsys):
        """Control: the same wiring with nothing missing must not exit at all.

        Without this, the test above would also pass if the new code exited 6
        unconditionally.
        """
        from Auto3D.results import WorkflowResult

        self._drive(
            tmp_path,
            monkeypatch,
            lambda options, **kw: WorkflowResult(str(tmp_path / "combined.sdf"), failures=[]),
        )

        out = _squash(capsys.readouterr().out)
        assert "Results" in out
        assert "failed" not in out

    def test_ctrl_c_reports_what_is_known_instead_of_nothing(self, tmp_path, monkeypatch, capsys):
        """KeyboardInterrupt is a BaseException, so the blanket `except
        Exception` on this path could never see it: Ctrl-C printed nothing at
        all and the user was left guessing whether anything reached disk.

        Keyed on the job directory, not on the exit code: ``job_name: kestrel``
        is echoed back only by a handler that actually read the configuration.
        """
        from Auto3D.cli.errors import EXIT_INTERRUPTED

        def interrupted_main(options, **kwargs):
            raise KeyboardInterrupt

        with pytest.raises(SystemExit) as exc_info:
            self._drive(tmp_path, monkeypatch, interrupted_main, job_name="kestrel")

        assert exc_info.value.code == EXIT_INTERRUPTED
        err = _squash(capsys.readouterr().err)
        assert "Interruptedbytheuser" in err
        assert "inputs_kestrel" in err, "the interrupt report never named the job directory"


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
