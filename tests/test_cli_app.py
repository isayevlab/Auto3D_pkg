# tests/test_cli_app.py
"""Tests for main Typer application."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_app_exists():
    """Main app should exist."""
    from Auto3D.cli.app import app
    assert app is not None


def test_help_works(runner):
    """--help should show available commands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "config" in result.stdout
    assert "models" in result.stdout


def test_version_works(runner):
    """--version should show version."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_run_help(runner):
    """run --help should show run options."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout or "-c" in result.stdout


def test_config_help(runner):
    """config --help should show subcommands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout


def test_models_help(runner):
    """models --help should show subcommands."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout


def test_models_list_shows_engines(runner):
    """models list should show available engines."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "AIMNET" in result.stdout


def test_models_info_aimnet(runner):
    """models info should show engine details."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "AIMNET"])
    assert result.exit_code == 0
    assert "AIMNet2" in result.stdout


def test_models_info_ani2x(runner):
    """models info should show ANI2x details."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "ANI2x"])
    assert result.exit_code == 0
    assert "ANI-2x" in result.stdout


def test_models_info_unknown_engine(runner):
    """models info should fail for unknown engine."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["models", "info", "UNKNOWN"])
    assert result.exit_code == 1


def test_validate_valid_smi(runner, tmp_path):
    """validate should pass for valid SMILES file."""
    from Auto3D.cli.app import app

    smi_file = tmp_path / "test.smi"
    smi_file.write_text("CCO ethanol\nCC(=O)O acetic_acid\n")

    result = runner.invoke(app, ["validate", str(smi_file)])
    assert result.exit_code == 0
    assert "Valid" in result.stdout or "valid" in result.stdout.lower()


def test_validate_invalid_smi(runner, tmp_path):
    """validate should fail for invalid SMILES."""
    from Auto3D.cli.app import app

    smi_file = tmp_path / "test.smi"
    smi_file.write_text("INVALID_SMILES mol1\n")

    result = runner.invoke(app, ["validate", str(smi_file)])
    assert result.exit_code == 1


@pytest.fixture
def tmp_path_cwd(tmp_path, monkeypatch):
    """Change to tmp_path and return it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_config_init_creates_file(runner, tmp_path_cwd):
    """config init should create YAML file."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-o", "test.yaml"])

    assert result.exit_code == 0
    assert (tmp_path_cwd / "test.yaml").exists()


def test_config_init_with_preset(runner, tmp_path_cwd):
    """config init with preset should use preset values."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-o", "test.yaml", "-p", "quick"])

    assert result.exit_code == 0
    content = (tmp_path_cwd / "test.yaml").read_text()
    assert "opt_steps: 500" in content


def test_config_init_default_path(runner, tmp_path_cwd):
    """config init without -o should create auto3d.yaml."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init"])

    assert result.exit_code == 0
    assert (tmp_path_cwd / "auto3d.yaml").exists()


def test_config_init_invalid_preset(runner, tmp_path_cwd):
    """config init with invalid preset should fail."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-p", "invalid"])

    assert result.exit_code == 1
    assert "Unknown preset" in result.stdout


def test_config_show_displays_file(runner, tmp_path_cwd):
    """config show should display YAML file."""
    from Auto3D.cli.app import app

    # Create a config file first
    config_file = tmp_path_cwd / "test.yaml"
    config_file.write_text("path: input.smi\nk: 5\n")

    result = runner.invoke(app, ["config", "show", str(config_file)])

    assert result.exit_code == 0
    assert "path" in result.stdout


def test_config_show_not_found(runner, tmp_path_cwd):
    """config show should fail if file not found."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "show", "nonexistent.yaml"])

    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_config_validate_valid(runner, tmp_path_cwd):
    """config validate should pass for valid config."""
    from Auto3D.cli.app import app

    # Create a valid config file
    config_file = tmp_path_cwd / "valid.yaml"
    config_file.write_text(
        "path: input.smi\n"
        "k: 5\n"
        "optimizing_engine: AIMNET\n"
        "use_gpu: true\n"
    )

    result = runner.invoke(app, ["config", "validate", str(config_file)])

    assert result.exit_code == 0
    assert "Valid" in result.stdout or "Validation Passed" in result.stdout


def test_config_validate_invalid(runner, tmp_path_cwd):
    """config validate should fail for invalid config."""
    from Auto3D.cli.app import app

    # Create an invalid config file (missing required 'path')
    config_file = tmp_path_cwd / "invalid.yaml"
    config_file.write_text("k: 5\noptimizing_engine: AIMNET\n")

    result = runner.invoke(app, ["config", "validate", str(config_file)])

    assert result.exit_code == 1
