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


def test_models_list_shows_aimnet_registry(runner):
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    out = result.stdout
    assert "AIMNET" in out
    assert "aimnet2-2025" in out  # registry families surfaced
    assert "ANI2x" in out


def test_models_info_aimnet_element_set(runner):
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "info", "AIMNET"])
    assert result.exit_code == 0
    for el in ("B", "As", "Se"):
        assert el in result.stdout  # full 14-element AIMNet2 set


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
    assert "Unknown preset" in result.stderr


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
    assert "not found" in result.stderr


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


def test_run_requires_input(runner):
    """run should require input file argument."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_run_with_nonexistent_file(runner):
    """run should fail with nonexistent file."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["run", "nonexistent.smi"])
    assert result.exit_code != 0


def test_json_output_is_pure_json(runner, tmp_path_cwd, monkeypatch):
    """--json stdout must be parseable even when the k/window warning fires."""
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d
    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem
    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO")); AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1"); w.write(m)
    monkeypatch.setattr(a3d, "main", lambda options: str(out))

    result = runner.invoke(app, ["run", str(smi), "--json"])
    assert result.exit_code == 0
    json.loads(result.stdout)  # must not raise


# Error handling tests

def test_error_hint_configuration_error():
    """get_error_hint should return hint for ConfigurationError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import ConfigurationError

    hint = get_error_hint(ConfigurationError("test"))
    assert hint is not None
    assert "config init" in hint


def test_error_hint_input_validation_error():
    """get_error_hint should return hint for InputValidationError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import InputValidationError

    hint = get_error_hint(InputValidationError("test"))
    assert hint is not None
    assert "validate" in hint


def test_error_hint_model_not_found_error():
    """get_error_hint should return hint for ModelNotFoundError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import ModelNotFoundError

    hint = get_error_hint(ModelNotFoundError("test"))
    assert hint is not None
    assert "AIMNET" in hint


def test_error_hint_gpu_error():
    """get_error_hint should return hint for GPUError."""
    from Auto3D.cli.errors import get_error_hint
    from Auto3D.exceptions import GPUError

    hint = get_error_hint(GPUError("test"))
    assert hint is not None
    assert "--no-gpu" in hint


def test_handle_error_exits():
    """handle_error should raise SystemExit."""
    from Auto3D.cli.errors import handle_error

    with pytest.raises(SystemExit) as exc_info:
        handle_error(Exception("test error"))
    assert exc_info.value.code == 1


def test_models_info_aimnet2_pd(runner):
    """models info aimnet2-pd works and shows Pd in its element set."""
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "info", "aimnet2-pd"])
    assert result.exit_code == 0
    assert "Pd" in result.stdout


def test_models_info_aimnet2_alias(runner):
    """models info aimnet2 (the canonical default) must resolve to the AIMNET entry."""
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "info", "aimnet2"])
    assert result.exit_code == 0
    assert "AIMNet2" in result.stdout


def test_models_info_unkeyed_aimnet2_variant_resolves(runner):
    """Any aimnet2-* registry name -- including a future variant without its own
    ENGINE_INFO block -- must resolve to the base AIMNet2 entry, not 'Unknown
    engine' (check_valid_configuration already accepts any aimnet2* name)."""
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["models", "info", "aimnet2-future"])
    assert result.exit_code == 0, result.stdout
    assert "AIMNet2" in result.stdout


def test_config_validate_missing_file(runner, tmp_path_cwd):
    """config validate on a missing file should fail with a not-found hint."""
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["config", "validate", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stderr
