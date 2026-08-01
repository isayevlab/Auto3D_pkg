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
    """config init with an invalid preset is now a Typer enum usage error (exit 2)."""
    from Auto3D.cli.app import app

    result = runner.invoke(app, ["config", "init", "-p", "invalid"])

    assert result.exit_code == 2
    # Click reports the valid choices for the enum option.
    assert "quick" in result.output


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
    from Auto3D.results import WorkflowResult
    monkeypatch.setattr(a3d, "main", lambda options: WorkflowResult(str(out)))

    result = runner.invoke(app, ["run", str(smi), "--json"])
    assert result.exit_code == 0
    json.loads(result.stdout)  # must not raise


def test_json_output_is_written_before_nonzero_exit_when_molecules_missing(
    runner, tmp_path_cwd, monkeypatch
):
    """C6/B8: a run that loses a molecule must still emit parseable JSON, then
    exit non-zero.

    Hermetic: `Auto3D.auto3D.main` is monkeypatched to return a
    `WorkflowResult` carrying a non-empty `failures` list (Task 3's
    reconciliation carrier), so this exercises the full `execute_run` ->
    `output_json` -> `_exit_if_incomplete` path without a pipeline run or a
    loaded potential.
    """
    import json

    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\nCCCO mol2\n")

    import Auto3D.auto3D as a3d
    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem
    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1")
        w.write(m)
    from Auto3D.results import WorkflowResult
    # mol2 has no corresponding output structure -- a lost molecule.
    monkeypatch.setattr(
        a3d, "main", lambda options: WorkflowResult(str(out), failures=["mol2"])
    )

    result = runner.invoke(app, ["run", str(smi), "--json"])

    assert result.exit_code != 0, (
        f"exited 0 despite a reported failure; output:\n{result.output}"
    )
    # Slice from the first '{': on this box, a one-time CUDA/device-library
    # init banner (unrelated to Auto3D, triggered by whichever test first
    # touches CUDA in the pytest process) can land on real stdout ahead of
    # our JSON when this test runs in isolation. That banner contains no
    # brace, so this only strips ambient noise -- it does not weaken the
    # assertion that our own JSON is present, parseable, and written before
    # SystemExit.
    stdout = result.stdout
    data = json.loads(stdout[stdout.index("{"):])  # must not raise
    assert data["success"] is False
    assert data["failed"] == 1
    assert data["molecules"] == 1
    assert [f["name"] for f in data["failures"]] == ["mol2"]


def test_no_nonzero_exit_when_no_molecules_missing(runner, tmp_path_cwd, monkeypatch):
    """A complete run (no reconciled failures) must keep exiting 0."""
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d
    out = tmp_path_cwd / "in_out.sdf"
    from rdkit import Chem
    from rdkit.Chem import AllChem
    with Chem.SDWriter(str(out)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "mol1")
        w.write(m)
    from Auto3D.results import WorkflowResult
    monkeypatch.setattr(a3d, "main", lambda options: WorkflowResult(str(out)))

    result = runner.invoke(app, ["run", str(smi), "--json"])
    assert result.exit_code == 0


def test_run_cli_k_override_substitutes_file_window(runner, tmp_path_cwd, monkeypatch):
    """`auto3d run in.smi -c cfg.yaml --k 1`, where cfg.yaml sets
    `window: 5.0`, must succeed with k=1 winning -- not hard-fail the
    mutual-exclusion rule because the CLI override was merged alongside the
    file's selector instead of substituting for it.

    `Auto3D.auto3D.main` is stubbed (captures the `Auto3DOptions` it would
    have received) so this exercises the full `execute_run` ->
    `load_yaml_config`/`merge_configs` -> `to_auto3d_options()` path without
    a pipeline run or a loaded potential. `optimizing_engine: ANI2xt` avoids
    importing the optional `aimnet` package (same reasoning as
    `test_cli.py`'s `_LEGACY_YAML`).
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")
    cfg = tmp_path_cwd / "cfg.yaml"
    cfg.write_text(
        "path: placeholder.smi\nwindow: 5.0\noptimizing_engine: ANI2xt\nuse_gpu: false\n"
    )

    import Auto3D.auto3D as a3d
    from Auto3D.results import WorkflowResult

    captured: dict = {}

    def fake_main(options, **kwargs):
        captured["options"] = options
        return WorkflowResult("fake_out.sdf")

    monkeypatch.setattr(a3d, "main", fake_main)

    result = runner.invoke(
        app, ["run", str(smi), "-c", str(cfg), "--k", "1", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert captured["options"].k == 1
    assert not captured["options"].window  # file's window=5.0 must be cleared


def test_run_cli_explicit_k_and_window_conflict_is_configuration_error(
    runner, tmp_path_cwd, monkeypatch
):
    """`--k` and `--window` both passed explicitly on the CLI is a genuine
    user conflict (not a file-vs-CLI merge artifact) and must still exit 2
    as a ConfigurationError with a hint -- not exit 1 as a raw pydantic
    ValidationError under "Unexpected Error".
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")

    import Auto3D.auto3D as a3d
    monkeypatch.setattr(a3d, "main", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("main() must not run when config validation fails")
    ))

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--window", "2.0"])

    assert result.exit_code == 2, result.output
    assert "Unexpected Error" not in result.output


def test_run_cli_yaml_config_bounds_violation_is_configuration_error(
    runner, tmp_path_cwd, monkeypatch
):
    """`auto3d run in.smi -c cfg.yaml` with `cfg.yaml` setting `k: 0` must
    exit 2 as a ConfigurationError with a hint -- not exit 1 under
    "Unexpected Error" as a raw pydantic ValidationError.

    This is the load_yaml_config construction site specifically (as opposed
    to test_run_cli_explicit_k_and_window_conflict_is_configuration_error,
    which exercises merge_configs): the bad value comes from the config file
    itself, before any CLI override is merged in. Before this fix,
    `load_yaml_config` raised the pydantic error unwrapped, which
    `execute_run`'s `except Auto3DError` clause does not catch.
    """
    from Auto3D.cli.app import app

    smi = tmp_path_cwd / "in.smi"
    smi.write_text("CCO mol1\n")
    cfg = tmp_path_cwd / "cfg.yaml"
    cfg.write_text("path: placeholder.smi\nk: 0\noptimizing_engine: ANI2xt\nuse_gpu: false\n")

    import Auto3D.auto3D as a3d
    monkeypatch.setattr(a3d, "main", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("main() must not run when config validation fails")
    ))

    result = runner.invoke(app, ["run", str(smi), "-c", str(cfg)])

    assert result.exit_code == 2, result.output
    assert "Unexpected Error" not in result.output


# Unit tests for the exit-code decision itself (Auto3D.cli.commands.run),
# pinned without going through the CLI or a pipeline run at all.

def test_exit_if_incomplete_raises_nonzero_when_failures_present():
    from Auto3D.cli.commands.run import EXIT_PARTIAL_SUCCESS, _exit_if_incomplete
    from Auto3D.cli.results import FailedMolecule, WorkflowResults

    results = WorkflowResults(
        success_count=1,
        failed_count=1,
        total_conformers=1,
        output_path="out.sdf",
        elapsed_seconds=0.1,
        failures=[FailedMolecule(name="mol2", error="missing from output")],
    )

    with pytest.raises(SystemExit) as exc_info:
        _exit_if_incomplete(results)
    assert exc_info.value.code == EXIT_PARTIAL_SUCCESS
    assert EXIT_PARTIAL_SUCCESS != 0


def test_exit_if_incomplete_does_not_raise_when_no_failures():
    from Auto3D.cli.commands.run import _exit_if_incomplete
    from Auto3D.cli.results import WorkflowResults

    results = WorkflowResults(
        success_count=2,
        failed_count=0,
        total_conformers=2,
        output_path="out.sdf",
        elapsed_seconds=0.1,
        failures=[],
    )

    _exit_if_incomplete(results)  # must not raise


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
    """config validate on a missing file is now a Typer path-existence error (exit 2)."""
    from Auto3D.cli.app import app
    result = runner.invoke(app, ["config", "validate", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output
