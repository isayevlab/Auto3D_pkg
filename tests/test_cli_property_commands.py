# tests/test_cli_property_commands.py
"""Fast CLI tests for the new first-class property subcommands and the
modernization changes (exit codes, enums, path validation, --save-intermediate,
config init --force). The heavy API functions are mocked, so no NNP runs here.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from Auto3D.cli.app import app

runner = CliRunner()


@pytest.fixture
def sdf(tmp_path):
    p = tmp_path / "mols.sdf"
    p.write_text("")  # existence is all that matters; calc_* is mocked
    return p


@pytest.fixture
def smi(tmp_path):
    p = tmp_path / "mols.smi"
    p.write_text("CCO ethanol\n")
    return p


# --- parity: each new command reaches its API function ----------------------

def test_energy_invokes_calc_spe(sdf):
    with patch("Auto3D.SPE.calc_spe", return_value="out_E.sdf") as m:
        res = runner.invoke(app, ["energy", str(sdf), "--no-gpu", "--engine", "ANI2x"])
    assert res.exit_code == 0, res.output
    _, kwargs = m.call_args
    assert kwargs["use_gpu"] is False
    assert kwargs["allow_tf32"] is False
    assert kwargs["out_path"] is None


def test_energy_output_flag(sdf, tmp_path):
    out = tmp_path / "custom.sdf"
    with patch("Auto3D.SPE.calc_spe", return_value=str(out)) as m:
        res = runner.invoke(app, ["energy", str(sdf), "--no-gpu", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert m.call_args.kwargs["out_path"] == str(out)


def test_optimize_invokes_opt_geometry(sdf):
    with patch("Auto3D.ASE.geometry.opt_geometry", return_value="out_opt.sdf") as m:
        res = runner.invoke(app, ["optimize", str(sdf), "--no-gpu", "--opt-steps", "5"])
    assert res.exit_code == 0, res.output
    assert m.call_args.kwargs["opt_steps"] == 5


def test_tautomers_invokes_get_stable_tautomers(smi):
    with patch("Auto3D.tautomer.get_stable_tautomers", return_value="out_taut.sdf") as m:
        res = runner.invoke(app, ["tautomers", str(smi), "--no-gpu", "--tauto-k", "3"])
    assert res.exit_code == 0, res.output
    assert m.call_args.kwargs["tauto_k"] == 3


# --- error handling / exit codes --------------------------------------------

def test_missing_input_path_exits_2(tmp_path):
    res = runner.invoke(app, ["energy", str(tmp_path / "nope.sdf")])
    assert res.exit_code == 2  # Typer exists=True usage error
    assert "Traceback" not in res.output


def test_tautomers_k_and_window_mutually_exclusive(smi):
    res = runner.invoke(app, ["tautomers", str(smi), "--tauto-k", "1", "--tauto-window", "2"])
    assert res.exit_code == 2  # ConfigurationError -> exit 2
    assert "Traceback" not in res.output


def test_thermo_without_ase_raises_dependency_error(sdf, monkeypatch):
    # Make `from Auto3D.ASE.thermo import calc_thermo` fail like a missing extra.
    monkeypatch.setitem(sys.modules, "Auto3D.ASE.thermo", None)
    res = runner.invoke(app, ["thermo", str(sdf), "--no-gpu"])
    assert res.exit_code == 3  # DependencyError -> exit 3
    assert "Traceback" not in res.output
    # M26: DependencyError used to carry no dependency_name, so this hint was
    # unreachable and every dependency failure showed "Install the missing
    # dependency: unknown" -- pin the actual install hint the user sees, not
    # just the exit code (which would still pass on that regression).
    assert "pip install ase" in res.output
    assert "unknown" not in res.output


# --- engine-name validation (M21 / C11) --------------------------------------
#
# calc_spe/opt_geometry/calc_thermo pass `engine` straight to create_model with
# no CLIConfig/resolve_engine_name gate of their own; the docstring comment
# above KNOWN_ENGINES used to claim this was "validated downstream" without
# that ever being verified. It was not: before this fix, none of these three
# commands rejected a typo'd registry name (e.g. 'aimnet2-2025x') until it
# failed deep inside model construction. Each API function is mocked here so
# a real NNP is never constructed; `m.assert_not_called()` confirms the
# rejection happens before the mocked call, i.e. before any work is done.

def test_energy_rejects_unknown_engine_before_doing_any_work(sdf):
    with patch("Auto3D.SPE.calc_spe") as m:
        res = runner.invoke(
            app, ["energy", str(sdf), "--no-gpu", "--engine", "aimnet2-2025x"]
        )
    assert res.exit_code == 2  # ConfigurationError -> exit 2
    assert "aimnet2-2025x" in res.output
    m.assert_not_called()


def test_optimize_rejects_unknown_engine_before_doing_any_work(sdf):
    with patch("Auto3D.ASE.geometry.opt_geometry") as m:
        res = runner.invoke(
            app, ["optimize", str(sdf), "--no-gpu", "--engine", "aimnet2-2025x"]
        )
    assert res.exit_code == 2  # ConfigurationError -> exit 2
    assert "aimnet2-2025x" in res.output
    m.assert_not_called()


def test_thermo_rejects_unknown_engine_before_doing_any_work(sdf):
    with patch("Auto3D.ASE.thermo.calc_thermo") as m:
        res = runner.invoke(
            app, ["thermo", str(sdf), "--no-gpu", "--engine", "aimnet2-2025x"]
        )
    assert res.exit_code == 2  # ConfigurationError -> exit 2
    assert "aimnet2-2025x" in res.output
    m.assert_not_called()


def test_tautomers_rejects_unknown_engine_before_doing_any_work(smi):
    """tautomers already routes optimizing_engine through CLIConfig, so this
    was not part of the M21 gap -- confirming it stays closed."""
    with patch("Auto3D.tautomer.get_stable_tautomers") as m:
        res = runner.invoke(
            app, ["tautomers", str(smi), "--no-gpu", "--engine", "aimnet2-2025x"]
        )
    assert res.exit_code != 0
    assert "aimnet2-2025x" in res.output
    m.assert_not_called()


# --- GPU policy: fatal, not a silent CPU fallback (M23) ---------------------
#
# calc_spe/opt_geometry/calc_thermo call model_factory.get_device directly and
# never went through check_input/check_valid_configuration, so a CPU-only box
# used to fall back to CPU silently -- no error, no warning -- while `auto3d
# run`/smiles2mols raised (a ConfigurationError with an unrelated "config
# init" hint, or GPUError, depending on entry point). This dev box has 8 CUDA
# devices (see task-7 brief), so the no-CUDA case is simulated by patching
# torch.cuda.is_available where check_gpu_requested (the single source of
# truth for this check, Auto3D.utils.validation) reads it. The mocked API
# function must never be called: the check must happen before any real work.

def test_energy_rejects_when_gpu_requested_without_cuda(sdf):
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False), \
         patch("Auto3D.SPE.calc_spe") as m:
        res = runner.invoke(app, ["energy", str(sdf)])  # gpu defaults to True
    assert res.exit_code == 4  # GPUError -> exit 4
    assert "--no-gpu" in res.output
    m.assert_not_called()


def test_optimize_rejects_when_gpu_requested_without_cuda(sdf):
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False), \
         patch("Auto3D.ASE.geometry.opt_geometry") as m:
        res = runner.invoke(app, ["optimize", str(sdf)])
    assert res.exit_code == 4
    assert "--no-gpu" in res.output
    m.assert_not_called()


def test_thermo_rejects_when_gpu_requested_without_cuda(sdf):
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False), \
         patch("Auto3D.ASE.thermo.calc_thermo") as m:
        res = runner.invoke(app, ["thermo", str(sdf)])
    assert res.exit_code == 4
    assert "--no-gpu" in res.output
    m.assert_not_called()


def test_energy_no_gpu_still_works_without_cuda(sdf):
    """--no-gpu must keep working on a CPU-only box (not a blanket failure)."""
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False), \
         patch("Auto3D.SPE.calc_spe", return_value="out_E.sdf") as m:
        res = runner.invoke(app, ["energy", str(sdf), "--no-gpu"])
    assert res.exit_code == 0, res.output
    m.assert_called_once()


def test_run_rejects_when_gpu_requested_without_cuda(smi):
    """`auto3d run` must fail the same way, before any worker is forked or any
    model is touched: check_valid_configuration (called from
    WorkflowOrchestrator._validate_input) raises before preflight_model."""
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False):
        res = runner.invoke(app, ["run", str(smi), "--k", "1"])
    assert res.exit_code == 4  # GPUError -> exit 4, not 2 (ConfigurationError)
    assert "--no-gpu" in res.output


def test_exit_code_mapping():
    from Auto3D.cli.errors import exit_code_for
    from Auto3D.exceptions import (
        ConfigurationError,
        DependencyError,
        GPUError,
        InputValidationError,
        ModelLoadError,
        OptimizationError,
    )
    assert exit_code_for(ConfigurationError("x")) == 2
    assert exit_code_for(InputValidationError("x")) == 2
    assert exit_code_for(DependencyError("x")) == 3
    assert exit_code_for(GPUError("x")) == 4
    assert exit_code_for(ModelLoadError("x")) == 5  # ModelError subclass -> 5
    assert exit_code_for(OptimizationError("x")) == 1  # generic
    assert exit_code_for(RuntimeError("x")) == 1


# --- modernization ----------------------------------------------------------

def test_engine_autocomplete():
    from Auto3D.cli.commands.properties import engine_autocomplete
    assert "aimnet2-2025" in engine_autocomplete("aimnet2")
    assert "ANI2x" in engine_autocomplete("ANI")
    assert engine_autocomplete("ZZZ") == []


def test_run_save_intermediate_sets_verbose(smi):
    """--save-intermediate must propagate to Auto3DOptions.verbose."""
    from Auto3D.results import WorkflowResult
    captured = {}

    def fake_main(options, progress_callback=None):
        captured["verbose"] = options.verbose
        return WorkflowResult("nonexistent_out.sdf")  # counts -> 0 (missing file)

    with patch("Auto3D.auto3D.main", side_effect=fake_main):
        res = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--save-intermediate"])
    assert res.exit_code == 0, res.output
    assert captured.get("verbose") is True


def test_run_without_save_intermediate_keeps_verbose_false(smi):
    from Auto3D.results import WorkflowResult
    captured = {}

    def fake_main(options, progress_callback=None):
        captured["verbose"] = options.verbose
        return WorkflowResult("nonexistent_out.sdf")

    with patch("Auto3D.auto3D.main", side_effect=fake_main):
        res = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])
    assert res.exit_code == 0, res.output
    assert captured.get("verbose") is False


def test_config_init_force(tmp_path):
    target = tmp_path / "cfg.yaml"
    sentinel = "path: x.smi\n"
    target.write_text(sentinel)
    # Without --force: refuse to clobber (exit 1, file left untouched).
    res = runner.invoke(app, ["config", "init", "-o", str(target)])
    assert res.exit_code == 1
    assert target.read_text() == sentinel  # not overwritten
    # With --force: overwrite.
    res2 = runner.invoke(app, ["config", "init", "-o", str(target), "--force"])
    assert res2.exit_code == 0, res2.output
    assert target.read_text() != sentinel  # regenerated


def test_config_init_preset_enum_valid(tmp_path):
    target = tmp_path / "cfg.yaml"
    res = runner.invoke(app, ["config", "init", "-o", str(target), "-p", "thorough"])
    assert res.exit_code == 0, res.output
    assert target.exists()


def test_models_test_success(monkeypatch):
    """`models test` loads the engine and runs a forward; reports success."""
    import torch

    class _StubAdapter:
        def forward(self, coords, species, charges):
            return torch.zeros(1), torch.zeros(1, 5, 3)

    monkeypatch.setattr("Auto3D.model_factory.get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr("Auto3D.model_factory.create_model", lambda *a, **k: _StubAdapter())
    res = runner.invoke(app, ["models", "test", "AIMNET", "--no-gpu"])
    assert res.exit_code == 0, res.output
    assert "working" in res.output


def test_models_test_load_failure_exit_code(monkeypatch):
    """A load failure (e.g. missing dependency) exits with the mapped code."""
    from Auto3D.exceptions import DependencyError

    def _boom(*a, **k):
        raise DependencyError("torchani not installed")

    monkeypatch.setattr("Auto3D.model_factory.get_device", lambda *a, **k: __import__("torch").device("cpu"))
    monkeypatch.setattr("Auto3D.model_factory.create_model", _boom)
    res = runner.invoke(app, ["models", "test", "ANI2x", "--no-gpu"])
    assert res.exit_code == 3  # DependencyError -> 3
    assert "Traceback" not in res.output


def test_models_test_non_finite_exit_code(monkeypatch):
    """Non-finite outputs are reported as a model (numerical) error -> exit 5."""
    import torch

    class _NanAdapter:
        def forward(self, coords, species, charges):
            return torch.tensor([float("nan")]), torch.zeros(1, 5, 3)

    monkeypatch.setattr("Auto3D.model_factory.get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr("Auto3D.model_factory.create_model", lambda *a, **k: _NanAdapter())
    res = runner.invoke(app, ["models", "test", "AIMNET", "--no-gpu"])
    assert res.exit_code == 5  # NumericalError (ModelError) -> 5


def test_models_test_rejects_when_gpu_requested_without_cuda(monkeypatch):
    """`models test` reached model_factory.get_device directly and never went
    through check_gpu_requested, so it silently fell back to CPU on a CPU-only
    box instead of failing like `energy`/`optimize`/`thermo` -- the last M23
    gap (see cli/commands/models.py). Simulate the CPU-only box by patching
    torch.cuda.is_available where check_gpu_requested reads it, and confirm
    create_model is never reached: the check must happen before any real work
    (before the model would even be constructed, let alone downloaded)."""
    with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False), \
         patch("Auto3D.model_factory.create_model") as m:
        res = runner.invoke(app, ["models", "test", "AIMNET"])  # gpu defaults to True
    assert res.exit_code == 4  # GPUError -> exit 4
    assert "--no-gpu" in res.output
    m.assert_not_called()


def test_models_test_no_gpu_still_works_without_cuda(monkeypatch):
    """--no-gpu must keep working on a CPU-only box (not a blanket failure)."""
    import torch

    class _StubAdapter:
        def forward(self, coords, species, charges):
            return torch.zeros(1), torch.zeros(1, 5, 3)

    monkeypatch.setattr("Auto3D.utils.validation.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("Auto3D.model_factory.get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr("Auto3D.model_factory.create_model", lambda *a, **k: _StubAdapter())
    res = runner.invoke(app, ["models", "test", "AIMNET", "--no-gpu"])
    assert res.exit_code == 0, res.output


def test_models_test_gpu_works_when_cuda_present(monkeypatch):
    """--gpu (the default) must still succeed when CUDA is actually available."""
    import torch

    class _StubAdapter:
        def forward(self, coords, species, charges):
            return torch.zeros(1), torch.zeros(1, 5, 3)

    monkeypatch.setattr("Auto3D.utils.validation.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("Auto3D.model_factory.get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr("Auto3D.model_factory.create_model", lambda *a, **k: _StubAdapter())
    res = runner.invoke(app, ["models", "test", "AIMNET"])  # gpu defaults to True
    assert res.exit_code == 0, res.output


def test_run_interactive_forwards_progress_callback(smi):
    """Interactive `auto3d run` supplies a live-progress callback to main()."""
    from Auto3D.results import WorkflowResult
    captured = {}

    def fake_main(options, progress_callback=None):
        captured["cb"] = progress_callback
        if progress_callback:  # exercise the render path with a sample event
            progress_callback({"job": 1, "step": 10, "total": 2,
                               "converged": 1, "dropped": 0, "active": 1})
        return WorkflowResult("nonexistent_out.sdf")

    with patch("Auto3D.auto3D.main", side_effect=fake_main):
        res = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])
    assert res.exit_code == 0, res.output
    assert callable(captured["cb"])


def test_run_quiet_passes_no_progress_callback(smi):
    """--quiet keeps stdout clean: no live display, callback is None."""
    from Auto3D.results import WorkflowResult
    captured = {}

    def fake_main(options, progress_callback=None):
        captured["cb"] = progress_callback
        return WorkflowResult("nonexistent_out.sdf")

    with patch("Auto3D.auto3D.main", side_effect=fake_main):
        res = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--quiet"])
    assert res.exit_code == 0
    assert captured["cb"] is None


def test_api_functions_expose_new_params():
    """calc_spe/opt_geometry/calc_thermo must accept out_path/use_gpu/allow_tf32
    so the CLI can drive output location, GPU choice, and TF32 uniformly."""
    import inspect

    from Auto3D.ASE.geometry import opt_geometry
    from Auto3D.ASE.thermo import calc_thermo
    from Auto3D.SPE import calc_spe

    for fn in (calc_spe, opt_geometry, calc_thermo):
        params = inspect.signature(fn).parameters
        assert {"out_path", "use_gpu", "allow_tf32"} <= set(params), fn.__name__
