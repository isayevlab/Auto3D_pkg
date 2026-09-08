# tests/test_cli_exit_codes.py
"""One test per row of the CLI's exit-code table (``docs/source/cli.rst``).

The table used to exist twice in ``cli.rst``, in two versions that disagreed,
and neither listed code ``6`` -- which ``auto3d run`` has emitted since Phase
4. Worse, several codes it named could not be produced at all: eight raise
sites hard-coded ``SystemExit(1)`` and never reached ``exit_code_for``, so
``auto3d config validate`` answered ``1`` for the same ``k: 0`` file that
``auto3d run -c`` answered ``2`` for -- a pre-flight checker whose exit code
disagreed with the run it exists to predict. The documented "invalid GPU index
-> exit 4" example could not happen either, because ``get_device`` returned
``cuda:99`` on an 8-device box without checking.

So: **every row of that table gets a test here that provokes it through the
real CLI and asserts the integer.** A row with no test is a promise with
nothing behind it, which is the defect class this file exists to close.

The exit code alone is never the whole assertion. Several distinct guards map
to the same integer -- ``GPUError`` from the out-of-range index check and
``GPUError`` from ``check_gpu_requested`` both exit ``4``; the CLI's own
``k``/``window`` check and ``main()``'s both exit ``2`` -- so each test also
keys on the message, or on a side effect that only the intended path produces.
An exit-code-only assertion here would pass with the fix reverted.

Box constraints (see the task brief): no NNP is ever loaded and no model is
ever downloaded. Every invocation below fails before model construction, or
mocks the API function outright.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

import Auto3D.engines.model_factory
from Auto3D.presentation.cli.app import app

runner = CliRunner()

TORCHANI_INSTALLED = importlib.util.find_spec("torchani") is not None


def _flat(text: str) -> str:
    """Collapse whitespace: Rich wraps panels at the console width, so a
    substring assertion on a message can otherwise fail (or, for a negative
    assertion, go vacuously true) purely because of a line break."""
    return " ".join(text.split())


# --- 0: success --------------------------------------------------------------


def test_exit_0_clean_validate(tmp_path):
    smi = tmp_path / "ok.smi"
    smi.write_text("CCO ethanol\nCCC propane\n")

    result = runner.invoke(app, ["validate", str(smi)])

    assert result.exit_code == 0, result.output
    assert "Validation Passed" in _flat(result.stdout)


# --- 1: generic / unexpected error -------------------------------------------


def test_exit_1_unexpected_internal_failure(tmp_path):
    """A non-``Auto3DError`` keeps the generic code.

    ``auto3d validate`` had no error handling at all before this change, so a
    ``.smi`` file that is not valid UTF-8 escaped as a raw
    ``UnicodeDecodeError`` traceback rather than a panel. It is still exit 1 --
    an unrecognized internal failure has no better code -- but it is now
    presented, and the panel's advice to rerun with ``-v`` is now true, because
    ``validate`` gained a ``-v`` flag in the same change.
    """
    smi = tmp_path / "latin1.smi"
    smi.write_bytes(b"CCO caf\xe9\n")

    result = runner.invoke(app, ["validate", str(smi)])

    assert result.exit_code == 1, result.output
    err = _flat(result.stderr)
    assert "Unexpected Error" in err
    assert "UnicodeDecodeError" in err
    assert "Traceback" not in err  # no traceback without -v


def test_exit_1_panel_promise_of_verbose_traceback_is_real(tmp_path):
    """The exit-1 panel says "Run with -v/--verbose for a full traceback."

    On ``validate`` that sentence was a lie: the command had no ``-v`` option,
    so passing one was a usage error (exit 2) and the traceback was
    unreachable. Pinned as its own test because the code is the same either
    way -- only the traceback distinguishes them.
    """
    smi = tmp_path / "latin1.smi"
    smi.write_bytes(b"CCO caf\xe9\n")

    result = runner.invoke(app, ["validate", str(smi), "-v"])

    assert result.exit_code == 1, result.output
    assert "Traceback" in result.stderr
    assert "Traceback" not in result.stdout  # diagnostics stay off stdout


# --- 2: configuration / input-validation error -------------------------------


def test_exit_2_config_validate_agrees_with_the_run_it_predicts(tmp_path):
    """The headline defect: the same file, two codes.

    ``auto3d config validate`` answered ``1`` and ``auto3d run -c`` answered
    ``2`` for one ``k: 0`` config. Both are asserted here, in one test, against
    one file -- a test that only checked ``config validate == 2`` would still
    pass if ``run`` later drifted to some other code, and the guarantee being
    made is that the two *agree*.
    """
    cfg = tmp_path / "k0.yaml"
    cfg.write_text("path: mols.smi\nk: 0\noptimizing_engine: ANI2xt\nuse_gpu: false\n")
    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")

    import Auto3D.entry.auto3D as a3d

    checked = runner.invoke(app, ["config", "validate", str(cfg)])
    with patch.object(a3d, "main", side_effect=AssertionError("must not run")):
        ran = runner.invoke(app, ["run", str(smi), "-c", str(cfg)])

    assert checked.exit_code == 2, checked.output
    assert ran.exit_code == 2, ran.output
    assert checked.exit_code == ran.exit_code
    # Not the code alone: confirm the checker actually rejected the config
    # rather than failing for some unrelated reason that happens to map to 2.
    assert "Validation Passed" not in checked.output


def test_exit_2_validate_rejects_a_file_the_runner_rejects(tmp_path):
    """``auto3d validate`` on a file with unparseable SMILES.

    Was exit 1, which no other input-rejection in the CLI used -- ``auto3d
    run`` raises ``InputValidationError`` and exits 2 for input it cannot read.
    The second assertion is the discriminator: the table of bad lines must
    still be printed, so this is the validator's own verdict and not some
    earlier failure that never got as far as reading the file.
    """
    smi = tmp_path / "bad.smi"
    smi.write_text("not_a_smiles mol1\nCCO mol2\n")

    result = runner.invoke(app, ["validate", str(smi)])

    assert result.exit_code == 2, result.output
    out = _flat(result.stdout)
    assert "Validation Failed" in out
    assert "Invalid SMILES" in out
    # InputValidationError's class hint is "Run 'auto3d validate <file>' to
    # check your input file" -- nonsense at the end of `auto3d validate`, and
    # suppressed with an explicit empty hint at the raise site.
    assert "Run 'auto3d validate" not in _flat(result.stderr)


def test_exit_2_validate_unsupported_format(tmp_path):
    junk = tmp_path / "mols.txt"
    junk.write_text("CCO m1\n")

    result = runner.invoke(app, ["validate", str(junk)])

    assert result.exit_code == 2, result.output
    err = _flat(result.stderr)
    assert "Unsupported file format: .txt" in err
    assert "Supported formats: .smi, .sdf" in err


def test_exit_2_config_init_refuses_to_clobber(tmp_path):
    """``config init``'s overwrite refusal must match the ``-o`` overwrite
    refusal the property commands got earlier on this branch, which raises
    ``ConfigurationError`` and exits 2. The CHANGELOG already claimed the two
    printed "the same message"; only after this change do they also produce
    the same code.

    The file-contents assertion is the discriminator: exit 2 is also what a
    Click usage error gives, and a usage error would leave the file untouched
    too -- but it would not print the "already exists" message, so both
    assertions together pin the intended guard.
    """
    target = tmp_path / "cfg.yaml"
    sentinel = "path: precious.smi\n"
    target.write_text(sentinel)

    result = runner.invoke(app, ["config", "init", "-o", str(target)])

    assert result.exit_code == 2, result.output
    assert target.read_text() == sentinel  # refused, not overwritten
    err = _flat(result.stderr)
    assert "already exists" in err
    assert "--force" in err
    # The ConfigurationError class hint ("Run 'auto3d config init' to generate
    # a valid config file") is the command the user just ran.
    assert "generate a valid config file" not in err


def test_exit_2_config_show_missing_file(tmp_path):
    result = runner.invoke(app, ["config", "show", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 2, result.output
    assert "not found" in _flat(result.stderr)


def test_exit_2_models_info_unknown_engine():
    result = runner.invoke(app, ["models", "info", "NOT_AN_ENGINE"])

    assert result.exit_code == 2, result.output
    err = _flat(result.stderr)
    assert "Unknown engine: NOT_AN_ENGINE" in err
    assert "Available:" in err


def test_exit_2_click_usage_error(tmp_path):
    """Click's own usage errors are 2 as well, which is why the table folds
    them into the same row rather than claiming a separate code for them."""
    result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.smi")])

    assert result.exit_code == 2, result.output


# --- 3: missing optional dependency ------------------------------------------


def _hide_torchani(monkeypatch):
    """Make ``import torchani`` fail exactly as an uninstalled package does.

    ``sys.modules[name] = None`` makes the import machinery raise
    ``ImportError`` with ``.name == "torchani"``, which is the attribute the
    translation in ``ModelFactory.create`` keys on -- so this simulates the
    *absence* of torchani specifically, not a torchani that is present but
    broken (which deliberately still propagates untranslated).

    Hiding the import is not sufficient on its own. ``ModelFactory._cache`` is a
    class attribute, so it outlives every test in the process: if any earlier
    test built an ANI2x model on this device, ``create`` returns that instance
    and never reaches an import at all. The command then succeeds and the test
    fails asserting exit 3 -- but only when the suite happens to run a model
    build first, which is why this surfaced as an intermittent failure under
    ``pytest-randomly`` rather than a reliable one.

    Swapping in an empty dict rather than calling ``clear_cache()`` keeps this
    scoped: monkeypatch restores the populated cache afterwards, so tests that
    follow are not made slower by a cold factory.
    """
    from Auto3D.engines.model_factory import ModelFactory

    monkeypatch.setitem(sys.modules, "torchani", None)
    monkeypatch.setattr(ModelFactory, "_cache", {})


def test_exit_3_models_test_without_torchani(monkeypatch):
    """``auto3d models test ANI2x`` reported a missing torchani as an
    "Unexpected Error" at exit 1 with no install hint, while ``auto3d run``
    reported the identical environment problem as exit 3 with
    "pip install torchani" -- because only ``run`` passes through
    ``check_input``'s dependency probe. The translation now lives in
    ``ModelFactory.create``, which every entry point reaches.
    """
    _hide_torchani(monkeypatch)

    result = runner.invoke(app, ["models", "test", "ANI2x", "--no-gpu"])

    assert result.exit_code == 3, result.output
    err = _flat(result.stderr)
    assert "Dependency Error" in err
    # The hint is the point: exit 3 with "Install the missing dependency:
    # unknown" would be the same integer and none of the value.
    assert "pip install torchani" in err
    assert "unknown" not in err


def test_exit_3_energy_without_torchani(monkeypatch, tmp_path):
    """Same dependency, same code, from a different command -- the brief's
    "from every command" requirement.

    ``calc_spe`` runs for real here (a single ethanol molecule; the failure
    happens at ``create_model``, before any weights are touched) rather than
    being mocked, because mocking it would skip the very call that raises.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _hide_torchani(monkeypatch)

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    mol.SetProp("_Name", "ethanol")
    sdf = tmp_path / "mols.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    out = tmp_path / "energies.sdf"

    result = runner.invoke(
        app, ["energy", str(sdf), "--no-gpu", "--engine", "ANI2x", "-o", str(out)]
    )

    assert result.exit_code == 3, result.output
    assert "pip install torchani" in _flat(result.stderr)
    # Failed before doing any work: no output file was produced.
    assert not out.exists()


@pytest.mark.skipif(
    TORCHANI_INSTALLED, reason="torchani is installed, so its absence cannot be tested"
)
def test_exit_3_without_any_mocking_when_torchani_is_genuinely_absent():
    """Unmocked control for the two tests above.

    ``_hide_torchani`` simulates the missing package; this asserts the same
    outcome with nothing patched at all, on an environment that really lacks
    torchani (this dev box and CI both do). Without it, a translation keyed on
    something ``sys.modules[...] = None`` happens to produce but a real
    ``ModuleNotFoundError`` does not would pass the mocked tests and fail in
    the field.
    """
    result = runner.invoke(app, ["models", "test", "ANI2xt", "--no-gpu"])

    assert result.exit_code == 3, result.output
    assert "pip install torchani" in _flat(result.stderr)


def test_a_broken_torchani_is_not_reported_as_a_missing_one(monkeypatch, tmp_path):
    """Negative control for the translation's scope.

    An ``ImportError`` naming some *other* module -- a torchani that is
    installed but whose own dependency is broken -- must keep its own identity
    rather than being relabeled "install torchani", which would be wrong
    advice. Same judgment ``preflight_model`` documents: a failure the code
    cannot positively identify keeps its traceback.
    """
    import Auto3D.engines.model_factory as mf

    def _broken_adapter(device, compile_model=False):
        raise ModuleNotFoundError(
            "No module named 'some_transitive_dep'", name="some_transitive_dep"
        )

    monkeypatch.setitem(
        mf.ModelFactory._engines._entries,
        "ANI2x",
        mf.ModelFactory._engines.entry("ANI2x").__class__(name="ANI2x", value=_broken_adapter),
    )
    monkeypatch.setattr(mf.ModelFactory, "_cache", {})

    result = runner.invoke(app, ["models", "test", "ANI2x", "--no-gpu"])

    assert result.exit_code == 1, result.output
    err = _flat(result.stderr)
    assert "some_transitive_dep" in err
    assert "pip install torchani" not in err


def _hide_aimnet(monkeypatch):
    """Same technique and reasoning as ``_hide_torchani`` above, for aimnet:
    block the import and swap ``ModelFactory._cache`` for an empty dict so a
    previously built AIMNet model cannot satisfy the request without
    reaching an import."""
    from Auto3D.engines.model_factory import ModelFactory
    from tests.helpers_no_aimnet import hide_aimnet

    hide_aimnet(monkeypatch)
    monkeypatch.setattr(ModelFactory, "_cache", {})


def test_exit_3_models_test_without_aimnet(monkeypatch):
    """The conda-forge package ships without aimnet (pip-only dependency
    chain), so requesting the default engine in that environment must report
    exit 3 with the install hint -- mirroring the torchani case above, which
    is the same integer from a different missing dependency."""
    _hide_aimnet(monkeypatch)

    result = runner.invoke(app, ["models", "test", "AIMNET", "--no-gpu"])

    assert result.exit_code == 3, result.output
    err = _flat(result.stderr)
    assert "Dependency Error" in err
    assert "Install: pip install aimnet" in err
    assert "unknown" not in err


def test_exit_3_run_without_aimnet(monkeypatch, tmp_path):
    """The flagship conda-user path: ``auto3d run`` with the default engine,
    in the environment the conda-forge package creates. ``models test`` above
    covers the same dependency from the health-check command; this covers it
    from the entry point people actually script against.

    No ``--engine`` is passed, so this exercises the default (AIMNET) resolving
    through to the aimnet import -- the exit-130 test elsewhere in this file
    passes ``--engine ANI2xt`` specifically to avoid that import, which is the
    corroboration that omitting it here reaches the real dependency check.
    Preflight raises in the parent before any worker fork or model download,
    so this stays fast-tier safe.
    """
    _hide_aimnet(monkeypatch)

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])

    assert result.exit_code == 3, result.output
    err = _flat(result.stderr)
    assert "Dependency Error" in err
    assert "Install: pip install aimnet" in err


# --- 4: GPU error ------------------------------------------------------------
#
# Two different guards produce exit 4, and both must keep working on both kinds
# of machine: this dev box has 8 CUDA devices, every CI runner has none. Each
# test below therefore patches `torch.cuda.is_available`/`device_count` at the
# module where the guard reads them, so the outcome does not depend on the
# host, and asserts on the *message* -- the integer alone cannot tell the two
# guards apart, and a bounds check that only ran when CUDA happened to be
# present would be green here and red on CI.


def test_exit_4_out_of_range_gpu_index(monkeypatch):
    """The documented "invalid GPU index -> exit 4" example, made real.

    ``get_device(99)`` used to return ``torch.device("cuda:99")`` unchecked,
    deferring the failure into CUDA -- where it surfaces as a driver error far
    from the ``--gpu-idx`` that caused it, and maps to the generic exit 1.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with patch.object(Auto3D.engines.model_factory, "create_model") as m:
        result = runner.invoke(app, ["models", "test", "AIMNET", "--gpu-idx", "99"])

    assert result.exit_code == 4, result.output
    err = _flat(result.stderr)
    # Keyed on the message, not the code: `check_gpu_requested`'s "no CUDA at
    # all" refusal is also a GPUError and also exit 4, so an integer-only
    # assertion passes with this bounds check removed on any CPU-only host.
    assert "GPU index 99 is invalid" in err
    assert "2 CUDA device(s) visible" in err
    assert "No cuda device was detected" not in err
    # And it must fire before anything is constructed or downloaded.
    m.assert_not_called()


def test_exit_4_no_cuda_at_all(monkeypatch):
    """The other exit-4 path, pinned alongside so the two stay distinguishable
    by message even though they share an integer."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with patch.object(Auto3D.engines.model_factory, "create_model") as m:
        result = runner.invoke(app, ["models", "test", "AIMNET"])

    assert result.exit_code == 4, result.output
    err = _flat(result.stderr)
    assert "No cuda device was detected" in err
    assert "GPU index" not in err
    m.assert_not_called()


class TestGetDeviceBoundsCheck:
    """Unit-level pins for the bounds check itself.

    ``torch.cuda.device_count`` is patched in every case, including the ones
    expected to succeed, so each test states the whole world it depends on and
    runs identically on a host with 8 CUDA devices and on one with none.
    """

    def test_out_of_range_raises_gpu_error(self, monkeypatch):
        from Auto3D.engines.model_factory import get_device
        from Auto3D.foundation.exceptions import GPUError

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

        with pytest.raises(GPUError, match="GPU index 4 is invalid"):
            get_device(4, use_gpu=True)

    def test_negative_index_raises_gpu_error(self, monkeypatch):
        """``torch.device("cuda:-1")`` is not even constructible, so a negative
        index used to fail with a ``RuntimeError`` from torch (exit 1)."""
        from Auto3D.engines.model_factory import get_device
        from Auto3D.foundation.exceptions import GPUError

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

        with pytest.raises(GPUError, match="GPU index -1 is invalid"):
            get_device(-1, use_gpu=True)

    def test_last_valid_index_is_accepted(self, monkeypatch):
        """The bound is exclusive at the top: index 3 of 4 devices is valid.
        An off-by-one here would reject a legitimate device."""
        from Auto3D.engines.model_factory import get_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

        assert get_device(3, use_gpu=True) == torch.device("cuda:3")

    def test_no_gpu_never_consults_the_index(self, monkeypatch):
        """``--no-gpu`` must stay a plain CPU request, not a validated one:
        ``gpu_idx`` keeps its default of 0 on the CLI even when the user asked
        for CPU, and a box with no devices would otherwise fail every
        ``--no-gpu`` run."""
        from Auto3D.engines.model_factory import get_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        assert get_device(99, use_gpu=False) == torch.device("cpu")

    def test_no_cuda_never_consults_the_index(self, monkeypatch):
        """On a CPU-only host the index is irrelevant: ``check_gpu_requested``
        owns the "you asked for GPU and there is none" refusal, and this
        function must not produce a second, differently-worded one."""
        from Auto3D.engines.model_factory import get_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert get_device(99, use_gpu=True) == torch.device("cpu")


# --- 5: model error ----------------------------------------------------------


def test_exit_5_unloadable_custom_model(tmp_path):
    """A custom NNP path that is not a loadable model.

    Real, not mocked, and cheap: the file is a few bytes of junk, so torch
    rejects it long before anything resembling a model is built.
    """
    broken = tmp_path / "model.pt"
    broken.write_text("this is not a torch archive")

    result = runner.invoke(app, ["models", "test", str(broken), "--no-gpu"])

    assert result.exit_code == 5, result.output
    err = _flat(result.stderr)
    assert "could not be loaded" in err
    assert "ModelLoad Error" in err


def test_exit_5_non_finite_model_output(monkeypatch):
    """The other ``ModelError`` subclass: a model that loads but produces NaN."""
    # AdapterModuleMixin supplies the ModelAdapter members this stub does not
    # care about; `models test` asks the adapter for the species convention now
    # instead of resolving it from the engine name separately (audit C4).
    from tests.helpers_adapter import AdapterModuleMixin

    class _NanAdapter(AdapterModuleMixin):
        def forward(self, coords, species, charges):
            return torch.tensor([float("nan")]), torch.zeros(1, 5, 3)

    monkeypatch.setattr(
        Auto3D.engines.model_factory, "get_device", lambda *a, **k: torch.device("cpu")
    )
    monkeypatch.setattr(Auto3D.engines.model_factory, "create_model", lambda *a, **k: _NanAdapter())

    result = runner.invoke(app, ["models", "test", "AIMNET", "--no-gpu"])

    assert result.exit_code == 5, result.output
    assert "non-finite" in _flat(result.stderr)


# --- 6: partial success ------------------------------------------------------


def test_exit_6_run_with_missing_molecules(tmp_path, monkeypatch):
    """``run`` completed, but an input molecule produced no output.

    Distinct from 1-5 by construction: nothing raised, a results summary was
    printed, and the run is still incomplete. ``main`` is stubbed so no NNP is
    loaded; the reconciliation ``main`` would have done is expressed directly
    as ``WorkflowResult.failures``.
    """
    import Auto3D.entry.auto3D as a3d
    from Auto3D.foundation.results import WorkflowResult

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\nCCC m2\n")
    # A path that does not exist: `count_output` reports (0, 0) for it, which
    # keeps this test about the failure list and not about parsing an SDF.
    out = tmp_path / "out.sdf"

    monkeypatch.setattr(
        a3d, "main", lambda options, **kw: WorkflowResult(str(out), failures=["m2"])
    )

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])

    assert result.exit_code == 6, result.output
    # The guarantee attached to code 6 is that the report still gets printed
    # before the process exits -- otherwise a caller learns a molecule was
    # lost but never which one. `-v` is what turns the count into names.
    out = _flat(result.stdout)
    assert "1 failed" in out

    verbose = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "-v"])
    assert verbose.exit_code == 6, verbose.output
    assert "m2" in _flat(verbose.stdout)


def test_exit_6_json_document_still_reaches_stdout(tmp_path, monkeypatch):
    """Same run with ``--json``: the document must be parseable, and must be
    the results document (not ``handle_error``'s failure document), because a
    partial run is not an exception."""
    import Auto3D.entry.auto3D as a3d
    from Auto3D.foundation.results import WorkflowResult

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\nCCC m2\n")
    # A path that does not exist: `count_output` reports (0, 0) for it, which
    # keeps this test about the failure list and not about parsing an SDF.
    out = tmp_path / "out.sdf"

    monkeypatch.setattr(
        a3d, "main", lambda options, **kw: WorkflowResult(str(out), failures=["m2"])
    )

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--json"])

    assert result.exit_code == 6, result.output
    document = json.loads(result.stdout)
    assert [f["name"] for f in document["failures"]] == ["m2"]


def test_exit_6_is_not_used_for_a_clean_run(tmp_path, monkeypatch):
    """Control: the same wiring with nothing missing must exit 0, so exit 6
    is genuinely about lost molecules and not about, say, an empty output."""
    import Auto3D.entry.auto3D as a3d
    from Auto3D.foundation.results import WorkflowResult

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")
    out = tmp_path / "out.sdf"

    monkeypatch.setattr(a3d, "main", lambda options, **kw: WorkflowResult(str(out), failures=[]))

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])

    assert result.exit_code == 0, result.output


# --- 7: optimization error (no 3D structure converged) -----------------------


def test_exit_7_no_structure_converged(tmp_path, monkeypatch):
    """``main`` raises ``OptimizationError`` -- no chunk produced a converged
    structure, so there is no results summary to print at all.

    Distinct from 6 by construction: 6 is a run that *returned* but lost some
    molecules; this is a run that never returned, which is why it goes through
    ``handle_error`` (a stderr panel, or a JSON failure document with
    ``--json``) instead of the results-summary path. Distinct from 1-5 only by
    the exception class -- before this fix, ``OptimizationError`` was absent
    from ``EXIT_CODES`` and fell through to the generic code 1, making "no 3D
    structure converged" indistinguishable from an internal crash.
    """
    import Auto3D.entry.auto3D as a3d
    from Auto3D.foundation.exceptions import OptimizationError

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")

    def _raise_optimization_error(options, **kw):
        raise OptimizationError("No 3D structure converged.")

    monkeypatch.setattr(a3d, "main", _raise_optimization_error)

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu"])

    assert result.exit_code == 7, result.output
    err = _flat(result.stderr)
    assert "Optimization Error" in err
    assert "No 3D structure converged." in err


def test_exit_7_json_document_reports_the_failure(tmp_path, monkeypatch):
    """With ``--json``, the failure still reaches stdout as a parseable
    document -- same guarantee ``handle_error`` gives every other exception,
    keyed on the new code rather than the generic ``1``."""
    import json

    import Auto3D.entry.auto3D as a3d
    from Auto3D.foundation.exceptions import OptimizationError

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\n")

    def _raise_optimization_error(options, **kw):
        raise OptimizationError("No 3D structure converged.")

    monkeypatch.setattr(a3d, "main", _raise_optimization_error)

    result = runner.invoke(app, ["run", str(smi), "--k", "1", "--no-gpu", "--json"])

    assert result.exit_code == 7, result.output
    document = json.loads(result.stdout)
    assert document["success"] is False
    assert document["error_type"] == "OptimizationError"
    assert document["exit_code"] == 7


# --- 130: interrupted --------------------------------------------------------


def _squash(text: str) -> str:
    """Strip ANSI and box-drawing characters, then remove all whitespace.

    ``_flat`` above is not enough for the interrupt panel: Rich folds a long job
    directory at the panel width, putting a border character *inside* the path.
    """
    import re

    plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    return "".join(re.sub(r"[─-╿]", "", plain).split())


def test_exit_130_ctrl_c_reports_how_far_the_run_got(tmp_path, monkeypatch):
    """Ctrl-C mid-run. 130 is 128 + SIGINT, the shell convention.

    ``KeyboardInterrupt`` is a ``BaseException``, so ``execute_run``'s
    ``except Exception`` never saw it: an interrupted run printed *nothing* and
    the user could not tell how far it had got or whether anything reached disk.

    **The exit code is worthless as an assertion here and is checked only for
    completeness.** ``typer/core.py`` turns an escaping ``KeyboardInterrupt``
    into ``click.exceptions.Exit(130)`` all by itself, so this test's
    ``exit_code == 130`` passed with the handler deleted outright -- verified,
    not assumed. What the framework does *not* do is say anything about the run,
    and it does nothing at all for the legacy ``auto3d config.yaml`` entry point,
    which is not a Typer command and dumps a raw traceback instead.

    So every load-bearing assertion below keys on the report: the job name is
    echoed back only by a handler that read the configuration, and the counts
    only by one that read the live display.
    """
    import Auto3D.entry.auto3D as a3d
    from Auto3D.presentation.cli.errors import EXIT_INTERRUPTED

    smi = tmp_path / "mols.smi"
    smi.write_text("CCO m1\nCCC m2\n")

    def interrupted_main(options, progress_callback=None, **kwargs):
        progress_callback(
            {"job": 0, "total": 9, "converged": 4, "dropped": 1, "active": 4, "step": 250}
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(a3d, "main", interrupted_main)

    # ANI2xt short-circuits engine resolution, so this test never imports
    # `aimnet`/`warp` for a run that is stubbed out anyway.
    result = runner.invoke(
        app,
        ["run", str(smi), "--k", "1", "--no-gpu", "--engine", "ANI2xt", "--job-name", "kestrel"],
    )

    assert result.exit_code == EXIT_INTERRUPTED, result.output
    err = _squash(result.stderr)
    assert "Interruptedbytheuser" in err
    assert "4converged" in err and "4active" in err and "1dropped" in err
    assert "atstep250" in err
    assert "mols_kestrel" in err, "the interrupt report never named the job directory"
    # Diagnostics only: an interrupted run must not put a results document,
    # or anything else, on the stream --json reserves.
    assert "Interrupted" not in result.stdout


# --- the table itself --------------------------------------------------------


def test_documented_table_lists_exactly_the_codes_the_cli_can_emit():
    """``cli.rst`` must document every code and no phantom ones.

    ``cli.rst`` carried two tables that disagreed; the surviving one is
    generated from nothing, so this checks it against the three structures that
    actually decide the codes -- ``EXIT_CODES``, ``EXIT_PARTIAL_SUCCESS`` and
    ``EXIT_INTERRUPTED`` -- rather than against a second hand-written list that
    would drift the same way the two tables did.
    """
    import re
    from pathlib import Path

    from Auto3D.presentation.cli.commands.run import EXIT_PARTIAL_SUCCESS
    from Auto3D.presentation.cli.errors import EXIT_CODES, EXIT_INTERRUPTED

    expected = {0, 1, *EXIT_CODES.values(), EXIT_PARTIAL_SUCCESS, EXIT_INTERRUPTED}

    cli_rst = Path(__file__).resolve().parents[1] / "docs" / "source" / "cli.rst"
    lines = cli_rst.read_text().splitlines()

    headings = [n for n, ln in enumerate(lines) if ln.strip().lower() == "exit codes"]
    assert len(headings) == 1, (
        f"cli.rst must carry exactly one exit-code section, found "
        f"{len(headings)} at lines {[n + 1 for n in headings]}"
    )

    # The section runs to the next heading underline; 60 lines is comfortably
    # more than the table and comfortably less than the next section's own rows.
    section = "\n".join(lines[headings[0] : headings[0] + 60])
    # `\d+`, not `\d`: a single-digit pattern silently stopped matching the row
    # for 130 rather than failing on it, which would have left the newest code
    # documented-but-unchecked -- the table drifting again, quietly.
    documented = {int(m) for m in re.findall(r"^\s+\* - ``(\d+)``$", section, re.MULTILINE)}
    assert documented == expected, (
        f"cli.rst documents exit codes {sorted(documented)} but the CLI emits {sorted(expected)}"
    )
