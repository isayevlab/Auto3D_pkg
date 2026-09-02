"""The same configuration must be validated identically through every door.

Task 1 (C10, M27) closed the gap: Auto3D.foundation.config.FIELD_BOUNDS is the single
table of numeric bounds, enforced by Auto3DOptions.__post_init__ directly and
by Auto3DOptions's `_check_bounds` model validator (both call
`Auto3D.foundation.config.check_field_bounds`). `auto3d run -c` and the legacy
`auto3d config.yaml` invocation (auto3Dcli._run_legacy_yaml) both build a
Auto3DOptions and convert it with `.to_auto3d_options()`, so both also get
extra="forbid", the engine registry check, and Literal validation -- the
legacy path no longer constructs Auto3DOptions directly. Task 3 (C11) closed
the last gap: calc_spe, opt_geometry and calc_thermo now call
`Auto3D.engines.models.policy.check_engine_supports_molecules` (the guard
extracted from check_smi_format/check_sdf_format's formerly-duplicated
element set) and `resolve_engine_name`, the same two checks main() and
smiles2mols already ran via check_input. Task 4 (M15) closed
`TestSmiles2MolsHonesty`'s gap: smiles2mols now raises ConfigurationError for
enumerate_tautomer/a non-rdkit isomer_engine instead of silently ignoring
them, calls check_valid_configuration the same way main() does, and takes a
private copy of its config argument up front so it never mutates the
caller's object. `TestAuxiliaryEntryPointGPUGuard` closes the sibling gap for
the GPU policy: calc_spe, opt_geometry and calc_thermo now also call
`Auto3D.engines.models.policy.check_gpu_requested` directly, so `use_gpu=True`
without CUDA is fatal through the Python API too, not only through the CLI
wrappers in cli/commands/properties.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import Auto3D.entry.auto3D
import Auto3D.presentation.cli.errors
from Auto3D.foundation.config import FIELD_BOUNDS, SENTINEL_FIELDS, Auto3DOptions
from Auto3D.foundation.exceptions import Auto3DError, ConfigurationError, OptimizationError
from tests.helpers_adapter import FakeAdapter


class TestMutuallyExclusiveSelectors:
    """k and window are documented as mutually exclusive; passing both must raise."""

    def test_k_and_window_together_raise(self, isolated_input):
        """select_tautomers raises for this; the conformer ranker must too."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=10, window=5.0)


class TestAuxiliaryEntryPointGuards:
    """calc_spe / opt_geometry / calc_thermo must run the same guard as main()."""

    def test_calc_spe_rejects_charged_input_for_ani(self, job_dir, monkeypatch):
        """A carboxylate must be refused by ANI2x, not silently neutralized.

        calc_spe would otherwise load a real ANI2x model to reach the point
        being tested. To stay within the box's "never load an NNP" constraint,
        the model machinery (get_device/create_model/EnForce_ANI/pad_from_mols)
        is stubbed out the same way tests/test_isomer_engine_hardening.py and
        tests/test_durability.py::TestSameFileGuard already do for calc_spe;
        the guard being tested for (or its absence) sits earlier in the real
        function, so calc_spe itself still runs for real here.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.entry.SPE as spe_mod
        from Auto3D.entry.SPE import calc_spe

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "acetate")

        sdf = job_dir / "charged.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))

        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter(species_pad=0))

        class FakeEnForce:
            def __init__(self, adapter):
                pass

            def forward_batched(self, coords, numbers, charges, atom_mask=None):
                n = coords.shape[0]
                return torch.ones(n, dtype=torch.float64), torch.zeros_like(coords)

        monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)

        def fake_pad(mols, adapter, device):
            n = len(mols)
            coords = torch.zeros(n, 1, 3)
            numbers = torch.zeros(n, 1, dtype=torch.long)
            charges = torch.zeros(n, dtype=torch.long)
            atom_mask = torch.ones(n, 1, dtype=torch.bool)
            return coords, numbers, charges, atom_mask

        monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

        # `use_gpu=False` and `ConfigurationError` (not the base `Auto3DError`)
        # are both load-bearing. check_gpu_requested runs at SPE.py:61, well
        # before the C11 guard at SPE.py:115, and raises GPUError -- also an
        # Auto3DError. So with the default `use_gpu=True` on a CPU-only box
        # (every CI runner; this dev box has 8 CUDA devices and hides it) this
        # call raised GPUError at validation.py:80, `pytest.raises(Auto3DError)`
        # swallowed it, and the test passed green having never reached the
        # charged-input guard it exists to pin. GPUError and ConfigurationError
        # are siblings, so narrowing the expected type makes the wrong reason
        # structurally unable to satisfy this test.
        with pytest.raises(ConfigurationError):
            calc_spe(str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf"))

    def test_opt_geometry_rejects_charged_input_for_ani(self, job_dir, monkeypatch):
        """opt_geometry must run the same check_engine_supports_molecules
        guard as calc_spe -- only calc_spe's copy was pinned before this.

        `optimizing` (batch_opt.batchopt, imported into Auto3D.entry.ASE.geometry)
        is stubbed with a fake that just copies the input through, the same
        defensive reason test_calc_spe_rejects_charged_input_for_ani stubs
        calc_spe's model machinery: if the guard is missing, execution must
        not reach a real ANI2x model load, it must instead reach this
        harmless fake and return normally -- which is the "guard missing"
        signal this test watches for.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.entry.ASE.geometry as geometry_mod
        from Auto3D.entry.ASE.geometry import opt_geometry

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "acetate")
        mol.SetProp("E_tot", "0.0")

        sdf = job_dir / "charged.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        class _FakeOptEngine:
            def __init__(self, in_f, out_f, model_name, device, config, *a, **k):
                self.in_f = in_f
                self.out_f = out_f

            def run(self):
                mols = [m for m in Chem.SDMolSupplier(self.in_f, removeHs=False) if m is not None]
                with Chem.SDWriter(self.out_f) as w:
                    for m in mols:
                        m.SetProp("E_tot", "0.0")
                        w.write(m)
                return True  # matches optimizing.run()'s real True-on-write contract

        monkeypatch.setattr(geometry_mod, "optimizing", _FakeOptEngine)

        # ConfigurationError, not the base Auto3DError: see the note in
        # test_calc_spe_rejects_charged_input_for_ani. This test already passes
        # use_gpu=False, but the narrow type is what keeps a GPUError (or any
        # other unrelated Auto3DError) from satisfying it if that ever changes.
        with pytest.raises(ConfigurationError):
            opt_geometry(str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf"))

    def test_calc_thermo_rejects_charged_input_for_ani(self, job_dir, monkeypatch):
        """calc_thermo must run the same check_engine_supports_molecules
        guard as calc_spe -- only calc_spe's copy was pinned before this.

        `_load_hessian_model` and `model_name2model_calculator` are stubbed
        with fakes that never load real weights: if the guard is missing,
        execution reaches these fakes instead of a real ANI2x model, and the
        fake model's deliberate failure is swallowed by calc_thermo's own
        per-molecule `except Exception` (thermo.py), so calc_thermo returns
        normally with the molecule recorded as failed rather than raising --
        exactly the "guard missing" signal this test watches for.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.entry.ASE.thermo.driver as thermo_mod
        from Auto3D.entry.ASE.thermo import calc_thermo

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "acetate")

        sdf = job_dir / "charged.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        class _FakeCalculator:
            def set_charge(self, charge):
                pass

        class _FakeModel:
            def __call__(self, *a, **k):
                raise RuntimeError("stub model: guard should have fired first")

        monkeypatch.setattr(thermo_mod, "_load_hessian_model", lambda *a, **k: None)
        monkeypatch.setattr(
            thermo_mod,
            "model_name2model_calculator",
            lambda *a, **k: (_FakeModel(), _FakeCalculator()),
        )

        # ConfigurationError, not the base Auto3DError: see the note in
        # test_calc_spe_rejects_charged_input_for_ani. This test already passes
        # use_gpu=False, but the narrow type is what keeps a GPUError (or any
        # other unrelated Auto3DError) from satisfying it if that ever changes.
        with pytest.raises(ConfigurationError):
            calc_thermo(str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf"))


class TestAuxiliaryEntryPointGPUGuard:
    """calc_spe / opt_geometry / calc_thermo must also run
    `Auto3D.engines.models.policy.check_gpu_requested` directly, not only through
    their CLI wrappers (cli/commands/properties.py).

    Each function reaches `model_factory.get_device`, which never raises --
    it silently returns `torch.device('cpu')` -- so before this guard, a
    scripted `use_gpu=True` call to any of the three on a CPU-only box would
    silently compute on CPU with no signal anything was wrong. Same shape as
    `TestAuxiliaryEntryPointGuards` above (a check present at the CLI, absent
    for direct Python-API callers), now closed for the GPU policy too.

    This dev box has 8 CUDA devices, so the no-CUDA case is simulated by
    patching `torch.cuda.is_available` where check_gpu_requested (the single
    source of truth for this check) reads it -- the same technique
    tests/test_validation.py::TestGpuPolicyIsUniform and
    tests/test_cli_property_commands.py's GPU-policy tests already use.
    check_gpu_requested is called before get_device/create_model/optimizing/
    _load_hessian_model/model_name2model_calculator in each function, so none
    of that model machinery needs to be stubbed here: the guard fires before
    any of it runs, and before the input SDF path is even read.
    """

    def test_calc_spe_rejects_gpu_request_without_cuda(self, job_dir):
        from unittest.mock import patch

        from Auto3D.entry.SPE import calc_spe
        from Auto3D.foundation.exceptions import GPUError

        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(GPUError, match="No cuda device") as exc_info:
                calc_spe(str(job_dir / "nonexistent.sdf"), "AIMNET")
        assert "--no-gpu" in str(exc_info.value)

    def test_opt_geometry_rejects_gpu_request_without_cuda(self, job_dir):
        from unittest.mock import patch

        from Auto3D.entry.ASE.geometry import opt_geometry
        from Auto3D.foundation.exceptions import GPUError

        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(GPUError, match="No cuda device") as exc_info:
                opt_geometry(str(job_dir / "nonexistent.sdf"), "AIMNET")
        assert "--no-gpu" in str(exc_info.value)

    def test_calc_thermo_rejects_gpu_request_without_cuda(self, job_dir):
        from unittest.mock import patch

        from Auto3D.entry.ASE.thermo import calc_thermo
        from Auto3D.foundation.exceptions import GPUError

        with patch.object(torch.cuda, "is_available", return_value=False):
            with pytest.raises(GPUError, match="No cuda device") as exc_info:
                calc_thermo(str(job_dir / "nonexistent.sdf"), "AIMNET")
        assert "--no-gpu" in str(exc_info.value)


class TestSmiles2MolsHonesty:
    """smiles2mols must not silently ignore options it cannot honor."""

    @staticmethod
    def _stub_pipeline(monkeypatch):
        """Stub the isomer/optimize/rank/reorder stages of smiles2mols so the
        function runs to completion without embedding real conformers or
        loading a real NNP (box constraint: no NNP loads, no downloads, no
        network). Same pattern as
        tests/test_workflow.py::test_smiles2mols_uses_args_threshold, which
        stubs the same four seams for the same reason.
        """
        import Auto3D.entry.auto3D as auto3D_mod

        class _StubIsomerEngine:
            def run(self):
                return None

        class _StubOpt:
            def __init__(self, *a, **k):
                pass

            def run(self):
                return True  # matches optimizing.run()'s real True-on-write contract

        class _StubRank:
            def __init__(self, *a, **k):
                pass

            def run(self):
                return []

        monkeypatch.setattr(
            auto3D_mod.IsomerEngineFactory,
            "create",
            staticmethod(lambda **kwargs: _StubIsomerEngine()),
        )
        monkeypatch.setattr(auto3D_mod, "optimizing", _StubOpt)
        monkeypatch.setattr(auto3D_mod, "ranking", _StubRank)
        monkeypatch.setattr(auto3D_mod, "reorder_sdf", lambda *a, **k: [])

    def test_unsupported_option_raises(self, isolated_input, monkeypatch):
        """Requesting tautomer enumeration must raise rather than be ignored.

        The isomer/optimize/rank/reorder stages are stubbed (see
        _stub_pipeline) so this stays hermetic and exercises only whether
        enumerate_tautomer is honored, not the full pipeline.
        """
        from Auto3D.entry.auto3D import smiles2mols

        self._stub_pipeline(monkeypatch)

        args = Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, use_gpu=False)
        args.enumerate_tautomer = True

        with pytest.raises((Auto3DError, NotImplementedError)):
            smiles2mols(["CCO"], args)

    def test_caller_config_is_not_mutated(self, isolated_input, monkeypatch):
        """smiles2mols must not modify the config object it was given.

        The isomer/optimize/rank/reorder stages are stubbed (see
        _stub_pipeline) so this stays hermetic; only the mutation of the
        caller's config is under test here.
        """
        from Auto3D.entry.auto3D import smiles2mols

        self._stub_pipeline(monkeypatch)

        args = Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, use_gpu=False)
        before = args.path

        try:
            smiles2mols(["CCO"], args)
        except Exception:
            pass  # only the mutation matters here, not whether the run succeeded

        assert args.path == before, f"caller's config was mutated: {before!r} -> {args.path!r}"


class TestSmiles2MolsRaisesWhenNothingWasOptimized:
    """Issue 8: an all-embedding-failure run must raise OptimizationError, not
    surface as an opaque OSError once the TemporaryDirectory smiles2mols
    writes into has already been (or is about to be) torn down.

    The isomer/embedding stage is stubbed to write nothing (as a genuine
    all-SMILES-failed-to-embed run would leave `meta["enumerated_sdf"]`), so
    the REAL `optimizing` class hits its own missing-input early return and
    the guard added in `smiles2mols` fires against that real return value.
    `preflight_model`/`create_model`/`get_device` are stubbed to avoid a
    network call or a real NNP load.
    """

    def test_raises_optimization_error(self, monkeypatch):
        import Auto3D.entry.auto3D as auto3D_mod

        class _StubIsomerEngine:
            def run(self):
                return None  # writes nothing: meta["enumerated_sdf"] never exists

        monkeypatch.setattr(
            auto3D_mod.IsomerEngineFactory,
            "create",
            staticmethod(lambda **kwargs: _StubIsomerEngine()),
        )
        monkeypatch.setattr(auto3D_mod, "preflight_model", lambda *a, **k: None)
        monkeypatch.setattr(auto3D_mod, "get_device", lambda *a, **k: torch.device("cpu"))
        monkeypatch.setattr(auto3D_mod, "create_model", lambda *a, **k: FakeAdapter())

        args = Auto3DOptions(k=1, use_gpu=False)

        with pytest.raises(OptimizationError):
            auto3D_mod.smiles2mols(["CCO"], args)


class TestDuplicateInchikeyInputs:
    """Two inputs that collide on InChIKey must both survive, not merge.

    Note: this test's class was omitted from the task brief's Step 1 code
    block even though M17 is named in the brief's own "Findings" line and in
    the phase plan (docs/superpowers/plans/2026-07-30-phase0-verification-harness.md:898)
    -- added here to close that gap so the file actually covers every
    finding it claims to.
    """

    def test_duplicate_smiles_both_survive(self, isolated_input, monkeypatch):
        """Two identical SMILES (same InChIKey) must each yield a structure
        in the output -- not collapse into a single winner.

        Only the NNP-loading `optimizing` stage is stubbed (real RDKit
        embedding of two ethanol molecules is trivial and hermetic, and the
        defect lives downstream of it, in the real grouping/ranking code);
        ranking and reorder_sdf run for real since they are what is under
        test.
        """
        from rdkit import Chem

        import Auto3D.entry.auto3D as auto3D_mod
        from Auto3D.entry.auto3D import smiles2mols

        class _FakeOptimizing:
            """Marks every real embedded conformer 'Converged' with a
            distinct energy, so ranking.run's real grouping/top-k logic
            executes exactly as it would after a genuine optimization --
            without loading an NNP."""

            def __init__(self, in_f, out_f, *a, **k):
                self.in_f = in_f
                self.out_f = out_f

            def run(self):
                mols = [m for m in Chem.SDMolSupplier(self.in_f, removeHs=False) if m is not None]
                with Chem.SDWriter(self.out_f) as w:
                    for i, m in enumerate(mols):
                        m.SetProp("Converged", "True")
                        m.SetProp("E_tot", str(float(i)))
                        w.write(m)
                return True  # matches optimizing.run()'s real True-on-write contract

        monkeypatch.setattr(auto3D_mod, "optimizing", _FakeOptimizing)

        args = Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, use_gpu=False, mpi_np=1)
        mols = smiles2mols(["CCO", "CCO"], args)

        assert len(mols) == 2, (
            f"expected one output structure per input (2), got {len(mols)}: "
            f"{[m.GetProp('_Name') for m in mols]}"
        )
        # M17: the second "CCO" is disambiguated to "<inchikey>_2" by
        # smiles2smi's InChIKey-collision handling (utils/smi_io.py), and
        # ranking.species_id must recover that suffix intact (rsplit on the
        # last two "_"-delimited components) rather than strip everything
        # after the FIRST underscore -- which would collapse both outputs
        # back onto the bare InChIKey. `len(mols) == 2` alone does not catch
        # that regression: both mols still "survive", just under the same
        # name. Reverting ranking.species_id to the old `split("_")[0]`
        # makes this fail (both mols come back named identically) while
        # leaving len(mols) == 2 true.
        names = [m.GetProp("_Name") for m in mols]
        assert len(set(names)) == 2, (
            f"expected 2 distinct output names (InChIKey disambiguation "
            f"must survive ranking), got {names}"
        )


class TestSelectorRequiredEverywhere:
    """A missing conformer selector must be refused by every entry point.

    `auto3d run` used to be the sole exception: it injected `k=1` with a
    warning while `main()`, `smiles2mols` and the legacy `auto3d config.yaml`
    form all raised. That made the CLI the one surface that would pick a
    scientific parameter on the user's behalf -- a user who forgot `--k`
    silently got one conformer per molecule, which is a result, not an error,
    and therefore indistinguishable from a deliberate choice downstream.
    """

    def test_cli_run_refuses_when_neither_selector_is_given(self, tmp_path):
        from typer.testing import CliRunner

        from Auto3D.presentation.cli.app import app

        smi = tmp_path / "mols.smi"
        smi.write_text("CCO m1\n")

        result = CliRunner().invoke(app, ["run", str(smi)])

        assert result.exit_code == 2, result.output
        assert "k or window" in result.output

        # The exit code alone does NOT discriminate: with the CLI check
        # removed, main()'s own check_valid_configuration raises the same
        # ConfigurationError and handle_error maps it to exit 2 as well.
        # Verified by mutation -- an exit-code-only assertion passes either
        # way. What distinguishes "refused by the CLI, before any work" from
        # "refused later, inside main()" is the startup banner, which
        # execute_run prints AFTER this check and BEFORE calling main().
        assert "Engine:" not in result.output, (
            "the startup banner was printed, so the run was not refused "
            "until after the CLI had already committed to starting it"
        )
        assert not list(tmp_path.glob("mols_*")), (
            "run created a job directory before refusing the configuration"
        )

    def test_config_validate_agrees_with_the_runner(self, tmp_path):
        """The pre-flight checker must not bless what the runner rejects.

        This previously emitted `Neither 'k' nor 'window' specified - using
        k=1` as a *warning* and exited 0 -- true of `auto3d run` at the time,
        and true of nothing now.
        """
        from typer.testing import CliRunner

        from Auto3D.presentation.cli.app import app

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("path: mols.smi\noptimizing_engine: AIMNET\n")

        result = CliRunner().invoke(app, ["config", "validate", str(cfg)])

        assert result.exit_code != 0, result.output
        assert "Validation Passed" not in result.output
        assert "using k=1" not in result.output

    def test_the_python_api_still_refuses(self, tmp_path):
        """Negative control: the API's refusal is what the CLI now matches."""
        from Auto3D.entry.auto3D import main
        from Auto3D.foundation.config import Auto3DOptions

        smi = tmp_path / "mols.smi"
        smi.write_text("CCO m1\n")

        with pytest.raises(ConfigurationError, match="k or window"):
            main(Auto3DOptions(path=str(smi), use_gpu=False))
