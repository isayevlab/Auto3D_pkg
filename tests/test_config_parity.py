"""The same configuration must be validated identically through every door.

Task 1 (C10, M27) closed the gap: Auto3D.config.FIELD_BOUNDS is the single
table of numeric bounds, enforced by Auto3DOptions.__post_init__ directly and
by CLIConfig's `_check_bounds` model validator (both call
`Auto3D.config.check_field_bounds`). `auto3d run -c` and the legacy
`auto3d config.yaml` invocation (auto3Dcli._run_legacy_yaml) both build a
CLIConfig and convert it with `.to_auto3d_options()`, so both also get
extra="forbid", the engine registry check, and Literal validation -- the
legacy path no longer constructs Auto3DOptions directly. Task 3 (C11) closed
the last gap: calc_spe, opt_geometry and calc_thermo now call
`Auto3D.utils.validation.check_engine_supports_molecules` (the guard
extracted from check_smi_format/check_sdf_format's formerly-duplicated
element set) and `resolve_engine_name`, the same two checks main() and
smiles2mols already ran via check_input. Task 4 (M15) closed
`TestSmiles2MolsHonesty`'s gap: smiles2mols now raises ConfigurationError for
enumerate_tautomer/a non-rdkit isomer_engine instead of silently ignoring
them, calls check_valid_configuration the same way main() does, and takes a
private copy of its config argument up front so it never mutates the
caller's object. `TestAuxiliaryEntryPointGPUGuard` closes the sibling gap for
the GPU policy: calc_spe, opt_geometry and calc_thermo now also call
`Auto3D.utils.validation.check_gpu_requested` directly, so `use_gpu=True`
without CUDA is fatal through the Python API too, not only through the CLI
wrappers in cli/commands/properties.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from Auto3D.cli.config_schema import CLIConfig
from Auto3D.config import FIELD_BOUNDS, SENTINEL_FIELDS, Auto3DOptions
from Auto3D.exceptions import Auto3DError, ConfigurationError


class TestAuto3DOptionsBounds:
    """Auto3DOptions must enforce what CLIConfig enforces."""

    def test_negative_threshold_is_rejected(self, isolated_input):
        """A non-positive RMSD threshold must raise, not silently disable dedup."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, threshold=-1)

    def test_zero_max_confs_is_rejected(self, isolated_input):
        """max_confs must be at least 1."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, max_confs=0)

    def test_zero_convergence_threshold_is_rejected(self, isolated_input):
        """A zero force threshold is unsatisfiable and must be refused."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(
                path=isolated_input("smiles2.smi"), k=1, convergence_threshold=0
            )


class TestMutuallyExclusiveSelectors:
    """k and window are documented as mutually exclusive; passing both must raise."""

    def test_k_and_window_together_raise(self, isolated_input):
        """select_tautomers raises for this; the conformer ranker must too."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=10, window=5.0)


class TestSentinelScopeParityAllElevenFields:
    """``check_field_bounds``'s None/False "not specified" skip must be scoped
    to exactly ``SENTINEL_FIELDS`` (k, window, memory, max_confs) -- the four
    optional fields -- on BOTH Auto3DOptions and CLIConfig, for every one of
    the eleven ``FIELD_BOUNDS`` entries, not just those four.

    Before this fix, the skip in ``check_field_bounds`` applied to all eleven
    ``FIELD_BOUNDS`` keys unconditionally. That accidentally let
    ``Auto3DOptions(path="x.smi", threshold=None)`` (and the same for
    mpi_np/opt_steps/convergence_threshold/patience/batchsize_atoms/capacity)
    through silently, while ``CLIConfig(path=Path("x.smi"), threshold=None)``
    always raised -- those seven fields are typed as plain ``int``/``float``
    (not ``| None``) on CLIConfig, so pydantic's own type validation rejects
    ``None`` there regardless of what ``check_field_bounds`` does, and
    ``False`` reaches ``check_field_bounds`` already coerced to ``0``/``0.0``
    by pydantic (bool is an int subclass) and fails the bound. Reproduced live
    before this fix: ``Auto3DOptions(path="x.smi", threshold=None)`` did not
    raise while ``CLIConfig(path=Path("x.smi"), threshold=None)`` did.

    This iterates ``Auto3D.config.FIELD_BOUNDS``/``SENTINEL_FIELDS`` directly
    (rather than hand-listing the eleven field names a second time here) so a
    field added to one set without the other trips this test immediately,
    instead of silently reintroducing the entry-point divergence this test
    exists to close.
    """

    @staticmethod
    def _kwargs(field: str, value) -> dict:
        # A minimal, valid override set with only `field` set to `value` --
        # every other field stays at its (valid) default, so a rejection can
        # only be attributed to `field`.
        return {field: value}

    @pytest.mark.parametrize("value", [None, False], ids=["None", "False"])
    @pytest.mark.parametrize("field", sorted(FIELD_BOUNDS))
    def test_sentinel_scope_agrees_across_entry_points(self, field, value, isolated_input):
        path = isolated_input("smiles2.smi")
        auto3d_kwargs = {"path": path, **self._kwargs(field, value)}
        cli_kwargs = {"path": Path(path), **self._kwargs(field, value)}

        if field in SENTINEL_FIELDS:
            # Optional field: None/False means "not specified" on both paths.
            Auto3DOptions(**auto3d_kwargs)  # must not raise
            CLIConfig(**cli_kwargs)  # must not raise
        else:
            # Non-optional field: None/False has no "unset" meaning and must
            # be rejected on both paths -- not just one.
            with pytest.raises(ConfigurationError):
                Auto3DOptions(**auto3d_kwargs)
            with pytest.raises(ValidationError):
                CLIConfig(**cli_kwargs)


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

        import Auto3D.SPE as spe_mod
        from Auto3D.SPE import calc_spe

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "acetate")

        sdf = job_dir / "charged.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        monkeypatch.setattr(spe_mod, "get_device", lambda *a, **k: torch.device("cpu"))

        class FakeAdapter:
            coord_pad = 0.0
            species_pad = 0

        monkeypatch.setattr(spe_mod, "create_model", lambda *a, **k: FakeAdapter())

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
            calc_spe(
                str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf")
            )

    def test_opt_geometry_rejects_charged_input_for_ani(self, job_dir, monkeypatch):
        """opt_geometry must run the same check_engine_supports_molecules
        guard as calc_spe -- only calc_spe's copy was pinned before this.

        `optimizing` (batch_opt.batchopt, imported into Auto3D.ASE.geometry)
        is stubbed with a fake that just copies the input through, the same
        defensive reason test_calc_spe_rejects_charged_input_for_ani stubs
        calc_spe's model machinery: if the guard is missing, execution must
        not reach a real ANI2x model load, it must instead reach this
        harmless fake and return normally -- which is the "guard missing"
        signal this test watches for.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.ASE.geometry as geometry_mod
        from Auto3D.ASE.geometry import opt_geometry

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
                mols = [
                    m for m in Chem.SDMolSupplier(self.in_f, removeHs=False)
                    if m is not None
                ]
                with Chem.SDWriter(self.out_f) as w:
                    for m in mols:
                        m.SetProp("E_tot", "0.0")
                        w.write(m)

        monkeypatch.setattr(geometry_mod, "optimizing", _FakeOptEngine)

        # ConfigurationError, not the base Auto3DError: see the note in
        # test_calc_spe_rejects_charged_input_for_ani. This test already passes
        # use_gpu=False, but the narrow type is what keeps a GPUError (or any
        # other unrelated Auto3DError) from satisfying it if that ever changes.
        with pytest.raises(ConfigurationError):
            opt_geometry(
                str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf")
            )

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

        import Auto3D.ASE.thermo as thermo_mod
        from Auto3D.ASE.thermo import calc_thermo

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
            calc_thermo(
                str(sdf), "ANI2x", use_gpu=False, out_path=str(job_dir / "out.sdf")
            )


class TestAuxiliaryEntryPointGPUGuard:
    """calc_spe / opt_geometry / calc_thermo must also run
    `Auto3D.utils.validation.check_gpu_requested` directly, not only through
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

        from Auto3D.exceptions import GPUError
        from Auto3D.SPE import calc_spe

        with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False):
            with pytest.raises(GPUError, match="No cuda device") as exc_info:
                calc_spe(str(job_dir / "nonexistent.sdf"), "AIMNET")
        assert "--no-gpu" in str(exc_info.value)

    def test_opt_geometry_rejects_gpu_request_without_cuda(self, job_dir):
        from unittest.mock import patch

        from Auto3D.ASE.geometry import opt_geometry
        from Auto3D.exceptions import GPUError

        with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False):
            with pytest.raises(GPUError, match="No cuda device") as exc_info:
                opt_geometry(str(job_dir / "nonexistent.sdf"), "AIMNET")
        assert "--no-gpu" in str(exc_info.value)

    def test_calc_thermo_rejects_gpu_request_without_cuda(self, job_dir):
        from unittest.mock import patch

        from Auto3D.ASE.thermo import calc_thermo
        from Auto3D.exceptions import GPUError

        with patch("Auto3D.utils.validation.torch.cuda.is_available", return_value=False):
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
        import Auto3D.auto3D as auto3D_mod

        class _StubIsomerEngine:
            def run(self):
                return None

        class _StubOpt:
            def __init__(self, *a, **k):
                pass

            def run(self):
                return None

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
        from Auto3D.auto3D import smiles2mols

        self._stub_pipeline(monkeypatch)

        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=1, use_gpu=False
        )
        args.enumerate_tautomer = True

        with pytest.raises((Auto3DError, NotImplementedError)):
            smiles2mols(["CCO"], args)

    def test_caller_config_is_not_mutated(self, isolated_input, monkeypatch):
        """smiles2mols must not modify the config object it was given.

        The isomer/optimize/rank/reorder stages are stubbed (see
        _stub_pipeline) so this stays hermetic; only the mutation of the
        caller's config is under test here.
        """
        from Auto3D.auto3D import smiles2mols

        self._stub_pipeline(monkeypatch)

        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=1, use_gpu=False
        )
        before = args.path

        try:
            smiles2mols(["CCO"], args)
        except Exception:
            pass  # only the mutation matters here, not whether the run succeeded

        assert args.path == before, (
            f"caller's config was mutated: {before!r} -> {args.path!r}"
        )


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

        import Auto3D.auto3D as auto3D_mod
        from Auto3D.auto3D import smiles2mols

        class _FakeOptimizing:
            """Marks every real embedded conformer 'Converged' with a
            distinct energy, so ranking.run's real grouping/top-k logic
            executes exactly as it would after a genuine optimization --
            without loading an NNP."""

            def __init__(self, in_f, out_f, *a, **k):
                self.in_f = in_f
                self.out_f = out_f

            def run(self):
                mols = [
                    m
                    for m in Chem.SDMolSupplier(self.in_f, removeHs=False)
                    if m is not None
                ]
                with Chem.SDWriter(self.out_f) as w:
                    for i, m in enumerate(mols):
                        m.SetProp("Converged", "True")
                        m.SetProp("E_tot", str(float(i)))
                        w.write(m)

        monkeypatch.setattr(auto3D_mod, "optimizing", _FakeOptimizing)

        args = Auto3DOptions(
            path=isolated_input("smiles2.smi"), k=1, use_gpu=False, mpi_np=1
        )
        mols = smiles2mols(["CCO", "CCO"], args)

        assert len(mols) == 2, (
            f"expected one output structure per input (2), got {len(mols)}: "
            f"{[m.GetProp('_Name') for m in mols]}"
        )
        # M17: the second "CCO" is disambiguated to "<inchikey>_2" by
        # smiles2smi's InChIKey-collision handling (utils/file_ops.py), and
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


class TestValidationParityAcrossEntryPoints:
    """Phase 5's exit criterion, asserted directly: the same configuration
    must be judged identically by `auto3d run -c`, the legacy
    `auto3d config.yaml` invocation, and the Python API's
    `Auto3DOptions(**yaml)`.
    """

    @staticmethod
    def _params(isolated_input) -> dict:
        # optimizing_engine is 'ANI2xt' (a built-in name), not the default
        # 'AIMNET', so resolving it never imports the optional `aimnet`
        # package -- see the matching note on tests/test_cli.py's
        # _LEGACY_YAML for why that import is avoided in tests.
        return {
            "path": isolated_input("smiles2.smi"),
            "k": 1,
            "window": None,
            "memory": None,
            "capacity": 42,
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
            "batchsize_atoms": 1024,
            "allow_tf32": False,
            "verbose": False,
            "job_name": "",
        }

    @staticmethod
    def _run_legacy_and_capture(tmp_path, params, monkeypatch):
        """Drive auto3Dcli._run_legacy_yaml against a real YAML file built
        from `params` (using the same "None"-string encoding the shipped
        parameters.yaml example uses), with Auto3D.auto3D.main stubbed out
        so no pipeline actually runs.

        Returns (options, error): `options` is the Auto3DOptions that would
        have been passed to main() on success; `error` is the exception
        _run_legacy_yaml caught and handed to handle_error on failure.
        Exactly one of the two is None.
        """
        import yaml as yaml_mod

        from Auto3D.auto3Dcli import _run_legacy_yaml

        text_params = dict(params)
        for key in ("window", "memory", "max_confs"):
            if text_params[key] is None:
                text_params[key] = "None"
        yaml_path = tmp_path / "legacy_params.yaml"
        yaml_path.write_text(yaml_mod.dump(text_params))

        captured: dict = {}

        def fake_main(options, **kwargs):
            captured["options"] = options
            return "fake_output.sdf"

        monkeypatch.setattr("Auto3D.auto3D.main", fake_main)

        errors: list[Exception] = []

        # **kwargs, not a fixed signature. The real handle_error is
        # (error, verbose=0, json_output=False); a stub that omits json_output
        # raises TypeError the moment anything calls it with that argument,
        # and the CLI always does. Under pytest-randomly -- which CI runs and
        # which the box's -p no:randomly hides -- this stub was reachable from
        # later tests and turned their expected exit 2 into exit 1. Thirteen
        # tests failed on an unlucky seed, zero on a lucky one. Accepting the
        # real signature makes the stub harmless wherever it is reached.
        def _capture(error, *args, **kwargs):
            errors.append(error)

        monkeypatch.setattr("Auto3D.cli.errors.handle_error", _capture)

        _run_legacy_yaml(str(yaml_path))
        if errors:
            return None, errors[0]
        return captured["options"], None

    def test_valid_config_agrees_across_entry_points(
        self, tmp_path, isolated_input, monkeypatch
    ):
        """A config every path accepts must produce the same Auto3DOptions."""
        params = self._params(isolated_input)

        via_cliconfig = CLIConfig(**params).to_auto3d_options()
        via_api = Auto3DOptions(**params)
        via_legacy, error = self._run_legacy_and_capture(tmp_path, params, monkeypatch)

        assert error is None, f"legacy YAML path rejected a valid config: {error}"
        # k/window are "not specified" via either None (CLIConfig's sentinel)
        # or False (Auto3DOptions's own sentinel) -- both falsy, both meaning
        # the same thing to ranking.py's `if self.k: ... elif self.window:`
        # -- so compare truthiness for those two, exact value for the rest.
        for field in ("k", "window"):
            assert bool(getattr(via_cliconfig, field)) == bool(getattr(via_api, field)), field
            assert bool(getattr(via_legacy, field)) == bool(getattr(via_api, field)), field
        for field in (
            "threshold", "convergence_threshold", "max_confs",
            "optimizing_engine", "opt_steps", "mpi_np", "patience",
            "batchsize_atoms", "memory", "capacity",
        ):
            assert getattr(via_cliconfig, field) == getattr(via_api, field), field
            assert getattr(via_legacy, field) == getattr(via_api, field), field

    def test_negative_threshold_rejected_by_all_three_entry_points(
        self, tmp_path, isolated_input, monkeypatch
    ):
        """threshold=-1 (C10) must be rejected by every entry point -- not
        silently accepted through one door while another guards it.
        """
        params = self._params(isolated_input)
        params["threshold"] = -1

        with pytest.raises(ConfigurationError):
            Auto3DOptions(**params)

        with pytest.raises(ValidationError):
            CLIConfig(**params)

        options, error = self._run_legacy_and_capture(tmp_path, params, monkeypatch)
        assert options is None, "legacy YAML path accepted threshold=-1"
        # ConfigurationError, not a raw pydantic ValidationError: the legacy
        # path now builds its CLIConfig via build_cli_config (Task 3), which
        # translates ValidationError into ConfigurationError so `handle_error`
        # shows exit code 2 with a hint instead of the generic "Unexpected
        # Error" panel at exit 1 -- the same fix `auto3d run -c` already got.
        assert isinstance(error, ConfigurationError)


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

        from Auto3D.cli.app import app

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

        from Auto3D.cli.app import app

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("path: mols.smi\noptimizing_engine: AIMNET\n")

        result = CliRunner().invoke(app, ["config", "validate", str(cfg)])

        assert result.exit_code != 0, result.output
        assert "Validation Passed" not in result.output
        assert "using k=1" not in result.output

    def test_the_python_api_still_refuses(self, tmp_path):
        """Negative control: the API's refusal is what the CLI now matches."""
        from Auto3D.auto3D import main
        from Auto3D.config import Auto3DOptions

        smi = tmp_path / "mols.smi"
        smi.write_text("CCO m1\n")

        with pytest.raises(ConfigurationError, match="k or window"):
            main(Auto3DOptions(path=str(smi), use_gpu=False))
