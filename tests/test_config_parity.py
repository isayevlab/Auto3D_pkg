"""The same configuration must be validated identically through every door.

Task 1 (C10, M27) closed the gap: Auto3D.config.FIELD_BOUNDS is the single
table of numeric bounds, enforced by Auto3DOptions.__post_init__ directly and
by CLIConfig's `_check_bounds` model validator (both call
`Auto3D.config.check_field_bounds`). `auto3d run -c` and the legacy
`auto3d config.yaml` invocation (auto3Dcli._run_legacy_yaml) both build a
CLIConfig and convert it with `.to_auto3d_options()`, so both also get
extra="forbid", the engine registry check, and Literal validation -- the
legacy path no longer constructs Auto3DOptions directly. Three auxiliary
entry points still skip the element/charge guard entirely (C11, Task 6+).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from Auto3D.cli.config_schema import CLIConfig
from Auto3D.config import Auto3DOptions
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

    @pytest.mark.xfail(
        strict=True,
        reason="M28: ConformerRanker.run tests `if self.k:` before "
        "`elif self.window:`, so window is silently ignored -- which is why the "
        "shipped 'thorough' preset's window: 5.0 has no effect",
    )
    def test_k_and_window_together_raise(self, isolated_input):
        """select_tautomers raises for this; the conformer ranker must too."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=10, window=5.0)


class TestAuxiliaryEntryPointGuards:
    """calc_spe / opt_geometry / calc_thermo must run the same guard as main()."""

    @pytest.mark.xfail(
        strict=True,
        reason="C11: check_input's charge/element guard runs only in main() and "
        "smiles2mols, so a charged species handed to ANI2x is evaluated as the "
        "neutral molecule -- tens of kcal/mol wrong, with wrong forces",
    )
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

            def forward_batched(self, coords, numbers, charges):
                n = coords.shape[0]
                return torch.ones(n, dtype=torch.float64), torch.zeros_like(coords)

        monkeypatch.setattr(spe_mod, "EnForce_ANI", FakeEnForce)

        def fake_pad(mols, model_name, device, coord_pad, species_pad):
            n = len(mols)
            coords = torch.zeros(n, 1, 3)
            numbers = torch.zeros(n, 1, dtype=torch.long)
            charges = torch.zeros(n, dtype=torch.long)
            atom_mask = torch.ones(n, 1, dtype=torch.bool)
            return coords, numbers, charges, atom_mask

        monkeypatch.setattr(spe_mod, "pad_from_mols", fake_pad)

        with pytest.raises(Auto3DError):
            calc_spe(str(sdf), "ANI2x", out_path=str(job_dir / "out.sdf"))


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

    @pytest.mark.xfail(
        strict=True,
        reason="M15: enumerate_tautomer, isomer_engine and mode_oe have no effect "
        "-- there is no TautomerProcessor in the function and the RDKit engine is "
        "hardcoded at auto3D.py:131",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="M15: auto3D.py:117 sets args['path'] = path0 and :125 sets "
        "args.input_format on the caller's object, leaving path pointing into a "
        "deleted TemporaryDirectory",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="M17: smiles2smi disambiguates a repeated InChIKey as 'KEY_2' "
        "specifically so the input is not dropped (utils/file_ops.py:117-127), "
        "but the pipeline's grouping steps (remove_enantiomers and "
        "ranking.run) key on _Name.split('_')[0], mapping the disambiguated "
        "id back onto the first input -- the two merge into one ranking "
        "group and, with k=1, reorder_sdf finds no molecule for the "
        "disambiguated id and the second input silently vanishes",
    )
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

            def __init__(self, in_f, out_f, name, device, config, *a, **k):
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
        monkeypatch.setattr(
            "Auto3D.cli.errors.handle_error",
            lambda error, verbose=0: errors.append(error),
        )

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
        assert isinstance(error, ValidationError)
