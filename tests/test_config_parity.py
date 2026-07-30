"""The same configuration must be validated identically through every door.

auto3d run -c goes through CLIConfig and gets extra="forbid", Literal engine
validation, and every Field bound. Auto3DOptions validates only k and window,
and the legacy `auto3d config.yaml` path constructs Auto3DOptions directly --
so the Python API that scientific users script against is the least protected
(C10, M27), and three auxiliary entry points skip the element/charge guard
entirely (C11).
"""
from __future__ import annotations

import pytest

from Auto3D.config import Auto3DOptions
from Auto3D.exceptions import Auto3DError


class TestAuto3DOptionsBounds:
    """Auto3DOptions must enforce what CLIConfig enforces."""

    @pytest.mark.xfail(
        strict=True,
        reason="C10: threshold=-1 is accepted, which sets pruneRmsThresh=-1 and "
        "makes `rmsd < -1` never true, silently disabling duplicate-conformer "
        "removal while presenting the output as deduplicated",
    )
    def test_negative_threshold_is_rejected(self, isolated_input):
        """A non-positive RMSD threshold must raise, not silently disable dedup."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, threshold=-1)

    @pytest.mark.xfail(
        strict=True,
        reason="M27: max_confs has no Field(ge=1) in any path, so max_confs=0 "
        "reaches EmbedMultipleConfs(numConfs=0) and every molecule yields nothing",
    )
    def test_zero_max_confs_is_rejected(self, isolated_input):
        """max_confs must be at least 1."""
        with pytest.raises(Auto3DError):
            Auto3DOptions(path=isolated_input("smiles2.smi"), k=1, max_confs=0)

    @pytest.mark.xfail(
        strict=True,
        reason="C10: convergence_threshold=0 makes `fmax > opttol` permanently "
        "true, so every structure burns the full 2000-step budget",
    )
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
            return coords, numbers, charges

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
