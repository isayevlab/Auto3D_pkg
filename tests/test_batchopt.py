# tests/test_batchopt.py
"""Unit tests for the batchopt module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from Auto3D.batch_opt.batchopt import optimizing, EnForce_ANI
from tests.helpers_adapter import FakeAdapter


class TestEnForceANI:
    """Tests for EnForce_ANI class."""

    def test_enforce_ani_delegates_to_adapter(self):
        """EnForce_ANI.forward should delegate to adapter's forward method."""
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([1.0, 2.0]),
            torch.randn(2, 5, 3)
        )

        # EnForce_ANI should accept the adapter directly
        model = EnForce_ANI(mock_adapter, batchsize_atoms=1024)

        coords = torch.randn(2, 5, 3)
        species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]])
        charges = torch.tensor([0, 0])

        energy, forces = model.forward(coords, species, charges)

        # Verify adapter's forward was called
        mock_adapter.forward.assert_called_once()
        call_args = mock_adapter.forward.call_args
        assert torch.equal(call_args[0][0], coords)
        assert torch.equal(call_args[0][1], species)
        assert torch.equal(call_args[0][2], charges)

    def test_enforce_ani_forward_batched(self):
        """EnForce_ANI.forward_batched should batch calls correctly."""
        mock_adapter = MagicMock()
        # Return consistent results for batching
        def mock_forward(coords, species, charges, atom_mask=None):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        model = EnForce_ANI(mock_adapter, batchsize_atoms=10)  # Small batch size

        # Create input that will require multiple batches (5 atoms * 4 batches = 20 atoms > 10)
        coords = torch.randn(4, 5, 3)
        species = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4, dtype=torch.long)

        energy, forces = model.forward_batched(coords, species, charges)

        # batch_size = max(1, batchsize_atoms // N) = max(1, 10 // 5) = 2
        # molecules per sub-batch; 4 molecules split into chunks of 2 -> the
        # adapter must be called exactly twice, not "at least once" (which a
        # single unbatched call would also satisfy, defeating the point of a
        # forward_batched-specific test). Mirrors test_model_wrapper.py's
        # stronger call_count==2 sibling for the same batchsize_atoms/N ratio.
        assert mock_adapter.forward.call_count == 2
        assert energy.shape == (4,)
        assert forces.shape == (4, 5, 3)


class TestConvergenceStatus:
    """Tests for convergence status determination (Issue #90)."""

    def test_ensemble_opt_returns_convergence_info(self):
        """ensemble_opt should return converged_mask and oscillating_count."""
        from Auto3D.batch_opt.batchopt import ensemble_opt
        from Auto3D.batch_opt.model_wrapper import EnForce_ANI

        # Create mock model
        mock_adapter = MagicMock()
        mock_adapter.forward.return_value = (
            torch.tensor([0.0, 0.0]),
            torch.zeros(2, 3, 3)  # Zero forces = instant convergence
        )
        model = EnForce_ANI(mock_adapter, batchsize_atoms=1024)

        # Create simple input
        coord = torch.randn(2, 3, 3)
        numbers = torch.tensor([[6, 1, 1], [6, 1, 1]], dtype=torch.long)
        charges = torch.tensor([0, 0], dtype=torch.long)
        param = {'opt_steps': 10, 'opttol': 0.01, 'patience': 5}

        result = ensemble_opt(model, coord, numbers, charges, param, torch.device("cpu"))

        # Verify new fields are present, with their actual VALUES: zero force
        # on step 1 means fmax (0.0) is at once below opttol (0.01) for both
        # structures, so both must be reported converged and neither must
        # have been counted as oscillating -- checking only key
        # presence/type/length (as before) would pass even if the values
        # were transposed, all-False, or a stray increment leaked into
        # oscillating_count.
        assert 'converged_mask' in result, "converged_mask missing from ensemble_opt return"
        assert 'oscillating_count' in result, "oscillating_count missing from ensemble_opt return"
        assert isinstance(result['converged_mask'], list)
        assert isinstance(result['oscillating_count'], list)
        assert result['converged_mask'] == [True, True]
        assert result['oscillating_count'] == [0, 0]


@pytest.mark.slow
class TestConvergenceFlagDerivation:
    """The Converged/Dropped_Oscillating flags must come from production code.

    The previous tests in this class re-implemented the derivation inside the
    test body and asserted it against itself, so no production code executed
    (audit M32).
    """

    def test_oscillating_structure_is_not_reported_converged(self, job_dir):
        """A structure at/over the patience limit must not be Converged=True."""
        pytest.importorskip("torchani")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.batch_opt.batchopt import optimizing

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "osc_test")

        sdf = job_dir / "in.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(mol)

        # patience=1 guarantees the oscillation path is taken on any structure
        # that does not reduce fmax on its very first step.
        from Auto3D.model_factory import create_model

        opt = optimizing(
            in_f=str(sdf),
            out_f=str(job_dir / "out.sdf"),
            adapter=create_model("ANI2xt", torch.device("cpu")),
            device=torch.device("cpu"),
            config={"opt_steps": 5, "opttol": 1e-9, "patience": 1, "batchsize_atoms": 1024},
        )
        opt.run()

        out = [m for m in Chem.SDMolSupplier(str(job_dir / "out.sdf"), removeHs=False) if m]
        assert out, "optimizer produced no output"

        # Invariant: this must hold for every molecule regardless of which path
        # its trajectory took, so it can never no-op. It fails if a regression
        # decouples Converged/Dropped_Oscillating (e.g. dropping the
        # `osc_count_i < patience` guard from `convergence_i`).
        for m in out:
            if m.HasProp("Dropped_Oscillating") and m.GetProp("Dropped_Oscillating") == "True":
                assert m.GetProp("Converged") != "True", (
                    "a structure dropped for oscillation was reported as converged"
                )

        # Non-vacuity check: the invariant above is only meaningful if the
        # oscillation path was actually taken. If this fails, the fixture
        # (patience=1, opttol=1e-9, ethanol's FIRE trajectory) has stopped
        # reproducing the oscillating condition on this platform/torch
        # version -- it does not mean the Converged/Dropped_Oscillating
        # invariant itself is broken.
        assert any(m.HasProp("Dropped_Oscillating") and m.GetProp("Dropped_Oscillating") == "True" for m in out), (
            "fixture did not reproduce an oscillating structure (patience=1 "
            "should force this on the very first non-converging step); the "
            "invariant above was never exercised"
        )


def test_make_buckets_groups_by_size(tmp_path, monkeypatch):
    """Buckets must be size-homogeneous; a size outlier splits into its own bucket."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch
    from Auto3D.batch_opt.batchopt import optimizing

    # Build an optimizing instance without running (just to call _make_buckets)
    inp = tmp_path / "in.sdf"
    sizes = ["C", "CC", "CCC", "C1CCCCCCCCCCCCCCCCCCC1"]  # tiny ... and one big ring
    mols = []
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(sizes):
            m = Chem.AddHs(Chem.MolFromSmiles(s)); AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i)); w.write(m); mols.append(m)
    # _make_buckets is pure-Python, so a conforming double is enough and no
    # model is loaded. `optimizing` no longer builds its own adapter, so there is
    # no create_model seam left to patch.
    eng = optimizing(str(inp), str(tmp_path/"o.sdf"), adapter=FakeAdapter(),
                     device=torch.device("cpu"),
                     config={"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
    buckets = eng._make_buckets(mols)
    # the big 20-carbon ring must not share a bucket with methane
    big_idx = 3
    big_bucket = [b for b in buckets if big_idx in b][0]
    assert all(mols[i].GetNumAtoms() > 0.8 * mols[big_idx].GetNumAtoms() for i in big_bucket)


def test_optimizing_preserves_input_order(tmp_path, monkeypatch):
    """Bucketing reorders internally but output order must match input."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch
    import Auto3D.batch_opt.batchopt as bo

    inp = tmp_path / "in.sdf"
    smis = ["CCCCCCCC", "C", "CCC"]  # 8,1,3 heavy atoms - deliberately unsorted
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(smis):
            m = Chem.AddHs(Chem.MolFromSmiles(s)); AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i)); w.write(m)

    def fake_ensemble_opt(net, coord, numbers, charges, param, device,
                          atom_mask=None, progress_cb=None):
        n = len(coord)
        return dict(coord=coord.tolist(), ids=list(range(n)), energy=[0.0]*n,
                    fmax=[0.0]*n, he=[], close=[], timing={},
                    numbers=numbers.tolist(), converged_mask=[True]*n,
                    oscillating_count=[0]*n)
    monkeypatch.setattr(bo, "ensemble_opt", fake_ensemble_opt)

    out = tmp_path / "out.sdf"
    eng = bo.optimizing(str(inp), str(out), adapter=FakeAdapter(),
                        device=torch.device("cpu"),
                        config={"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
    eng.run()
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert names == ["0", "1", "2"]  # original input order


class TestBatchOptDependsDownwards:
    """``batch_opt`` must depend on ``models/``, never on ``model_factory``.

    ``batchopt.py`` imported ``Auto3D.model_factory.create_model`` at module
    scope and called it in ``optimizing.__init__``: the numerical layer
    constructing its own dependency, and reaching UP into the layer that is
    supposed to sit above it. The visible symptom was in the tests -- every one
    of them had to monkeypatch ``Auto3D.batch_opt.batchopt.create_model``, a seam
    that existed only because the arrow pointed the wrong way.
    """

    def test_importing_batchopt_does_not_pull_in_the_factory(self):
        """Asserted in a fresh interpreter: an already-imported ``model_factory``
        would make this vacuous inside the test session."""
        import subprocess
        import sys

        program = (
            "import sys; import Auto3D.batch_opt.batchopt as b; "
            "assert 'Auto3D.model_factory' not in sys.modules, "
            "sorted(m for m in sys.modules if m.startswith('Auto3D')); "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout

    def test_batchopt_exposes_no_create_model_seam(self):
        import Auto3D.batch_opt.batchopt as bo

        assert not hasattr(bo, "create_model")


class TestOptimizingTakesAnAdapterNotAName:
    """Construction is the caller's job, and the caller must be in the worker.

    ``optimizing`` no longer knows an engine name at all: ``pad_from_mols`` asks
    the adapter, so there is nothing left for a name to decide.
    """

    @staticmethod
    def _config():
        return {
            "opt_steps": 100,
            "opttol": 0.003,
            "patience": 1000,
            "batchsize_atoms": 1024,
        }

    def test_an_engine_name_is_rejected(self):
        """The parameters after ``out_f`` are keyword-only, so a stale positional
        call fails at the call rather than silently binding a string into the
        slot that supplies the padding values."""
        with pytest.raises(TypeError):
            optimizing(
                "dummy.sdf", "out.sdf", "AIMNET", torch.device("cpu"), self._config()
            )

    def test_padding_values_come_from_the_injected_adapter(self):
        from tests.helpers_adapter import FakeAdapter

        adapter = FakeAdapter(coord_pad=1.5, species_pad=-2)
        opt = optimizing(
            "dummy.sdf",
            "out.sdf",
            adapter=adapter,
            device=torch.device("cpu"),
            config=self._config(),
        )
        assert opt.model is adapter
        assert opt.coord_pad == 1.5
        assert opt.species_pad == -2

    def test_optimizing_no_longer_carries_an_engine_name(self):
        from tests.helpers_adapter import FakeAdapter

        opt = optimizing(
            "dummy.sdf",
            "out.sdf",
            adapter=FakeAdapter(),
            device=torch.device("cpu"),
            config=self._config(),
        )
        assert not hasattr(opt, "name")
