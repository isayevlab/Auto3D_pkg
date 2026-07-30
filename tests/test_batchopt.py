# tests/test_batchopt.py
"""Unit tests for the batchopt module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.batch_opt.batchopt import optimizing, EnForce_ANI


class TestOptimizingUsesModelFactory:
    """Tests for optimizing class using ModelFactory."""

    def test_optimizing_uses_model_factory(self):
        """optimizing class should use ModelFactory for model creation."""
        with patch('Auto3D.batch_opt.batchopt.create_model') as mock_factory:
            mock_adapter = MagicMock()
            mock_adapter.coord_pad = 0.0
            mock_adapter.species_pad = 0
            mock_factory.return_value = mock_adapter

            config = {
                'opt_steps': 100,
                'opttol': 0.003,
                'patience': 1000,
                'batchsize_atoms': 1024
            }
            device = torch.device("cpu")
            opt = optimizing("dummy.sdf", "out.sdf", "AIMNET", device, config)

            # Check that create_model was called with the right model name and device
            mock_factory.assert_called_once_with("AIMNET", device, use_ensemble=False)
            # Verify the adapter's properties are used
            assert opt.coord_pad == 0.0
            assert opt.species_pad == 0

    def test_optimizing_uses_adapter_padding_values(self):
        """optimizing should get coord_pad and species_pad from the adapter."""
        with patch('Auto3D.batch_opt.batchopt.create_model') as mock_factory:
            mock_adapter = MagicMock()
            mock_adapter.coord_pad = 1.5
            mock_adapter.species_pad = -2
            mock_factory.return_value = mock_adapter

            config = {
                'opt_steps': 100,
                'opttol': 0.003,
                'patience': 1000,
                'batchsize_atoms': 1024
            }
            device = torch.device("cpu")
            opt = optimizing("dummy.sdf", "out.sdf", "AIMNET", device, config)

            # Verify padding values come from adapter
            assert opt.coord_pad == 1.5
            assert opt.species_pad == -2


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
        def mock_forward(coords, species, charges):
            batch_size = coords.shape[0]
            return torch.ones(batch_size), torch.ones(batch_size, coords.shape[1], 3)

        mock_adapter.forward.side_effect = mock_forward

        model = EnForce_ANI(mock_adapter, batchsize_atoms=10)  # Small batch size

        # Create input that will require multiple batches (5 atoms * 4 batches = 20 atoms > 10)
        coords = torch.randn(4, 5, 3)
        species = torch.ones(4, 5, dtype=torch.long)
        charges = torch.zeros(4, dtype=torch.long)

        energy, forces = model.forward_batched(coords, species, charges)

        # Should have called forward multiple times due to batching
        assert mock_adapter.forward.call_count >= 1
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

        # Verify new fields are present
        assert 'converged_mask' in result, "converged_mask missing from ensemble_opt return"
        assert 'oscillating_count' in result, "oscillating_count missing from ensemble_opt return"
        assert isinstance(result['converged_mask'], list)
        assert isinstance(result['oscillating_count'], list)
        assert len(result['converged_mask']) == 2
        assert len(result['oscillating_count']) == 2


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
        opt = optimizing(
            in_f=str(sdf),
            out_f=str(job_dir / "out.sdf"),
            name="ANI2xt",
            device=torch.device("cpu"),
            config={"opt_steps": 5, "opttol": 1e-9, "patience": 1, "batchsize_atoms": 1024},
        )
        opt.run()

        out = [m for m in Chem.SDMolSupplier(str(job_dir / "out.sdf"), removeHs=False) if m]
        assert out, "optimizer produced no output"
        for m in out:
            if m.HasProp("Dropped_Oscillating") and m.GetProp("Dropped_Oscillating") == "True":
                assert m.GetProp("Converged") != "True", (
                    "a structure dropped for oscillation was reported as converged"
                )


def test_make_buckets_groups_by_size(tmp_path, monkeypatch):
    """Buckets must be size-homogeneous; a size outlier splits into its own bucket."""
    from types import SimpleNamespace

    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch
    from Auto3D.batch_opt.batchopt import optimizing

    # _make_buckets is pure-Python; stub create_model so this never loads the
    # real AIMNet2 model.
    monkeypatch.setattr(
        "Auto3D.batch_opt.batchopt.create_model",
        lambda *a, **k: SimpleNamespace(coord_pad=0.0, species_pad=-1),
    )

    # Build an optimizing instance without running (just to call _make_buckets)
    inp = tmp_path / "in.sdf"
    sizes = ["C", "CC", "CCC", "C1CCCCCCCCCCCCCCCCCCC1"]  # tiny ... and one big ring
    mols = []
    with Chem.SDWriter(str(inp)) as w:
        for i, s in enumerate(sizes):
            m = Chem.AddHs(Chem.MolFromSmiles(s)); AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", str(i)); w.write(m); mols.append(m)
    eng = optimizing(str(inp), str(tmp_path/"o.sdf"), "AIMNET", torch.device("cpu"),
                     {"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
    buckets = eng._make_buckets(mols)
    # the big 20-carbon ring must not share a bucket with methane
    big_idx = 3
    big_bucket = [b for b in buckets if big_idx in b][0]
    assert all(mols[i].GetNumAtoms() > 0.8 * mols[big_idx].GetNumAtoms() for i in big_bucket)


def test_optimizing_preserves_input_order(tmp_path, monkeypatch):
    """Bucketing reorders internally but output order must match input."""
    from types import SimpleNamespace

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
                          species_pad=-1, progress_cb=None):
        n = len(coord)
        return dict(coord=coord.tolist(), ids=list(range(n)), energy=[0.0]*n,
                    fmax=[0.0]*n, he=[], close=[], timing={},
                    numbers=numbers.tolist(), converged_mask=[True]*n,
                    oscillating_count=[0]*n)
    monkeypatch.setattr(bo, "ensemble_opt", fake_ensemble_opt)
    # The optimization itself is faked above; stub create_model so constructing
    # `optimizing` does not load the real AIMNet2 model.
    monkeypatch.setattr(
        bo, "create_model",
        lambda *a, **k: SimpleNamespace(coord_pad=0.0, species_pad=-1),
    )

    out = tmp_path / "out.sdf"
    eng = bo.optimizing(str(inp), str(out), "AIMNET", torch.device("cpu"),
                        {"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
    eng.run()
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert names == ["0", "1", "2"]  # original input order
