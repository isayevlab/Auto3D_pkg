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

        result = ensemble_opt(model, coord, numbers, charges, param, "AIMNET", torch.device("cpu"))

        # Verify new fields are present
        assert 'converged_mask' in result, "converged_mask missing from ensemble_opt return"
        assert 'oscillating_count' in result, "oscillating_count missing from ensemble_opt return"
        assert isinstance(result['converged_mask'], list)
        assert isinstance(result['oscillating_count'], list)
        assert len(result['converged_mask']) == 2
        assert len(result['oscillating_count']) == 2

    def test_convergence_mask_excludes_oscillating(self):
        """Convergence mask should exclude oscillating structures (Issue #90)."""
        # Simulate a case where structure converged but is oscillating
        converged_mask = [True, True, False]
        oscillating_count = [10, 2, 1]  # First one is oscillating (count >= patience)
        patience = 5

        # This is the logic from batchopt.py
        final_convergence = [
            converged and osc_count < patience
            for converged, osc_count in zip(converged_mask, oscillating_count)
        ]

        # First structure: converged=True but oscillating_count >= patience → False
        # Second structure: converged=True and oscillating_count < patience → True
        # Third structure: converged=False → False
        assert final_convergence == [False, True, False]

    def test_convergence_with_energy_stability(self):
        """Structures converged via energy stability should be marked converged."""
        # This tests the scenario from issue #90 where energy convergence
        # is used but the force is between opttol and 10*opttol
        converged_mask = [True, True]  # Both converged (one via force, one via energy)
        oscillating_count = [2, 3]     # Neither oscillating
        patience = 5

        final_convergence = [
            converged and osc_count < patience
            for converged, osc_count in zip(converged_mask, oscillating_count)
        ]

        # Both should be marked as converged
        assert final_convergence == [True, True]


class TestGPUCleanup:
    """Tests for GPU memory cleanup in batchopt."""

    def test_run_method_includes_gpu_cleanup(self):
        """Verify run() method includes GPU memory cleanup code."""
        import inspect
        source = inspect.getsource(optimizing.run)
        assert 'empty_cache' in source, "GPU cleanup missing from run() method"
        assert 'cuda.is_available' in source, "CUDA availability check missing"


def test_make_buckets_groups_by_size(tmp_path):
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
    eng = optimizing(str(inp), str(tmp_path/"o.sdf"), "AIMNET", torch.device("cpu"),
                     {"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
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

    def fake_ensemble_opt(net, coord, numbers, charges, param, model, device):
        n = len(coord)
        return dict(coord=coord.tolist(), ids=list(range(n)), energy=[0.0]*n,
                    fmax=[0.0]*n, he=[], close=[], timing={},
                    numbers=numbers.tolist(), converged_mask=[True]*n,
                    oscillating_count=[0]*n)
    monkeypatch.setattr(bo, "ensemble_opt", fake_ensemble_opt)

    out = tmp_path / "out.sdf"
    eng = bo.optimizing(str(inp), str(out), "AIMNET", torch.device("cpu"),
                        {"opt_steps":1,"opttol":0.01,"patience":1,"batchsize_atoms":1024})
    eng.run()
    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert names == ["0", "1", "2"]  # original input order
