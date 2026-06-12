# tests/test_model_adapter.py
"""Unit tests for the Model Adapter module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Adapter classes are imported locally inside the tests that need them; the
# real-AIMNet2 tests now reuse the session-scoped ``aimnet_model`` fixture
# (see tests/conftest.py) so the model is loaded once per session, not per test.


def test_model_adapter_interface(aimnet_model):
    """A concrete adapter should expose the ModelAdapter interface and a
    forward(coords, species, charges) -> (energy, forces) signature.

    Uses the session-scoped ``aimnet_model`` fixture (an ``AIMNet2Adapter``
    built once via ``create_model("AIMNET", ...)``) instead of loading the real
    AIMNet2 model per-test (~7s). The test is read-only (reads attributes and
    calls ``forward``), so sharing the adapter is safe.
    """
    device = torch.device("cpu")
    adapter = aimnet_model

    # Test interface attributes exist
    assert hasattr(adapter, 'coord_pad')
    assert hasattr(adapter, 'species_pad')
    assert hasattr(adapter, 'device')

    # Test forward signature on two real methane molecules.
    coords = torch.tensor(
        [[[0., 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
          [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63]]]
    ).repeat(2, 1, 1).to(device)
    species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]], device=device)
    charges = torch.tensor([0.0, 0.0], device=device)

    energy, forces = adapter.forward(coords, species, charges)
    assert energy.shape == (2,)
    assert forces.shape == (2, 5, 3)


class TestModelAdapterProtocol:
    """Tests for the ModelAdapter protocol."""

    def test_protocol_is_defined(self):
        """Protocol should be properly defined."""
        from Auto3D.models.adapter import ModelAdapter

        # Check that ModelAdapter has forward method
        assert hasattr(ModelAdapter, 'forward')


class TestBaseModelAdapter:
    """Tests for the BaseModelAdapter base class."""

    def test_base_adapter_stores_model_and_device(self):
        """BaseModelAdapter should store model, device, and padding values."""
        from Auto3D.models.adapter import BaseModelAdapter

        # Create a mock model
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.parameters.return_value = iter([])

        device = torch.device("cpu")

        # We can't instantiate abstract class directly, so we need a concrete subclass
        class ConcreteAdapter(BaseModelAdapter):
            def forward(self, coords, species, charges):
                return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

        adapter = ConcreteAdapter(mock_model, device, coord_pad=1.0, species_pad=-1)

        assert adapter.model == mock_model
        assert adapter.device == device
        assert adapter.coord_pad == 1.0
        assert adapter.species_pad == -1


class TestAIMNet2Adapter:
    """Tests for the AIMNet2Adapter (aimnet-backed)."""

    def test_aimnet2_adapter_loads_default_model(self, aimnet_model):
        """AIMNet2Adapter resolves the 'aimnet2' registry name (no .jpt path).

        Uses the shared session ``aimnet_model`` fixture (built via
        ``create_model("AIMNET", ...)``, which resolves to the ``aimnet2``
        registry default) rather than reloading the real model. Read-only.
        """
        adapter = aimnet_model

        assert adapter.model_name == "aimnet2"
        # An underlying nn.Module is built from the aimnet registry.
        assert adapter.model is not None

    def test_aimnet2_adapter_has_correct_padding(self, aimnet_model):
        """AIMNet2Adapter should have coord_pad=0.0 and species_pad=0.

        Reuses the shared session ``aimnet_model`` fixture (read-only).
        """
        adapter = aimnet_model

        assert adapter.coord_pad == 0.0
        assert adapter.species_pad == 0

    # Note: the former test_aimnet_adapter_forward_calls_model (which mocked a
    # jit-loaded model and inspected the dict passed to it) is intentionally
    # dropped. The new adapter delegates to AIMNet2Calculator, and a real
    # forward pass is already covered end-to-end by
    # test_aimnet2_adapter_energy_forces_water / _padded_batch_matches_unpadded
    # below, which assert energy/force shapes and physical values.


class TestANI2xtAdapter:
    """Tests for the ANI2xt adapter."""

    def test_ani2xt_adapter_creates_model(self):
        """ANI2xtAdapter should create ANI2xt model."""
        # Import torchani to check if it's available (needed for ANI2xt)
        pytest.importorskip("torchani")

        from Auto3D.models.adapter import ANI2xtAdapter

        device = torch.device("cpu")
        adapter = ANI2xtAdapter(device)

        assert adapter.species_pad == -1
        assert adapter.coord_pad == 0.0


class TestANI2xAdapter:
    """Tests for the ANI2x adapter."""

    def test_ani2x_adapter_creates_model(self):
        """ANI2xAdapter should create ANI2x model from torchani."""
        # Import torchani to check if it's available
        pytest.importorskip("torchani")

        from Auto3D.models.adapter import ANI2xAdapter

        device = torch.device("cpu")
        adapter = ANI2xAdapter(device)

        # Verify adapter has correct padding values
        assert adapter.species_pad == -1
        assert adapter.coord_pad == 0.0


class TestCustomModelAdapter:
    """Tests for the CustomModelAdapter."""

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_custom_adapter_loads_from_path(self, mock_load):
        """CustomModelAdapter should load model from provided path."""
        from Auto3D.models.adapter import CustomModelAdapter

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_model.coord_pad = 1.0
        mock_model.species_pad = -2
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        adapter = CustomModelAdapter("/path/to/model.pt", device)

        mock_load.assert_called_once_with("/path/to/model.pt", map_location=device)
        assert adapter.coord_pad == 1.0
        assert adapter.species_pad == -2

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_custom_adapter_uses_defaults_when_no_attributes(self, mock_load):
        """CustomModelAdapter should use default padding when model has no attributes."""
        from Auto3D.models.adapter import CustomModelAdapter

        # Create a mock that simulates a model without coord_pad/species_pad
        # Using a class without those attributes
        class MockModel:
            def parameters(self):
                return iter([])

        mock_load.return_value = MockModel()

        device = torch.device("cpu")
        adapter = CustomModelAdapter("/path/to/model.pt", device)

        # Default values when model doesn't have the attributes
        assert adapter.coord_pad == 0.0
        assert adapter.species_pad == -1


def test_try_compile_uses_dynamic_default_mode(monkeypatch):
    import Auto3D.models.adapter as adapter
    captured = {}
    def fake_compile(model, **kwargs):
        captured.update(kwargs)
        return model
    monkeypatch.setattr(adapter.torch, "compile", fake_compile)
    import torch.nn as nn
    m = nn.Linear(2, 2)
    adapter._try_compile(m)
    assert captured.get("mode") == "default"
    assert captured.get("dynamic") is True


def test_aimnet2_adapter_energy_forces_water(aimnet_model):
    """Reuses the shared session ``aimnet_model`` adapter (read-only forward)
    instead of reloading the real AIMNet2 model (~7s)."""
    import torch

    ad = aimnet_model
    coord = torch.tensor([[[0.0, 0, 0], [0, 0, 0.97], [0, 0.92, -0.25]]])
    species = torch.tensor([[8, 1, 1]])
    charges = torch.tensor([0.0])
    e, f = ad.forward(coord, species, charges)
    assert e.shape == (1,)
    assert f.shape == (1, 3, 3)
    assert -3000 < float(e[0]) < -1000   # water total energy, eV
    assert ad.species_pad == 0 and ad.coord_pad == 0.0


def test_aimnet2_adapter_padded_batch_matches_unpadded(aimnet_model):
    """Padded multi-size batch must give per-molecule energies equal to solo runs.

    Reuses the shared session ``aimnet_model`` adapter (read-only forward).
    """
    import torch
    ad = aimnet_model

    water_c = torch.tensor([[0.,0,0],[0,0,0.97],[0,0.92,-0.25]])
    water_n = torch.tensor([8,1,1])
    meth_c = torch.tensor([[0.,0,0],[0.63,0.63,0.63],[-0.63,-0.63,0.63],[0.63,-0.63,-0.63],[-0.63,0.63,-0.63]])
    meth_n = torch.tensor([6,1,1,1,1])

    e_w, _ = ad.forward(water_c.unsqueeze(0), water_n.unsqueeze(0), torch.zeros(1))
    e_m, _ = ad.forward(meth_c.unsqueeze(0), meth_n.unsqueeze(0), torch.zeros(1))

    # padded batch: water padded to 5 with species_pad=0
    bc = torch.zeros(2,5,3); bc[0,:3]=water_c; bc[1,:5]=meth_c
    bn = torch.zeros(2,5,dtype=torch.long); bn[0,:3]=water_n; bn[1,:5]=meth_n
    e_b, f_b = ad.forward(bc, bn, torch.zeros(2))
    assert f_b.shape == (2,5,3)
    assert abs(float(e_b[0]) - float(e_w[0])) < 1e-2  # padded water == solo water (NaN-free!)
    assert abs(float(e_b[1]) - float(e_m[0])) < 1e-2
    # padded slots of water (rows 3,4) carry zero force
    assert torch.allclose(f_b[0,3:], torch.zeros(2,3), atol=1e-6)


def test_custom_model_adapter_runs(tmp_path):
    """Custom-NNP path: a scripted (species, coords, charges)->energies model
    must run through CustomModelAdapter and yield finite energy/forces."""
    import torch
    from Auto3D.models.adapter import CustomModelAdapter

    class _Toy(torch.nn.Module):
        coord_pad: float = 0.0
        species_pad: int = -1
        def forward(self, species, coords, charges):
            # simple harmonic-ish energy = sum of squared coords per molecule
            return (coords ** 2).sum(dim=(1, 2))

    p = tmp_path / "toy.pt"
    torch.jit.save(torch.jit.script(_Toy()), str(p))

    ad = CustomModelAdapter(str(p), torch.device("cpu"))
    coords = torch.randn(2, 4, 3)
    species = torch.tensor([[1, 6, 7, 8], [1, 1, 6, -1]])
    charges = torch.tensor([0.0, 0.0])
    e, f = ad.forward(coords, species, charges)
    assert e.shape == (2,)
    assert f.shape == (2, 4, 3)
    assert torch.isfinite(e).all() and torch.isfinite(f).all()
