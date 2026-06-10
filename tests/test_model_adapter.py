# tests/test_model_adapter.py
"""Unit tests for the Model Adapter module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.models.adapter import ModelAdapter, AIMNet2Adapter


def test_model_adapter_interface():
    """ModelAdapter should have consistent forward signature."""
    # This will fail until we implement the adapter
    device = torch.device("cpu")
    adapter = AIMNetAdapter(device)

    # Test interface attributes exist
    assert hasattr(adapter, 'coord_pad')
    assert hasattr(adapter, 'species_pad')
    assert hasattr(adapter, 'device')

    # Test forward signature
    coords = torch.randn(2, 5, 3, device=device)
    species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]], device=device)
    charges = torch.tensor([0, 0], device=device)

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


class TestAIMNetAdapter:
    """Tests for the AIMNetAdapter."""

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_aimnet_adapter_loads_default_model(self, mock_load):
        """AIMNetAdapter should load default model from models directory."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        adapter = AIMNetAdapter(device)

        mock_load.assert_called_once()
        call_args = mock_load.call_args
        path_arg = call_args[0][0]

        # Check the path contains expected components
        assert "models" in path_arg
        # Single model is now the default (aimnet2_wb97m-d3_0.jpt)
        assert "aimnet2_wb97m-d3_0.jpt" in path_arg

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_aimnet_adapter_has_correct_padding(self, mock_load):
        """AIMNetAdapter should have coord_pad=0.0 and species_pad=0."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        adapter = AIMNetAdapter(device)

        assert adapter.coord_pad == 0.0
        assert adapter.species_pad == 0

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_aimnet_adapter_forward_calls_model(self, mock_load):
        """AIMNetAdapter.forward should call the underlying model."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_model.return_value = {
            'energy': torch.tensor([1.0, 2.0]),
            'forces': torch.randn(2, 5, 3)
        }
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        adapter = AIMNetAdapter(device)

        coords = torch.randn(2, 5, 3, device=device)
        species = torch.tensor([[6, 1, 1, 1, 1], [6, 1, 1, 1, 1]], device=device)
        charges = torch.tensor([0, 0], device=device)

        energy, forces = adapter.forward(coords, species, charges)

        # Verify the model was called with the right dict
        mock_model.assert_called_once()
        call_args = mock_model.call_args[0][0]
        assert 'coord' in call_args
        assert 'numbers' in call_args
        assert 'charge' in call_args


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


def test_aimnet2_adapter_energy_forces_water():
    import torch
    from Auto3D.models.adapter import AIMNet2Adapter

    ad = AIMNet2Adapter("aimnet2", torch.device("cpu"))
    coord = torch.tensor([[[0.0, 0, 0], [0, 0, 0.97], [0, 0.92, -0.25]]])
    species = torch.tensor([[8, 1, 1]])
    charges = torch.tensor([0.0])
    e, f = ad.forward(coord, species, charges)
    assert e.shape == (1,)
    assert f.shape == (1, 3, 3)
    assert -3000 < float(e[0]) < -1000   # water total energy, eV
    assert ad.species_pad == 0 and ad.coord_pad == 0.0


def test_aimnet2_adapter_padded_batch_matches_unpadded():
    """Padded multi-size batch must give per-molecule energies equal to solo runs."""
    import torch
    from Auto3D.models.adapter import AIMNet2Adapter
    ad = AIMNet2Adapter("aimnet2", torch.device("cpu"))

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
