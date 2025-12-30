# tests/test_model_adapter.py
"""Unit tests for the Model Adapter module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.models.adapter import ModelAdapter, AIMNetAdapter


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
        assert "aimnet2_wb97m_ens_f.jpt" in path_arg

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
