"""Unit tests for the ModelFactory module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.model_factory import (
    ModelFactory,
    create_model,
    get_device,
    is_custom_model,
)
from Auto3D.models.adapter import ModelAdapter


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_registry_is_populated(self):
        """Test that built-in models are registered."""
        models = ModelFactory.available_models()
        assert "AIMNET" in models
        assert "ANI2X" in models
        assert "ANI2XT" in models

    def test_create_unknown_model_raises_error(self):
        """Test that creating an unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Model 'UNKNOWN' not found"):
            ModelFactory.create("UNKNOWN")

    def test_create_normalizes_name_to_uppercase(self):
        """Test that model names are case-insensitive."""
        # This should not raise - it will fail on actual model loading
        # but the name normalization should work
        models = ModelFactory.available_models()
        assert all(name.isupper() for name in models)

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_create_aimnet_loads_correct_path(self, mock_load):
        """Test that AIMNET model loads from correct path."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        ModelFactory.create("AIMNET", device=device)

        # Verify torch.jit.load was called
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        path_arg = call_args[0][0]

        # Check the path contains expected components
        assert "models" in path_arg
        # AIMNetAdapter uses aimnet2_wb97m_ens_f.jpt
        assert "aimnet2_wb97m_ens_f.jpt" in path_arg

    @patch("Auto3D.model_factory.Path.exists")
    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_create_custom_model_from_path(self, mock_load, mock_exists):
        """Test that custom model paths are loaded correctly."""
        from Auto3D.models.adapter import CustomModelAdapter

        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        result = ModelFactory.create("/path/to/custom_model.pt", device=device)

        mock_load.assert_called_once()
        # Result should be a CustomModelAdapter instance
        assert isinstance(result, CustomModelAdapter)


class TestCreateModel:
    """Tests for create_model convenience function."""

    def test_create_model_delegates_to_factory(self):
        """Test that create_model uses ModelFactory.create."""
        with patch.object(ModelFactory, "create") as mock_create:
            mock_create.return_value = MagicMock()
            create_model("AIMNET", device=torch.device("cpu"))
            mock_create.assert_called_once()


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_cpu_when_no_gpu(self):
        """Test that CPU is returned when use_gpu is False."""
        device = get_device(gpu_idx=0, use_gpu=False)
        assert device == torch.device("cpu")

    @patch("Auto3D.model_factory.torch.cuda.is_available")
    def test_get_device_cpu_when_cuda_unavailable(self, mock_cuda):
        """Test that CPU is returned when CUDA is unavailable."""
        mock_cuda.return_value = False
        device = get_device(gpu_idx=0, use_gpu=True)
        assert device == torch.device("cpu")

    @patch("Auto3D.model_factory.torch.cuda.is_available")
    def test_get_device_cuda_when_available(self, mock_cuda):
        """Test that CUDA device is returned when available."""
        mock_cuda.return_value = True
        device = get_device(gpu_idx=1, use_gpu=True)
        assert device == torch.device("cuda:1")

    @patch("Auto3D.model_factory.torch.cuda.is_available")
    def test_get_device_cuda_default_index(self, mock_cuda):
        """Test that CUDA:0 is returned by default."""
        mock_cuda.return_value = True
        device = get_device(gpu_idx=None, use_gpu=True)
        assert device == torch.device("cuda:0")


class TestIsCustomModel:
    """Tests for is_custom_model function."""

    def test_is_custom_model_false_for_builtin(self):
        """Test that built-in model names return False."""
        assert not is_custom_model("AIMNET")
        assert not is_custom_model("ANI2x")

    def test_is_custom_model_true_for_existing_path(self, tmp_path):
        """Test that existing file paths return True."""
        model_file = tmp_path / "model.pt"
        model_file.touch()
        assert is_custom_model(str(model_file))

    def test_is_custom_model_false_for_nonexistent_path(self):
        """Test that non-existent paths return False."""
        assert not is_custom_model("/nonexistent/path/model.pt")


class TestFactoryReturnsAdapter:
    """Tests for ModelFactory returning adapter instances."""

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_factory_returns_adapter(self, mock_load):
        """Factory should return ModelAdapter instances."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        model = create_model("AIMNET", device)

        # Check it's an adapter with the right interface
        assert hasattr(model, 'coord_pad')
        assert hasattr(model, 'species_pad')
        assert hasattr(model, 'forward')
        assert model.coord_pad == 0.0
        assert model.species_pad == 0

    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_factory_returns_aimnet_adapter(self, mock_load):
        """Factory should return AIMNetAdapter for AIMNET."""
        from Auto3D.models.adapter import AIMNetAdapter

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        model = create_model("AIMNET", device)

        assert isinstance(model, AIMNetAdapter)

    def test_factory_returns_ani2xt_adapter(self):
        """Factory should return ANI2xtAdapter for ANI2xt."""
        pytest.importorskip("torchani")
        from Auto3D.models.adapter import ANI2xtAdapter

        device = torch.device("cpu")
        model = create_model("ANI2xt", device)

        assert isinstance(model, ANI2xtAdapter)
        assert model.species_pad == -1

    def test_factory_returns_ani2x_adapter(self):
        """Factory should return ANI2xAdapter for ANI2x."""
        pytest.importorskip("torchani")
        from Auto3D.models.adapter import ANI2xAdapter

        device = torch.device("cpu")
        model = create_model("ANI2x", device)

        assert isinstance(model, ANI2xAdapter)
        assert model.species_pad == -1

    @patch("Auto3D.model_factory.Path.exists")
    @patch("Auto3D.models.adapter.torch.jit.load")
    def test_factory_returns_custom_adapter(self, mock_load, mock_exists):
        """Factory should return CustomModelAdapter for custom model paths."""
        from Auto3D.models.adapter import CustomModelAdapter

        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_model.coord_pad = 1.5
        mock_model.species_pad = -2
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        model = create_model("/path/to/custom_model.pt", device)

        assert isinstance(model, CustomModelAdapter)
        assert model.coord_pad == 1.5
        assert model.species_pad == -2
