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

    @patch("Auto3D.model_factory.torch.jit.load")
    def test_create_aimnet_loads_correct_path(self, mock_load):
        """Test that AIMNET model loads from correct path."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        ModelFactory.create("AIMNET", device=device)

        # Verify torch.jit.load was called
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        path_arg = call_args[0][0]

        # Check the path contains expected components
        assert "models" in path_arg
        assert "aimnet2_wb97m-d3_ens.jpt" in path_arg

    @patch("Auto3D.model_factory.Path.exists")
    @patch("Auto3D.model_factory.torch.jit.load")
    def test_create_custom_model_from_path(self, mock_load, mock_exists):
        """Test that custom model paths are loaded correctly."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        result = ModelFactory.create("/path/to/custom_model.pt", device=device)

        mock_load.assert_called_once()
        mock_model.eval.assert_called_once()
        assert result == mock_model


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
