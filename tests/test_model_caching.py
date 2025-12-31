"""Tests for model caching in ModelFactory."""
from __future__ import annotations

import pytest
import torch

from Auto3D.model_factory import ModelFactory, create_model


class TestModelCaching:
    """Tests for model caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        ModelFactory.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        ModelFactory.clear_cache()

    def test_same_model_returns_cached_instance(self):
        """Test that calling create_model with same args returns cached instance."""
        device = torch.device("cpu")
        model1 = create_model("AIMNET", device)
        model2 = create_model("AIMNET", device)
        assert model1 is model2  # Same instance from cache

    def test_different_models_create_different_instances(self):
        """Test that different model names create different instances."""
        device = torch.device("cpu")
        model_aimnet = create_model("AIMNET", device)
        model_ani2xt = create_model("ANI2xt", device)
        assert model_aimnet is not model_ani2xt

    def test_different_ensemble_settings_create_different_instances(self):
        """Test that different use_ensemble settings create different instances."""
        device = torch.device("cpu")
        model_single = create_model("AIMNET", device, use_ensemble=False)
        model_ensemble = create_model("AIMNET", device, use_ensemble=True)
        assert model_single is not model_ensemble

    def test_clear_cache_removes_models(self):
        """Test that clear_cache removes all cached models."""
        device = torch.device("cpu")
        create_model("AIMNET", device)
        create_model("ANI2xt", device)
        assert ModelFactory.get_cache_info()["size"] == 2
        ModelFactory.clear_cache()
        assert ModelFactory.get_cache_info()["size"] == 0

    def test_use_cache_false_bypasses_cache(self):
        """Test that use_cache=False creates new instance."""
        device = torch.device("cpu")
        model1 = create_model("AIMNET", device)
        model2 = create_model("AIMNET", device, use_cache=False)
        assert model1 is not model2

    def test_get_cache_info_returns_size(self):
        """Test that get_cache_info returns correct size."""
        device = torch.device("cpu")
        assert ModelFactory.get_cache_info()["size"] == 0
        create_model("AIMNET", device)
        assert ModelFactory.get_cache_info()["size"] == 1
        create_model("ANI2xt", device)
        assert ModelFactory.get_cache_info()["size"] == 2

    def test_cache_key_includes_compile_model(self):
        """Test that compile_model affects cache key."""
        device = torch.device("cpu")
        # Note: compile_model may not change the instance if not actually compiled,
        # but it should affect the cache key
        model_no_compile = create_model("AIMNET", device, compile_model=False)
        model_compile = create_model("AIMNET", device, compile_model=True)
        # These should be different cache entries
        assert ModelFactory.get_cache_info()["size"] == 2
