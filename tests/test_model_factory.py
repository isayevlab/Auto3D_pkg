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
        """available_models() advertises the AIMNet alias, aimnet registry
        names, and the built-in ANI engines (mixed case)."""
        models = ModelFactory.available_models()
        assert "AIMNET" in models
        assert "ANI2x" in models
        assert "ANI2xt" in models
        # AIMNET is NOT a hard-coded adapter key anymore: only ANI engines are.
        assert set(ModelFactory._adapters) == {"ANI2X", "ANI2XT"}
        assert "AIMNET" not in ModelFactory._adapters

    def test_create_unknown_model_raises_error(self, monkeypatch):
        """Unknown non-path names no longer raise a ValueError up front; they
        route to AIMNet2Adapter, which raises only when the aimnet registry
        cannot resolve the name. Patch the adapter to raise so we exercise the
        propagation path without touching the network."""
        from Auto3D import model_factory

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("unresolvable registry name")

        monkeypatch.setattr(model_factory, "AIMNet2Adapter", _Boom)
        with pytest.raises(Exception):
            ModelFactory.create(
                "totally-not-a-real-model-xyz",
                device=torch.device("cpu"),
                use_cache=False,
            )

    def test_create_is_case_insensitive_for_builtins(self, monkeypatch):
        """Built-in routing is case-insensitive: any casing of 'aimnet' routes
        to AIMNet2Adapter('aimnet2'); any casing of 'ani2x' routes to the ANI
        adapter. Registry/path names themselves are case-preserving."""
        from Auto3D import model_factory

        captured = {}

        class _FakeAIMNet2Adapter:
            def __init__(self, model_name, device, **kw):
                captured["aimnet"] = model_name

        monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)

        for alias in ("aimnet", "AImNeT", "AIMNET"):
            captured.clear()
            model_factory.ModelFactory.create(
                alias, device=torch.device("cpu"), use_cache=False
            )
            assert captured["aimnet"] == "aimnet2"

        # ANI engines resolve case-insensitively to their adapter class.
        assert model_factory.ModelFactory._adapters["ani2x".upper()] is \
            model_factory.ModelFactory._adapters["ANI2X"]
        assert "ANI2XT" in model_factory.ModelFactory._adapters

    @pytest.mark.slow
    def test_create_aimnet_returns_aimnet2_adapter(self, aimnet_model):
        """create('AIMNET') builds an AIMNet2Adapter bound to the 'aimnet2'
        registry name (no bundled .jpt path anymore).

        Reuses the session-scoped ``aimnet_model`` fixture (itself built via
        ``create_model("AIMNET", ...)`` -> ``ModelFactory.create``) so the real
        NNP is loaded once per session instead of an extra ~4s load here.
        """
        from Auto3D.models.adapter import AIMNet2Adapter

        assert isinstance(aimnet_model, AIMNet2Adapter)
        assert aimnet_model.model_name == "aimnet2"

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

    @patch("Auto3D.model_factory.torch.cuda.device_count")
    @patch("Auto3D.model_factory.torch.cuda.is_available")
    def test_get_device_cuda_when_available(self, mock_cuda, mock_count):
        """Test that CUDA device is returned when available.

        `device_count` must be patched alongside `is_available`, not left to
        the host: `get_device` now range-checks `gpu_idx`, and a CI runner with
        no CUDA reports `device_count() == 0`, so an unpatched version of this
        test asks for `cuda:1` out of zero devices and (correctly) raises
        `GPUError`. It passed on this 8-device dev box and would have gone red
        on CI -- exactly the shape of failure the bounds check must not
        introduce.
        """
        mock_cuda.return_value = True
        mock_count.return_value = 4
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

    @pytest.mark.slow
    def test_factory_returns_adapter(self, aimnet_model):
        """Factory should return ModelAdapter instances."""
        # Reuse the session-scoped aimnet_model fixture (one shared load that
        # survives ModelFactory.clear_cache()) instead of a fresh create_model.
        model = aimnet_model

        # Check it's an adapter with the right interface
        assert hasattr(model, 'coord_pad')
        assert hasattr(model, 'species_pad')
        assert hasattr(model, 'forward')
        assert model.coord_pad == 0.0
        assert model.species_pad == 0

    @pytest.mark.slow
    def test_factory_returns_aimnet_adapter(self, aimnet_model):
        """Factory should return an AIMNet2Adapter for AIMNET."""
        from Auto3D.models.adapter import AIMNet2Adapter

        # Reuse the session-scoped aimnet_model fixture (one shared load).
        model = aimnet_model

        assert isinstance(model, AIMNet2Adapter)
        assert model.model_name == "aimnet2"

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


def test_aimnet_alias_routes_to_aimnet2(monkeypatch):
    import torch
    from Auto3D import model_factory
    captured = {}

    class _FakeAIMNet2Adapter:
        def __init__(self, model_name, device, **kw):
            captured["model_name"] = model_name
    monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)
    model_factory.ModelFactory.clear_cache()
    model_factory.create_model("AIMNET", torch.device("cpu"), use_cache=False)
    assert captured["model_name"] == "aimnet2"


def test_registry_name_routes_to_aimnet2(monkeypatch):
    import torch
    from Auto3D import model_factory
    captured = {}

    class _FakeAIMNet2Adapter:
        def __init__(self, model_name, device, **kw):
            captured["model_name"] = model_name
    monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)
    model_factory.ModelFactory.clear_cache()
    model_factory.create_model("aimnet2-2025", torch.device("cpu"), use_cache=False)
    assert captured["model_name"] == "aimnet2-2025"


def test_existing_path_routes_to_custom(tmp_path, monkeypatch):
    import torch
    from Auto3D import model_factory
    f = tmp_path / "my.pt"; f.write_text("x")
    captured = {}

    class _FakeCustom:
        def __init__(self, path, device, **kw):
            captured["path"] = path
    monkeypatch.setattr(model_factory, "CustomModelAdapter", _FakeCustom)
    model_factory.create_model(str(f), torch.device("cpu"), use_cache=False)
    assert captured["path"] == str(f)


def test_builtin_name_beats_colliding_file(tmp_path, monkeypatch):
    """Name resolution must win over Path.exists(): a file literally named
    after a built-in engine (e.g. a stray "ANI2xt" left in the working
    directory) must still resolve to the built-in adapter, not be silently
    loaded as a custom NNP.

    Auto3D.ASE.thermo._load_hessian_model routes ANI2xt/ANI2x through this
    same ModelFactory.create dispatch, and Auto3D.ASE.thermo.
    aimnet_hessian_helper (which receives the same model_name string
    downstream) resolves by name first. If Path.exists() won here instead,
    the colliding file would be loaded as a CustomModelAdapter and then be
    called with ANI2xt's 2-argument calling convention -- wrong results, not
    an error naming the mismatch.
    """
    import torch
    from Auto3D import model_factory

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ANI2xt").write_text(
        "colliding file; must not be loaded as a custom NNP"
    )

    def _boom(path, device, **kw):
        raise AssertionError(
            f"colliding file at {path!r} was routed to CustomModelAdapter; "
            "a built-in engine name must resolve before Path.exists()."
        )
    monkeypatch.setattr(model_factory, "CustomModelAdapter", _boom)

    captured = {}

    class _FakeANI2xtAdapter:
        def __init__(self, device, **kw):
            captured["built_in"] = True

    monkeypatch.setitem(model_factory.ModelFactory._adapters, "ANI2XT", _FakeANI2xtAdapter)
    model_factory.ModelFactory.clear_cache()

    result = model_factory.create_model("ANI2xt", torch.device("cpu"), use_cache=False)

    assert captured.get("built_in") is True
    assert isinstance(result, _FakeANI2xtAdapter)


class TestRemovedParameters:
    """use_ensemble and **kwargs were dead and are gone in 4.0.

    These assert against ``create_model``'s call signature via
    ``inspect.signature(...).bind(...)`` rather than calling ``create_model``
    directly. Before the fix, both keywords are silently accepted (the second
    via **kwargs) and the call falls through to actually loading a real
    AIMNet2 model -- unsafe on this shared-GPU box with 8 shared CUDA devices.
    ``Signature.bind`` raises the exact same ``TypeError`` a real call would
    raise at argument-binding time (before the function body ever runs), so
    this is behaviorally equivalent to ``pytest.raises(TypeError): create_model(...)``
    without ever entering the function body or touching a model.
    """

    def test_use_ensemble_is_rejected(self):
        """Passing the removed parameter must fail loudly, not be ignored."""
        import inspect

        import torch

        from Auto3D.model_factory import create_model

        sig = inspect.signature(create_model)
        with pytest.raises(TypeError):
            sig.bind("AIMNET", torch.device("cpu"), use_ensemble=True)

    def test_unknown_kwarg_is_rejected(self):
        """**kwargs previously swallowed typos silently."""
        import inspect

        import torch

        from Auto3D.model_factory import create_model

        sig = inspect.signature(create_model)
        with pytest.raises(TypeError):
            sig.bind("AIMNET", torch.device("cpu"), use_ensembel=True)
