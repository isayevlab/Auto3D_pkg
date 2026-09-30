"""Unit tests for the ModelFactory module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from Auto3D.engines.model_factory import ModelFactory, create_model, get_device
from Auto3D.engines.models.contract import ModelAdapter


def test_the_factory_promises_the_contract_not_the_base_class():
    """Every factory signature must be annotated with the Protocol.

    ``ModelAdapter`` was ``@runtime_checkable`` and published, while every
    signature that wanted "an adapter" said ``BaseModelAdapter`` (the ABC)
    instead -- so the contract the factory actually honors was invisible in its
    own types, and production quietly accepted structural implementations the
    annotation excluded. Reading ``__annotations__`` (strings, because the module
    uses ``from __future__ import annotations``) is the only way to observe it.
    """
    import inspect

    from Auto3D.engines import model_factory

    assert inspect.get_annotations(model_factory.create_model)["return"] == "ModelAdapter"
    assert inspect.get_annotations(ModelFactory.create.__func__)["return"] == "ModelAdapter"
    # BaseModelAdapter survives in exactly one type position: a registry of
    # Auto3D's OWN adapter classes, which really is the concrete base.
    assert (
        inspect.get_annotations(ModelFactory)["_engines"]
        == "Registry[type[BaseModelAdapter] | None]"
    )
    # ...and it is no longer an incidental runtime re-export of this module.
    assert not hasattr(model_factory, "BaseModelAdapter")


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
        # The registry now holds every user-facing engine name; the ones built
        # from a local adapter class are those whose entry carries one. AIMNET
        # is registered (it is a name a user may pass) but resolves to None,
        # because it is looked up through the aimnet registry instead.
        built_locally = {
            n.upper()
            for n in ModelFactory._engines.available()
            if ModelFactory._engines.resolve(n) is not None
        }
        assert built_locally == {"ANI2X", "ANI2XT"}
        assert ModelFactory._engines.resolve("AIMNET") is None

    def test_create_unknown_model_raises_error(self, monkeypatch):
        """Unknown non-path names no longer raise a ValueError up front; they
        route to AIMNet2Adapter, which raises only when the aimnet registry
        cannot resolve the name. Patch the adapter to raise so we exercise the
        propagation path without touching the network."""
        from Auto3D.engines import model_factory

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("unresolvable registry name")

        monkeypatch.setattr(model_factory, "AIMNet2Adapter", _Boom)
        with pytest.raises(RuntimeError, match="unresolvable registry name"):
            ModelFactory.create(
                "totally-not-a-real-model-xyz",
                device=torch.device("cpu"),
                use_cache=False,
            )

    def test_create_is_case_insensitive_for_builtins(self, monkeypatch):
        """Built-in routing is case-insensitive: any casing of 'aimnet' routes
        to AIMNet2Adapter('aimnet2'); any casing of 'ani2x' routes to the ANI
        adapter. Registry/path names themselves are case-preserving."""
        from Auto3D.engines import model_factory

        captured = {}

        class _FakeAIMNet2Adapter:
            def __init__(self, model_name, device, **kw):
                captured["aimnet"] = model_name

        monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)

        for alias in ("aimnet", "AImNeT", "AIMNET"):
            captured.clear()
            model_factory.ModelFactory.create(alias, device=torch.device("cpu"), use_cache=False)
            assert captured["aimnet"] == "aimnet2"

        # ANI engines resolve case-insensitively to their adapter class.
        assert model_factory.ModelFactory._engines.resolve(
            "ani2x"
        ) is model_factory.ModelFactory._engines.resolve("ANI2X")
        assert "ANI2XT" in model_factory.ModelFactory._engines

    @pytest.mark.slow
    def test_create_aimnet_returns_aimnet2_adapter(self, aimnet_model):
        """create('AIMNET') builds an AIMNet2Adapter bound to the 'aimnet2'
        registry name (no bundled .jpt path anymore).

        Reuses the session-scoped ``aimnet_model`` fixture (itself built via
        ``create_model("AIMNET", ...)`` -> ``ModelFactory.create``) so the real
        NNP is loaded once per session instead of an extra ~4s load here.
        """
        from Auto3D.engines.models.adapter import AIMNet2Adapter

        assert isinstance(aimnet_model, AIMNet2Adapter)
        assert aimnet_model.model_name == "aimnet2"

    @patch.object(Path, "exists")
    @patch.object(torch.jit, "load")
    def test_create_custom_model_from_path(self, mock_load, mock_exists):
        """Test that custom model paths are loaded correctly."""
        from Auto3D.engines.models.adapter import CustomModelAdapter

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

    @patch.object(torch.cuda, "is_available")
    def test_get_device_cpu_when_cuda_unavailable(self, mock_cuda):
        """Test that CPU is returned when CUDA is unavailable."""
        mock_cuda.return_value = False
        device = get_device(gpu_idx=0, use_gpu=True)
        assert device == torch.device("cpu")

    @patch.object(torch.cuda, "device_count")
    @patch.object(torch.cuda, "is_available")
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

    @patch.object(torch.cuda, "is_available")
    def test_get_device_cuda_default_index(self, mock_cuda):
        """Test that CUDA:0 is returned by default."""
        mock_cuda.return_value = True
        device = get_device(gpu_idx=None, use_gpu=True)
        assert device == torch.device("cuda:0")


class TestFactoryReturnsAdapter:
    """Tests for ModelFactory returning adapter instances."""

    @pytest.mark.slow
    def test_factory_returns_adapter(self, aimnet_model):
        """Factory should return ModelAdapter instances."""
        # Reuse the session-scoped aimnet_model fixture (one shared load that
        # survives ModelFactory.clear_cache()) instead of a fresh create_model.
        model = aimnet_model

        # Check it's an adapter with the right interface
        assert hasattr(model, "coord_pad")
        assert hasattr(model, "species_pad")
        assert hasattr(model, "forward")
        assert model.coord_pad == 0.0
        assert model.species_pad == 0

    @pytest.mark.slow
    def test_factory_returns_aimnet_adapter(self, aimnet_model):
        """Factory should return an AIMNet2Adapter for AIMNET."""
        from Auto3D.engines.models.adapter import AIMNet2Adapter

        # Reuse the session-scoped aimnet_model fixture (one shared load).
        model = aimnet_model

        assert isinstance(model, AIMNet2Adapter)
        assert model.model_name == "aimnet2"

    def test_factory_returns_ani2xt_adapter(self):
        """Factory should return ANI2xtAdapter for ANI2xt."""
        pytest.importorskip("torchani")
        from Auto3D.engines.models.adapter import ANI2xtAdapter

        device = torch.device("cpu")
        model = create_model("ANI2xt", device)

        assert isinstance(model, ANI2xtAdapter)
        assert model.species_pad == -1

    def test_factory_returns_ani2x_adapter(self):
        """Factory should return ANI2xAdapter for ANI2x."""
        pytest.importorskip("torchani")
        from Auto3D.engines.models.adapter import ANI2xAdapter

        device = torch.device("cpu")
        model = create_model("ANI2x", device)

        assert isinstance(model, ANI2xAdapter)
        assert model.species_pad == -1

    @patch.object(Path, "exists")
    @patch.object(torch.jit, "load")
    def test_factory_returns_custom_adapter(self, mock_load, mock_exists):
        """Factory should return CustomModelAdapter for custom model paths."""
        from Auto3D.engines.models.adapter import CustomModelAdapter

        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        mock_model.coord_pad = 1.5
        mock_model.species_pad = -2
        # `load_custom_nnp` puts everything it returns in eval mode; a real
        # nn.Module answers .eval() with itself, so the double must too.
        mock_model.eval.return_value = mock_model
        mock_load.return_value = mock_model

        device = torch.device("cpu")
        model = create_model("/path/to/custom_model.pt", device)

        assert isinstance(model, CustomModelAdapter)
        assert model.coord_pad == 1.5
        assert model.species_pad == -2


def test_aimnet_alias_routes_to_aimnet2(monkeypatch):
    import torch

    from Auto3D.engines import model_factory

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

    from Auto3D.engines import model_factory

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

    from Auto3D.engines import model_factory

    f = tmp_path / "my.pt"
    f.write_text("x")
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

    Auto3D.entry.ASE.thermo._load_hessian_model routes ANI2xt/ANI2x through this
    same ModelFactory.create dispatch, and Auto3D.entry.ASE.thermo.
    aimnet_hessian_helper (which receives the same model_name string
    downstream) resolves by name first. If Path.exists() won here instead,
    the colliding file would be loaded as a CustomModelAdapter and then be
    called with ANI2xt's 2-argument calling convention -- wrong results, not
    an error naming the mismatch.
    """
    import torch

    from Auto3D.engines import model_factory

    monkeypatch.chdir(tmp_path)
    (tmp_path / "ANI2xt").write_text("colliding file; must not be loaded as a custom NNP")

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

    monkeypatch.setitem(
        model_factory.ModelFactory._engines._entries,
        "ANI2xt",
        model_factory.ModelFactory._engines.entry("ANI2xt").__class__(
            name="ANI2xt", value=_FakeANI2xtAdapter
        ),
    )
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

        from Auto3D.engines.model_factory import create_model

        sig = inspect.signature(create_model)
        with pytest.raises(TypeError):
            sig.bind("AIMNET", torch.device("cpu"), use_ensemble=True)

    def test_unknown_kwarg_is_rejected(self):
        """**kwargs previously swallowed typos silently."""
        import inspect

        import torch

        from Auto3D.engines.model_factory import create_model

        sig = inspect.signature(create_model)
        with pytest.raises(TypeError):
            sig.bind("AIMNET", torch.device("cpu"), use_ensembel=True)


def test_the_compile_probes_own_frames_are_excluded_from_later_warnings(
    tmp_path, monkeypatch, caplog
):
    """T-2: the probe IS the adapter's first forward, so it owns the first frames.

    ``verify_compiled_adapter`` triggers the lazy compilation itself, which means
    every frame that compilation attempts -- including one Dynamo suppresses --
    falls inside the delta ``_warn_if_compile_fell_back_to_eager`` measures from
    ``__init__``'s snapshot. Without re-baselining after the probe, the FIRST
    forward the caller runs reports the probe's fallback as if the caller's own
    geometry had provoked it, and (because the count is then already 1) the
    caller's real fallback is never reported at all.
    """
    import collections
    import logging

    import Auto3D.engines.models.adapter as adapter_mod
    from tests.helpers_custom_nnp import ScriptableNNP

    path = tmp_path / "eager_nnp.pt"
    torch.save(ScriptableNNP(), str(path))

    frames: collections.Counter = collections.Counter()
    monkeypatch.setitem(torch._dynamo.utils.counters, "frames", frames)

    class _SuppressedOnFirstForward(torch.nn.Module):
        """Numerically identical to the eager module, but bookkeeps like a
        compile whose one frame Dynamo suppressed on the first call."""

        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner
            self._seen = False

        def forward(self, *args, **kwargs):
            if not self._seen:
                self._seen = True
                frames["total"] += 1  # ... and deliberately no "ok"
            return self._orig_mod(*args, **kwargs)

    monkeypatch.setattr(
        adapter_mod.torch, "compile", lambda obj, **kw: _SuppressedOnFirstForward(obj)
    )

    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        adapter = create_model(str(path), torch.device("cpu"), compile_model=True, use_cache=False)
    # The probe forward is the one that saw the suppression, so it is the one
    # that reports it -- at construction, where it belongs.
    assert "fell back to eager" in caplog.text

    # ...and the adapter is handed back re-baselined, so the caller starts clean.
    assert adapter._compile_suppressed_seen == 0
    assert adapter._compile_frame_stats_before == {"total": 1}

    coords = torch.zeros(1, 2, 3)
    species = torch.ones(1, 2, dtype=torch.long)
    charges = torch.zeros(1)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        adapter.forward(coords, species, charges)
    assert caplog.text == "", "the probe's own frame was blamed on the caller"

    # A fallback the caller provokes IS reported -- the re-baseline must not
    # have cost the diagnostic it exists to make accurate, which is what
    # resetting _compile_suppressed_seen alongside the snapshot buys.
    frames["total"] += 1
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        adapter.forward(coords, species, charges)
    assert "fell back to eager" in caplog.text
