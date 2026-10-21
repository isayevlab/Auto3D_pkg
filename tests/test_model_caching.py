"""Tests for model caching in ModelFactory."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

import Auto3D.engines.model_factory as model_factory
from Auto3D.engines.model_factory import ModelFactory, create_model
from tests.helpers_custom_nnp import ScriptableNNP


class _SizedScriptableNNP(torch.nn.Module):
    """Contract-conforming and TorchScript-safe, with a parameter tensor
    whose size is a constructor argument -- so two saves can be made to
    differ in file SIZE on disk without differing in content otherwise, for
    the M2 same-mtime-different-size test below."""

    def __init__(self, width: int = 1):
        super().__init__()
        self.coord_pad: float = 0.0
        self.species_pad: int = -1
        self.scale = torch.nn.Parameter(torch.zeros(width))

    def forward(
        self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
    ) -> torch.Tensor:
        return (coords**2).sum(dim=(1, 2)) + self.scale.sum()


class _TinyLinearNNP(torch.nn.Module):
    """Contract-conforming custom NNP with one real parameter, so a
    ``double=True`` request's effect on ``create_model``'s returned adapter
    is observable by dtype (m3: this is never actually forwarded here, so the
    float32-coords-into-a-float64-layer mismatch the class docstring warns
    custom-NNP authors about never arises)."""

    def __init__(self):
        super().__init__()
        self.coord_pad = 0.0
        self.species_pad = -1
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, species, coords, charges):
        return self.weight * coords.pow(2).sum(dim=(1, 2))


class _FakeAIMNet2Adapter:
    """Lightweight stand-in for ``AIMNet2Adapter``.

    Mirrors the real constructor signature so ``ModelFactory.create`` builds
    its cache key (registry_name, device, compile_model) and stores/returns
    instances exactly as in production, but skips the multi-second real
    AIMNet2 model load. The caching dict, identity semantics, and
    ``get_cache_info()`` size are all exercised unchanged; only the expensive
    object construction is replaced with an instant one.
    """

    def __init__(
        self,
        model_name="aimnet2",
        device=None,
        compile_model=False,
    ):
        self.model_name = model_name
        self.device = device
        self.compile_model = compile_model


class TestModelCaching:
    """Tests for model caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        ModelFactory.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        ModelFactory.clear_cache()

    @pytest.fixture(autouse=True)
    def _fake_aimnet_adapter(self, monkeypatch):
        """Replace the real AIMNet2 adapter with an instant fake.

        Patches the ``AIMNet2Adapter`` symbol used by ``ModelFactory.create``
        so that ``create_model("AIMNET", ...)`` returns a cheap stub instead of
        loading the real ~4-7s NNP. The cache-key / cache-dict logic in
        ``create`` runs verbatim, so all identity and cache-size assertions stay
        meaningful. ANI-based tests ``importorskip("torchani")`` before reaching
        ``create_model`` and are unaffected by this patch.
        """
        monkeypatch.setattr(model_factory, "AIMNet2Adapter", _FakeAIMNet2Adapter)

    def test_same_model_returns_cached_instance(self):
        """Test that calling create_model with same args returns cached instance."""
        device = torch.device("cpu")
        model1 = create_model("AIMNET", device)
        model2 = create_model("AIMNET", device)
        assert model1 is model2  # Same instance from cache

    def test_different_models_create_different_instances(self):
        """Test that different model names create different instances."""
        pytest.importorskip("torchani")  # ANI2xt requires the optional ani extra
        device = torch.device("cpu")
        model_aimnet = create_model("AIMNET", device)
        model_ani2xt = create_model("ANI2xt", device)
        assert model_aimnet is not model_ani2xt

    def test_clear_cache_removes_models(self):
        """Test that clear_cache removes all cached models."""
        pytest.importorskip("torchani")  # ANI2xt requires the optional ani extra
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
        pytest.importorskip("torchani")  # ANI2xt requires the optional ani extra
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

    def test_custom_nnp_path_is_cached_and_invalidates_on_a_rewrite(self, tmp_path):
        """A custom-NNP path is cached by (resolved path, stamp, ...),
        so repeated calls reuse the instance -- and a rewrite of the file on
        disk, which changes its stamp, must not hand back the stale one."""
        path = tmp_path / "tiny.pt"
        torch.jit.script(ScriptableNNP()).save(str(path))
        cpu = torch.device("cpu")
        first = create_model(str(path), cpu)
        assert create_model(str(path), cpu) is first
        time.sleep(0.01)
        torch.jit.script(ScriptableNNP()).save(str(path))
        assert create_model(str(path), cpu) is not first

    def test_a_rewrite_evicts_the_stale_entry_instead_of_growing_the_cache(self, tmp_path):
        """M1: a plain ``cache[new_key] = adapter`` would leave the OLD
        (path, stale-stamp, ...) entry in place -- retained device memory on
        GPU until ``clear_cache()``. The rewrite must evict it, not merely
        shadow it, so the cache holds exactly one entry for this path."""
        path = tmp_path / "tiny.pt"
        torch.jit.script(ScriptableNNP()).save(str(path))
        cpu = torch.device("cpu")
        create_model(str(path), cpu)
        assert ModelFactory.get_cache_info()["size"] == 1
        time.sleep(0.01)
        torch.jit.script(ScriptableNNP()).save(str(path))
        create_model(str(path), cpu)
        assert ModelFactory.get_cache_info()["size"] == 1

    def test_a_same_tick_rewrite_is_still_detected_by_size(self, tmp_path, monkeypatch):
        """M2: mtime alone can miss a rewrite inside one filesystem tick on a
        coarse clock (NFS/CIFS); pin ``st_mtime_ns`` identical across both
        calls (standing in for that coarse clock) while the rewrite itself
        genuinely changes the on-disk SIZE (a bigger parameter tensor), so
        only the size term in the key can be what detects it."""
        path = tmp_path / "tiny.pt"
        torch.jit.script(_SizedScriptableNNP(width=1)).save(str(path))
        cpu = torch.device("cpu")
        real_stat = Path.stat
        pinned_mtime = real_stat(path).st_mtime_ns

        def _same_mtime_stat(self, *a, **k):
            result = real_stat(self, *a, **k)

            class _Stamped:
                st_mtime_ns = pinned_mtime
                st_size = result.st_size
                st_ino = result.st_ino

            return _Stamped()

        monkeypatch.setattr(Path, "stat", _same_mtime_stat)
        first = create_model(str(path), cpu)
        size_before = path.stat().st_size
        torch.jit.script(_SizedScriptableNNP(width=4096)).save(str(path))  # same mtime, bigger
        assert path.stat().st_size != size_before, "test premise: the rewrite must change size"
        assert create_model(str(path), cpu) is not first
        assert ModelFactory.get_cache_info()["size"] == 1

    def test_a_double_request_does_not_touch_the_cached_fp32_instance(self):
        """The cache key carries dtype, so a double=True request never
        returns (and never upcasts in place) the cached fp32 instance."""
        pytest.importorskip("torchani")
        cpu = torch.device("cpu")
        single = create_model("ANI2xt", cpu)
        double = create_model("ANI2xt", cpu, double=True)
        assert single is not double
        assert next(single.model.parameters()).dtype == torch.float32
        assert next(double.model.parameters()).dtype == torch.float64

    def test_a_double_request_on_a_custom_nnp_path_gets_its_own_float64_instance(self, tmp_path):
        """m3: double=True reaches the custom-NNP path too, through the same
        cache-key widening as the built-in engines, not just ANI2xt."""
        path = tmp_path / "tiny_linear.pt"
        torch.save(_TinyLinearNNP(), str(path))
        cpu = torch.device("cpu")

        single = create_model(str(path), cpu)
        double = create_model(str(path), cpu, double=True)

        assert single is not double
        assert next(single.model.parameters()).dtype == torch.float32
        assert next(double.model.parameters()).dtype == torch.float64
