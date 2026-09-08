"""Factory for creating neural network potential models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from Auto3D.engines.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    CustomModelAdapter,
)
from Auto3D.foundation.constants import (
    BUILTIN_ANI_MODELS,
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.foundation.exceptions import DependencyError, GPUError
from Auto3D.foundation.registry import Registry

if TYPE_CHECKING:
    # Annotation-only. Every signature here promises the CONTRACT
    # (Auto3D.engines.models.contract.ModelAdapter), not the implementation base class;
    # `_adapters` is the sole exception, because it really is a registry of
    # Auto3D's own classes. Kept behind TYPE_CHECKING so
    # `Auto3D.engines.model_factory.BaseModelAdapter` is no longer an incidental
    # re-export -- import it from Auto3D.engines.models.adapter, which defines it
    # (`Auto3D.engines.models` re-exports nothing).
    from Auto3D.engines.models.adapter import BaseModelAdapter
    from Auto3D.engines.models.contract import ModelAdapter

# Environment variable to enable torch.compile() by default
_COMPILE_ENV_VAR = "AUTO3D_COMPILE_MODEL"


class ModelFactory:
    """Factory for creating and managing NNP model adapters.

    Centralizes model creation logic to eliminate code duplication
    across batchopt.py, SPE.py, and thermo.py. Returns adapter instances
    that provide a consistent interface for all model types.

    Includes model caching to avoid reloading models from disk repeatedly.
    Use `clear_cache()` to free memory when models are no longer needed.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("AIMNET", device=torch.device("cuda:0"))
        >>> # Or use the convenience function
        >>> model = create_model("ANI2x", device=torch.device("cpu"))
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """

    #: Every engine name a user may pass, in the order `auto3d models list`
    #: shows them. The value is the adapter class for engines built from one
    #: locally, and ``None`` for names resolved through the aimnet registry --
    #: which is the distinction that made this three separate lists before.
    #:
    #: `_adapters` held only the two ANI engines, `available_models()` restated
    #: all six as a hand-written literal derived from nothing, and
    #: `cli/commands/models.py`'s `ENGINE_INFO` held six display entries keyed by
    #: the same names. Nothing connected them; they agreed by hand. Two of the
    #: three are collapsed here. `ENGINE_INFO` is the remaining one, and moving
    #: it into `info=` is the follow-up -- it belongs with the CLI changes rather
    #: than with this factory's.
    _engines: Registry[type[BaseModelAdapter] | None] = Registry(
        "optimizing engine", case_insensitive=True
    )
    _engines.register(MODEL_AIMNET, None)
    _engines.register("aimnet2-2025", None)
    _engines.register("aimnet2-nse", None)
    _engines.register("aimnet2-pd", None)
    _engines.register(MODEL_ANI2X, ANI2xAdapter)
    _engines.register(MODEL_ANI2XT, ANI2xtAdapter)

    # Model instance cache: key = (name, device_str, compile_model)
    _cache: dict[tuple[str, str, bool], ModelAdapter] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache to free memory.

        Call this at the end of a workflow to release GPU memory
        held by cached models.
        """
        cls._cache.clear()

    @classmethod
    def get_cache_info(cls) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dictionary with cache size information.
        """
        return {"size": len(cls._cache)}

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | None = None,
        compile_model: bool | None = None,
        use_cache: bool = True,
    ) -> ModelAdapter:
        """Create a model adapter by name.

        Args:
            name: ``AIMNET`` (alias for the registry default ``aimnet2``), any aimnet
                registry name (``aimnet2``, ``aimnet2-2025``, ``aimnet2-nse``,
                ``aimnet2-pd``, ...), ``ANI2x``, ``ANI2xt``, or a path to a
                custom NNP model file.
            device: Target device for the model.
            compile_model: Whether to use torch.compile() for optimization.
                If None, checks AUTO3D_COMPILE_MODEL environment variable.
            use_cache: Whether to cache and reuse model instances. Default True.
                Set False to force creating a new model instance. Has no
                effect for a custom-model path, which is always rebuilt.

        Returns:
            Initialized model adapter on the specified device.

        Raises:
            DependencyError: `name` is ANI2x/ANI2xt and `torchani` -- the
                optional dependency both adapters import lazily -- is not
                installed. Translated here rather than left as the raw
                ``ModuleNotFoundError`` because this is the single point every
                caller reaches: ``auto3d run`` got a ``DependencyError`` (exit
                3, "pip install torchani") from ``check_input``'s own probe,
                while ``auto3d energy``/``optimize``/``thermo``/``models test``
                never run ``check_input`` and so reported the identical
                environment problem as an "Unexpected Error" at exit 1 with no
                install hint at all.
        """
        if device is None:
            device = torch.device("cpu")

        # Check environment variable if compile_model not explicitly set
        if compile_model is None:
            compile_model = os.environ.get(_COMPILE_ENV_VAR, "").lower() in ("1", "true", "yes")

        name_upper = name.upper()

        # 1. Built-in ANI engines, checked by name FIRST. A reserved engine
        #    name (e.g. "ANI2xt") must always resolve to its built-in adapter,
        #    even if the working directory happens to contain a same-named
        #    file: name resolution cannot be hijacked by cwd contents, while a
        #    Path.exists() check can. (This used to be justified by agreement
        #    with Auto3D.entry.ASE.thermo.aimnet_hessian_helper, which resolved by
        #    name first; that helper is gone -- the Hessian path takes a
        #    ModelAdapter now -- so this factory is the only name resolver
        #    left, and the reason above stands on its own.)
        if name_upper in cls._engines and cls._engines.resolve(name_upper) is not None:
            cache_key = (name_upper, str(device), compile_model)
            if use_cache and cache_key in cls._cache:
                return cls._cache[cache_key]
            try:
                adapter_cls = cls._engines.resolve(name_upper)
                assert adapter_cls is not None  # guarded by the branch condition
                adapter = adapter_cls(device, compile_model=compile_model)
            except ImportError as exc:
                # Only translate the *absence of torchani itself*. A broken
                # torchani whose own transitive import fails names a different
                # module, and "Install: pip install torchani" would be a wrong
                # answer for it -- same judgment as `preflight_model`, which
                # leaves anything it cannot positively identify to propagate
                # with its own traceback rather than guessing a label.
                missing = getattr(exc, "name", None) or ""
                if missing != "torchani" and not missing.startswith("torchani."):
                    raise
                raise DependencyError(
                    f"{name} requires TorchANI, which is not installed.",
                    dependency_name="torchani",
                ) from exc
            if use_cache:
                cls._cache[cache_key] = adapter
            return adapter

        # 2. Existing path on disk -> custom NNP (file/custom model
        #    selection). Note: this branch returns before ever consulting
        #    cls._cache, so `use_cache` has no effect for a custom model path
        #    -- a fresh CustomModelAdapter is always created.
        if Path(name).exists():
            return CustomModelAdapter(name, device, compile_model=compile_model)

        # 3. Everything else -> aimnet registry name. "AIMNET" is the legacy
        #    alias for the registry default.
        registry_name = DEFAULT_AIMNET_MODEL if name_upper == MODEL_AIMNET.upper() else name
        cache_key = (registry_name, str(device), compile_model)
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]
        adapter = AIMNet2Adapter(registry_name, device, compile_model=compile_model)
        if use_cache:
            cls._cache[cache_key] = adapter
        return adapter

    @classmethod
    def available_models(cls) -> list[str]:
        """Registered engine names, in declaration order.

        Was a hand-written literal that restated the registry's contents and was
        derived from nothing -- so adding an engine and forgetting this list left
        it working but undiscoverable.
        """
        return cls._engines.available()


# The engines built from a local adapter class must be exactly
# BUILTIN_ANI_MODELS. This was an assert over a hand-maintained dict; it is now
# derived from the registry, so the two cannot drift apart. At module scope
# because a comprehension inside a class body cannot see that class's own
# attributes.
assert {
    name.upper()
    for name in ModelFactory._engines.available()
    if ModelFactory._engines.resolve(name) is not None
} == set(BUILTIN_ANI_MODELS)


def create_model(
    name: str,
    device: torch.device | None = None,
    compile_model: bool | None = None,
    use_cache: bool = True,
) -> ModelAdapter:
    """Convenience function to create a model adapter.

    Args:
        name: ``AIMNET``, any aimnet registry name, ``ANI2x``, ``ANI2xt``, or a
            path to a custom NNP model file. See ``available_models()``.
        device: Target device (default: CPU).
        compile_model: Whether to use torch.compile() for optimization.
            If None, checks AUTO3D_COMPILE_MODEL environment variable.
        use_cache: Whether to cache and reuse model instances. Default True.
            Has no effect for a custom-model path, which is always rebuilt.

    Returns:
        Initialized model adapter.

    Example:
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"))  # Fast single model (default)
        >>> # Enable torch.compile for ANI models
        >>> model = create_model("ANI2xt", device=torch.device("cuda:0"), compile_model=True)
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """
    return ModelFactory.create(name, device, compile_model=compile_model, use_cache=use_cache)


def get_device(gpu_idx: int | None = None, use_gpu: bool = True) -> torch.device:
    """Get the appropriate torch device.

    Args:
        gpu_idx: GPU index to use. None for automatic selection.
        use_gpu: Whether to use GPU if available.

    Returns:
        torch.device for the selected device.

    Raises:
        GPUError: `use_gpu` is True, CUDA is available, and `gpu_idx` names a
            device that does not exist. This function used to return
            ``torch.device("cuda:99")`` unchecked on an 8-device box, deferring
            the failure into CUDA itself -- a driver-level error raised at the
            first tensor move, far from the option that caused it, and mapped
            to the generic exit code 1 because it is not an ``Auto3DError``.
            ``check_valid_configuration`` already range-checked ``gpu_idx``,
            but only for ``main()``/``smiles2mols``; ``calc_spe``,
            ``opt_geometry``, ``calc_thermo`` and ``auto3d models test`` reach
            this function directly and had no bounds check at all. Raising here
            makes the documented "invalid GPU index -> exit 4" true for every
            entry point instead of only the one that validates its config.

            Only reachable when CUDA is present: with no visible device,
            ``check_gpu_requested`` has already refused ``use_gpu=True`` (also
            ``GPUError``, also exit 4) before any caller gets here, and
            ``use_gpu=False`` never consults ``gpu_idx``.
    """
    if use_gpu and torch.cuda.is_available():
        if gpu_idx is not None:
            device_count = torch.cuda.device_count()
            if not 0 <= gpu_idx < device_count:
                raise GPUError(
                    f"GPU index {gpu_idx} is invalid: {device_count} CUDA "
                    f"device(s) visible, so valid indices are "
                    f"0-{device_count - 1}.",
                    # The class hint ("Try --no-gpu ... or check CUDA
                    # installation") is a non-sequitur here: CUDA is installed
                    # and working, the index is simply out of range.
                    hint=("Pass a --gpu-idx within range, or --no-gpu to run on CPU."),
                )
            return torch.device(f"cuda:{gpu_idx}")
        return torch.device("cuda:0")
    return torch.device("cpu")
