"""Factory for creating neural network potential models."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from Auto3D.constants import (
    BUILTIN_ANI_MODELS,
    DEFAULT_AIMNET_MODEL,
    MODEL_AIMNET,
    MODEL_ANI2X,
    MODEL_ANI2XT,
)
from Auto3D.models.adapter import (
    AIMNet2Adapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)

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

    # Built-in (non-aimnet) engines kept for back-compat; keys are exactly the
    # members of BUILTIN_ANI_MODELS.
    _adapters: dict[str, type[BaseModelAdapter]] = {
        MODEL_ANI2XT.upper(): ANI2xtAdapter,
        MODEL_ANI2X.upper(): ANI2xAdapter,
    }
    assert set(_adapters) == set(BUILTIN_ANI_MODELS)

    # Model instance cache: key = (name, device_str, compile_model)
    _cache: dict[tuple[str, str, bool], BaseModelAdapter] = {}

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
    ) -> BaseModelAdapter:
        """Create a model adapter by name.

        Args:
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
            device: Target device for the model.
            compile_model: Whether to use torch.compile() for optimization.
                If None, checks AUTO3D_COMPILE_MODEL environment variable.
            use_cache: Whether to cache and reuse model instances. Default True.
                Set False to force creating a new model instance.

        Returns:
            Initialized model adapter on the specified device.

        Raises:
            ValueError: If model name is not recognized and not a valid path.
        """
        if device is None:
            device = torch.device("cpu")

        # Check environment variable if compile_model not explicitly set
        if compile_model is None:
            compile_model = os.environ.get(_COMPILE_ENV_VAR, "").lower() in ("1", "true", "yes")

        # 1. Existing path on disk -> custom NNP (file/custom model selection).
        if Path(name).exists():
            return CustomModelAdapter(name, device, compile_model=compile_model)

        name_upper = name.upper()

        # 2. Built-in ANI engines.
        if name_upper in cls._adapters:
            cache_key = (name_upper, str(device), compile_model)
            if use_cache and cache_key in cls._cache:
                return cls._cache[cache_key]
            adapter = cls._adapters[name_upper](device, compile_model=compile_model)
            if use_cache:
                cls._cache[cache_key] = adapter
            return adapter

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
        """Return list of registered model names."""
        return [MODEL_AIMNET, "aimnet2-2025", "aimnet2-nse", "aimnet2-pd", MODEL_ANI2X, MODEL_ANI2XT]


def create_model(
    name: str,
    device: torch.device | None = None,
    compile_model: bool | None = None,
    use_cache: bool = True,
) -> BaseModelAdapter:
    """Convenience function to create a model adapter.

    Args:
        name: Model name or path to custom model.
        device: Target device (default: CPU).
        compile_model: Whether to use torch.compile() for optimization.
            If None, checks AUTO3D_COMPILE_MODEL environment variable.
        use_cache: Whether to cache and reuse model instances. Default True.

    Returns:
        Initialized model adapter.

    Example:
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"))  # Fast single model (default)
        >>> # Enable torch.compile for ANI models
        >>> model = create_model("ANI2xt", device=torch.device("cuda:0"), compile_model=True)
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """
    return ModelFactory.create(
        name, device, compile_model=compile_model, use_cache=use_cache
    )


def get_device(gpu_idx: int | None = None, use_gpu: bool = True) -> torch.device:
    """Get the appropriate torch device.

    Args:
        gpu_idx: GPU index to use. None for automatic selection.
        use_gpu: Whether to use GPU if available.

    Returns:
        torch.device for the selected device.
    """
    if use_gpu and torch.cuda.is_available():
        if gpu_idx is not None:
            return torch.device(f"cuda:{gpu_idx}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def is_custom_model(name: str) -> bool:
    """Check if a model name refers to a custom user model.

    Args:
        name: Model name or path.

    Returns:
        True if the name is a path to an existing file.
    """
    return Path(name).exists()
