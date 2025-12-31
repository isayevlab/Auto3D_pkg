"""Factory for creating neural network potential models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from Auto3D.constants import MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT
from Auto3D.models.adapter import (
    AIMNetAdapter,
    ANI2xAdapter,
    ANI2xtAdapter,
    BaseModelAdapter,
    CustomModelAdapter,
)

# Environment variable to enable torch.compile() by default
_COMPILE_ENV_VAR = "AUTO3D_COMPILE_MODEL"
# Environment variable to disable ensemble for faster optimization
_ENSEMBLE_ENV_VAR = "AUTO3D_USE_ENSEMBLE"


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

    _adapters: dict[str, type[BaseModelAdapter]] = {
        MODEL_AIMNET.upper(): AIMNetAdapter,
        MODEL_ANI2XT.upper(): ANI2xtAdapter,
        MODEL_ANI2X.upper(): ANI2xAdapter,
    }

    # Model instance cache: key = (name, device_str, use_ensemble, compile_model)
    _cache: dict[tuple[str, str, bool, bool], BaseModelAdapter] = {}

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
        use_ensemble: bool | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """Create a model adapter by name.

        Args:
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
            device: Target device for the model.
            compile_model: Whether to use torch.compile() for optimization.
                If None, checks AUTO3D_COMPILE_MODEL environment variable.
            use_ensemble: For AIMNET only: whether to use ensemble model (default False).
                Single model is ~35x faster and accurate enough for geometry optimization.
                Set True for highest accuracy. If None, checks AUTO3D_USE_ENSEMBLE env var.
            use_cache: Whether to cache and reuse model instances. Default True.
                Set False to force creating a new model instance.
            **kwargs: Additional arguments passed to the adapter constructor.

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

        # Check environment variable for use_ensemble (default False for speed)
        if use_ensemble is None:
            env_val = os.environ.get(_ENSEMBLE_ENV_VAR, "").lower()
            use_ensemble = env_val in ("1", "true", "yes") if env_val else False

        name_upper = name.upper()

        # Check cache first (only for standard models, not custom paths)
        cache_key = (name_upper, str(device), use_ensemble, compile_model)
        if use_cache and name_upper in cls._adapters and cache_key in cls._cache:
            return cls._cache[cache_key]

        # Create new model adapter
        adapter: BaseModelAdapter
        if name_upper in cls._adapters:
            # Pass use_ensemble only to AIMNET adapter
            if name_upper == MODEL_AIMNET.upper():
                adapter = cls._adapters[name_upper](
                    device, compile_model=compile_model, use_ensemble=use_ensemble, **kwargs
                )
            else:
                adapter = cls._adapters[name_upper](device, compile_model=compile_model, **kwargs)

            # Cache the adapter
            if use_cache:
                cls._cache[cache_key] = adapter

            return adapter

        if Path(name).exists():
            return CustomModelAdapter(name, device, compile_model=compile_model)

        raise ValueError(
            f"Model '{name}' not found. Available models: {list(cls._adapters.keys())}. "
            f"Or provide a path to a custom NNP model file."
        )

    @classmethod
    def available_models(cls) -> list[str]:
        """Return list of registered model names."""
        return list(cls._adapters.keys())


def create_model(
    name: str,
    device: torch.device | None = None,
    compile_model: bool | None = None,
    use_ensemble: bool | None = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> BaseModelAdapter:
    """Convenience function to create a model adapter.

    Args:
        name: Model name or path to custom model.
        device: Target device (default: CPU).
        compile_model: Whether to use torch.compile() for optimization.
            If None, checks AUTO3D_COMPILE_MODEL environment variable.
        use_ensemble: For AIMNET: use ensemble model (default False).
            Single model is ~35x faster. Set True for highest accuracy.
        use_cache: Whether to cache and reuse model instances. Default True.
        **kwargs: Additional model arguments.

    Returns:
        Initialized model adapter.

    Example:
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"))  # Fast single model (default)
        >>> # Use ensemble for highest accuracy (slower)
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"), use_ensemble=True)
        >>> # Enable torch.compile for ANI models
        >>> model = create_model("ANI2xt", device=torch.device("cuda:0"), compile_model=True)
        >>> # Clear cache when done
        >>> ModelFactory.clear_cache()
    """
    return ModelFactory.create(
        name, device, compile_model=compile_model, use_ensemble=use_ensemble,
        use_cache=use_cache, **kwargs
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
