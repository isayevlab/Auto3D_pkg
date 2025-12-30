"""Factory for creating neural network potential models."""
from __future__ import annotations

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


class ModelFactory:
    """Factory for creating and managing NNP model adapters.

    Centralizes model creation logic to eliminate code duplication
    across batchopt.py, SPE.py, and thermo.py. Returns adapter instances
    that provide a consistent interface for all model types.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("AIMNET", device=torch.device("cuda:0"))
        >>> # Or use the convenience function
        >>> model = create_model("ANI2x", device=torch.device("cpu"))
    """

    _adapters: dict[str, type[BaseModelAdapter]] = {
        MODEL_AIMNET.upper(): AIMNetAdapter,
        MODEL_ANI2XT.upper(): ANI2xtAdapter,
        MODEL_ANI2X.upper(): ANI2xAdapter,
    }

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """Create a model adapter by name.

        Args:
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
            device: Target device for the model.
            **kwargs: Additional arguments passed to the adapter constructor.

        Returns:
            Initialized model adapter on the specified device.

        Raises:
            ValueError: If model name is not recognized and not a valid path.
        """
        if device is None:
            device = torch.device("cpu")

        name_upper = name.upper()

        if name_upper in cls._adapters:
            return cls._adapters[name_upper](device, **kwargs)

        if Path(name).exists():
            return CustomModelAdapter(name, device)

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
    **kwargs: Any,
) -> BaseModelAdapter:
    """Convenience function to create a model adapter.

    Args:
        name: Model name or path to custom model.
        device: Target device (default: CPU).
        **kwargs: Additional model arguments.

    Returns:
        Initialized model adapter.

    Example:
        >>> model = create_model("AIMNET", device=torch.device("cuda:0"))
    """
    return ModelFactory.create(name, device, **kwargs)


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
