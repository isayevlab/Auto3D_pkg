"""Factory for creating neural network potential models."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from Auto3D.constants import MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT

if TYPE_CHECKING:
    pass


class ModelFactory:
    """Factory for creating and managing NNP models.

    Centralizes model creation logic to eliminate code duplication
    across batchopt.py, SPE.py, and thermo.py.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("AIMNET", device=torch.device("cuda:0"))
        >>> # Or use the convenience function
        >>> model = create_model("ANI2x", device=torch.device("cpu"))
    """

    _registry: dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        """Decorator to register a model creator function.

        Args:
            name: Model name to register.

        Returns:
            Decorator function.
        """
        def decorator(creator: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            cls._registry[name.upper()] = creator
            return creator
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a model by name.

        Args:
            name: Model name ('AIMNET', 'ANI2x', 'ANI2xt') or path to custom model.
            device: Target device for the model.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            Initialized model on the specified device.

        Raises:
            ValueError: If model name is not recognized and not a valid path.
        """
        if device is None:
            device = torch.device("cpu")

        name_upper = name.upper()

        # Check if it's a registered model
        if name_upper in cls._registry:
            return cls._registry[name_upper](device=device, **kwargs)

        # Check if it's a path to a custom model
        if Path(name).exists():
            return cls._load_custom_model(name, device, **kwargs)

        raise ValueError(
            f"Model '{name}' not found. Available models: {list(cls._registry.keys())}. "
            f"Or provide a path to a custom NNP model file."
        )

    @classmethod
    def _load_custom_model(
        cls,
        path: str,
        device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        """Load a custom user-provided model.

        Args:
            path: Path to the model file (.pt or .jpt).
            device: Target device.
            **kwargs: Additional arguments.

        Returns:
            Loaded model on the specified device.
        """
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model

    @classmethod
    def available_models(cls) -> list[str]:
        """Return list of registered model names."""
        return list(cls._registry.keys())


# Register built-in models
@ModelFactory.register(MODEL_AIMNET)
def _create_aimnet(device: torch.device, **kwargs: Any) -> nn.Module:
    """Create AIMNet2 model."""
    model_path = Path(__file__).resolve().parent / "models" / "aimnet2_wb97m-d3_ens.jpt"
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


@ModelFactory.register(MODEL_ANI2XT)
def _create_ani2xt(device: torch.device, **kwargs: Any) -> nn.Module:
    """Create ANI2xt model."""
    try:
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
    except ImportError as e:
        raise ImportError(
            "ANI2xt model requires torchani. Install with: pip install torchani"
        ) from e

    periodic_table_index = kwargs.get("periodic_table_index", False)
    return ANI2xt(device, periodic_table_index=periodic_table_index)


@ModelFactory.register(MODEL_ANI2X)
def _create_ani2x(device: torch.device, **kwargs: Any) -> nn.Module:
    """Create ANI2x model."""
    try:
        import torchani
    except ImportError as e:
        raise ImportError(
            "ANI2x model requires torchani. Install with: pip install torchani"
        ) from e

    return torchani.models.ANI2x(periodic_table_index=True).to(device)


def create_model(
    name: str,
    device: torch.device | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Convenience function to create a model.

    Args:
        name: Model name or path to custom model.
        device: Target device (default: CPU).
        **kwargs: Additional model arguments.

    Returns:
        Initialized model.

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
