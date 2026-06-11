# src/Auto3D/torch_config.py
"""Centralized PyTorch configuration for Auto3D.

This module provides a centralized way to configure PyTorch backend settings
like TF32 precision and cuDNN benchmark mode. The settings are applied
globally to the PyTorch backends.

Example:
    >>> from Auto3D.torch_config import TorchConfig, configure_torch
    >>> config = TorchConfig(allow_tf32=True)
    >>> configure_torch(config)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TorchConfig:
    """Configuration for PyTorch behavior.

    This dataclass controls PyTorch backend settings that affect computation
    precision, performance, and reproducibility.

    Attributes:
        allow_tf32: Enable TF32 for faster but less precise matmul on Ampere+ GPUs.
                   Default False for maximum precision in scientific computing.
                   TF32 provides ~3x speedup on Ampere GPUs with slightly reduced
                   precision (10-bit mantissa vs 23-bit for FP32).
        cudnn_benchmark: Enable cuDNN autotuner for potential speedups.
                        When True, cuDNN will benchmark multiple convolution
                        algorithms and select the fastest. Best for fixed-size
                        inputs. Default False.
        deterministic: Enable deterministic algorithms. When True, operations will
                      use deterministic implementations (may be slower). Required
                      for full reproducibility. Default False.
        random_seed: Random seed for reproducibility. When set, seeds PyTorch,
                    CUDA, and NumPy RNGs. Default None (no seeding).

    Example:
        >>> config = TorchConfig(allow_tf32=True, cudnn_benchmark=True)
        >>> configure_torch(config)
        >>>
        >>> # For reproducible runs
        >>> config = TorchConfig(deterministic=True, random_seed=42)
        >>> configure_torch(config)
    """

    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    deterministic: bool = False
    random_seed: int | None = None


def configure_torch(config: TorchConfig | None = None) -> None:
    """Apply PyTorch configuration settings.

    This function configures global PyTorch backend settings based on the
    provided configuration. It should be called early in the application
    lifecycle, before any GPU computations are performed.

    Args:
        config: Configuration object. If None, uses default TorchConfig
                which disables TF32 for maximum precision.

    Example:
        >>> from Auto3D.torch_config import TorchConfig, configure_torch
        >>>
        >>> # Enable TF32 for faster computation
        >>> configure_torch(TorchConfig(allow_tf32=True))
        >>>
        >>> # Disable TF32 (default, maximum precision)
        >>> configure_torch(TorchConfig(allow_tf32=False))
        >>>
        >>> # For reproducible runs
        >>> configure_torch(TorchConfig(deterministic=True, random_seed=42))
        >>>
        >>> # Use defaults
        >>> configure_torch(None)
    """
    if config is None:
        config = TorchConfig()

    # Precision settings
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # Reproducibility settings
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        # Also seed numpy for RDKit and other operations
        import numpy as np
        np.random.seed(config.random_seed)

    # Set deterministic flags unconditionally so a later configure_torch() with
    # deterministic=False actually turns determinism back off (previously these
    # were only ever set to True, making them write-once-sticky). warn_only=True
    # so AIMNet2/ANI scatter / masked index-put ops -- which have no
    # deterministic CUDA kernel -- warn instead of raising and aborting the very
    # optimization loop deterministic mode is meant to make reproducible.
    torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = config.deterministic


# Note: We intentionally do NOT apply any default configuration on module import.
# This allows users to control when and how the configuration is applied,
# and prevents unexpected side effects from importing the module.
