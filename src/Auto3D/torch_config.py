# src/Auto3D/torch_config.py
"""Centralized PyTorch configuration for Auto3D.

This module provides a centralized way to configure PyTorch backend settings
like TF32 precision and cuDNN benchmark mode. The settings are applied
globally to the PyTorch backends.

**These settings are process-global state the caller may already own.** Every
Auto3D entry point (``main``, ``smiles2mols``, ``calc_spe``, ``opt_geometry``,
``calc_thermo``) calls :func:`configure_torch` on the way in, so anything this
module writes unconditionally is written into *the caller's* process and stays
that way after Auto3D returns. Only ``allow_tf32`` is written unconditionally,
because it is a real Auto3D option with a documented default
(``Auto3DOptions.allow_tf32``, ``--allow-tf32``) that the user chose by
choosing Auto3D's default. ``cudnn_benchmark`` and ``deterministic`` have no
Auto3D-level option, so they default to ``None`` = "leave the process's
setting exactly as it was". A script that calls
``torch.use_deterministic_algorithms(True)`` before Auto3D keeps determinism
afterwards; before this, Auto3D turned it off silently and offered no way to
ask for it back.

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
                   **Always applied**, in both directions: it is a documented
                   Auto3D option and False is the answer Auto3D's default gives.
        cudnn_benchmark: Enable the cuDNN autotuner. When True, cuDNN benchmarks
                        multiple convolution algorithms and selects the fastest;
                        best for fixed-size inputs. ``None`` (the default) leaves
                        ``torch.backends.cudnn.benchmark`` exactly as the calling
                        process set it.
        deterministic: Enable deterministic algorithms. ``True`` requests them,
                      ``False`` explicitly turns them off (so a run that enabled
                      them can restore fast mode), and ``None`` -- the default --
                      leaves both ``torch.use_deterministic_algorithms`` and
                      ``torch.backends.cudnn.deterministic`` untouched. Auto3D
                      itself never requests either value, so an entry point that
                      builds ``TorchConfig(allow_tf32=...)`` cannot disturb a
                      caller who configured determinism for reproducibility.
        deterministic_warn_only: The ``warn_only`` flag handed to
                      ``torch.use_deterministic_algorithms`` when
                      ``deterministic`` is not None. Defaults to True because
                      AIMNet2/ANI scatter and masked index-put ops have no
                      deterministic CUDA kernel and would otherwise raise, which
                      aborts the very optimization loop determinism is meant to
                      make reproducible. Pass False to have PyTorch raise on a
                      nondeterministic op instead -- the request is then honored
                      rather than downgraded to a warning.
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
    cudnn_benchmark: bool | None = None
    deterministic: bool | None = None
    deterministic_warn_only: bool = True
    random_seed: int | None = None


def configure_torch(config: TorchConfig | None = None) -> None:
    """Apply PyTorch configuration settings.

    This function configures global PyTorch backend settings based on the
    provided configuration. It should be called early in the application
    lifecycle, before any GPU computations are performed.

    Only what the config actually asks for is written: fields left at ``None``
    (``cudnn_benchmark``, ``deterministic``) leave the corresponding
    process-global flag alone. See the module docstring for why.

    Args:
        config: Configuration object. If None, uses default TorchConfig
                which disables TF32 for maximum precision and touches
                nothing else.

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

    # Precision settings. Set both the legacy allow_tf32 booleans (back-compat
    # for torch < 2.9) and the modern fp32_precision knob (canonical on torch
    # >= 2.9, where allow_tf32 is deprecated). "ieee" = full FP32, "tf32" = TF32.
    fp32_mode = "tf32" if config.allow_tf32 else "ieee"
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = fp32_mode
    if hasattr(torch.backends.cudnn, "fp32_precision"):
        torch.backends.cudnn.fp32_precision = fp32_mode
    if config.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # Reproducibility settings
    if config.random_seed is not None:
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
        # Also seed numpy for RDKit and other operations
        import numpy as np
        np.random.seed(config.random_seed)

    # Written only when the caller says which way they want it. An explicit
    # False still turns determinism back off (these flags used to be
    # write-once-sticky, so a process that enabled a reproducible run could
    # never restore fast mode), but the default None no longer reaches in and
    # disables determinism the caller set for their own reasons.
    if config.deterministic is not None:
        torch.use_deterministic_algorithms(
            config.deterministic, warn_only=config.deterministic_warn_only
        )
        torch.backends.cudnn.deterministic = config.deterministic


# Note: We intentionally do NOT apply any default configuration on module import.
# This allows users to control when and how the configuration is applied,
# and prevents unexpected side effects from importing the module.
