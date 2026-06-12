"""Shared loader for user-provided custom NNP model files.

A single definition of the "TorchScript archive OR eager nn.Module" load
contract, used by every entry point that accepts a custom model path
(model_factory/CustomModelAdapter, input validation, and the thermo Hessian
path) so the accepted formats and error behavior stay identical everywhere.
"""
from __future__ import annotations

import torch

from Auto3D.exceptions import ModelLoadError


def load_custom_nnp(
    model_path: str,
    device: torch.device,
    *,
    double: bool = False,
) -> torch.nn.Module:
    """Load a custom NNP from a file as a TorchScript archive or eager nn.Module.

    Tries ``torch.jit.load`` first (legacy TorchScript archives); falls back to
    ``torch.load`` for eager ``nn.Module`` checkpoints saved with
    ``torch.save`` -- modern AIMNet2-based models are no longer
    ``torch.jit.script``-able. Always returns an ``nn.Module`` on ``device``;
    any failure (corrupt file, or a payload that is not an ``nn.Module`` such as
    a bare ``state_dict``) is raised as :class:`~Auto3D.exceptions.ModelLoadError`.

    ``weights_only=False`` is required to deserialize a whole ``nn.Module`` and
    executes code from the file; these are trusted, user-supplied local paths
    the caller explicitly selected as the optimizing engine.

    Args:
        model_path: Path to the model file.
        device: Target device for the loaded model.
        double: If True, cast the returned module to float64.

    Returns:
        The loaded model as an ``nn.Module`` on ``device``.

    Raises:
        ModelLoadError: If the file cannot be loaded as either supported form.
    """
    try:
        model: torch.nn.Module = torch.jit.load(model_path, map_location=device)
    except RuntimeError:
        # Not a TorchScript archive -> try an eager nn.Module checkpoint.
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:  # corrupt / unpicklable file
            raise ModelLoadError(
                f"Custom NNP at {model_path} could not be loaded as a TorchScript "
                f"archive or an eager nn.Module: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(model, torch.nn.Module):
            raise ModelLoadError(
                f"Custom NNP at {model_path} did not deserialize to an nn.Module "
                f"(got {type(model).__name__})."
            )
        model = model.to(device).eval()
    return model.double() if double else model
