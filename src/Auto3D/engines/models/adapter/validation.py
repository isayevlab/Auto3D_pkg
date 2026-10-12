"""Finiteness checks shared by every adapter's forward and energy-only path.

Cut from this package's single-file predecessor so
:mod:`Auto3D.engines.models.adapter.base` and
:mod:`Auto3D.engines.models.adapter.aimnet2` can both depend on it without
depending on each other. Imports nothing from the rest of the ``adapter``
package.
"""

from __future__ import annotations

import torch

from Auto3D.foundation.exceptions import NumericalError


def _raise_for_energy(energy: torch.Tensor) -> None:
    """Raise the NaN/Inf diagnosis for a known-non-finite energy tensor.

    Split out so the energy-only path
    (:func:`validate_energies`, reached from
    :meth:`Auto3D.engines.batch_opt.model_wrapper.BatchedEnergyForce.energy_batched`) and the
    energy-and-forces path (:func:`_validate_outputs`) emit the SAME message for
    the same defect instead of two near-identical copies. Returns normally if
    the energy is finite after all, leaving the caller to decide what that means.
    """
    if torch.isnan(energy).any():
        nan_count = torch.isnan(energy).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} energy value(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(energy).any():
        inf_count = torch.isinf(energy).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} energy value(s). "
            "This may indicate atomic clashes or numerical overflow."
        )


def validate_energies(energy: torch.Tensor) -> None:
    """Reject a non-finite energy on a path that computed no forces.

    This is the NaN gate a single-point energy path needs: without it, an
    energy-only path that skips ``forward``'s :func:`_validate_outputs`
    would turn ``auto3d energy``'s exit-5 diagnosis into an SDF full of ``nan``.

    Args:
        energy: Energy tensor, shape (batch,).

    Raises:
        NumericalError: If NaN or Inf values are detected.
    """
    # One combined reduction (one host-device sync) on the happy path, for the
    # same reason as _validate_outputs below.
    if bool(torch.isfinite(energy).all()):
        return
    _raise_for_energy(energy)


def _validate_outputs(energy: torch.Tensor, forces: torch.Tensor) -> None:
    """Validate model outputs for numerical stability.

    Checks for NaN and Inf values in energy and force tensors, raising
    an exception if numerical instability is detected.

    Args:
        energy: Energy tensor from model forward pass.
        forces: Force tensor from model forward pass.

    Raises:
        NumericalError: If NaN or Inf values are detected.
    """
    # This runs on every NN forward, i.e. every FIRE step. Each `.any()`/`.item()`
    # on a CUDA tensor is a host-device sync that serializes the stream, so the
    # happy path (finite outputs) does a SINGLE combined reduction. The detailed,
    # additionally-synchronizing NaN/Inf breakdown is computed only on the rare
    # failure branch, where one extra sync is irrelevant.
    if bool(torch.isfinite(energy).all() & torch.isfinite(forces).all()):
        return

    _raise_for_energy(energy)
    if torch.isnan(forces).any():
        nan_count = torch.isnan(forces).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} force component(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(forces).any():
        inf_count = torch.isinf(forces).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} force component(s). "
            "This may indicate atomic clashes or numerical overflow."
        )
