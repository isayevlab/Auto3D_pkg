# src/Auto3D/batch_opt/optimization_engine.py
"""Optimization loop for batch geometry optimization.

This module contains the main optimization loop (n_steps) and status reporting
(print_stats) functions extracted from batchopt.py for better modularity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from Auto3D.batch_opt.fire_optimizer import FIRE


def _validate_state(state: dict[str, Any]) -> None:
    """Validate optimization state tensors.

    Validates that the input tensors have the correct dimensionality:
    - coord: 3D tensor (batch, atoms, 3)
    - numbers: 2D tensor (batch, atoms)
    - charges: 1D tensor (batch,)

    Args:
        state: Optimization state dictionary containing coord, numbers, and charges.

    Raises:
        ValueError: If tensor shapes are invalid.
    """
    coord = state['coord']
    numbers = state['numbers']
    charges = state['charges']

    if len(coord.shape) != 3:
        raise ValueError(
            f"coord must be 3D tensor (batch, atoms, 3), got shape {tuple(coord.shape)}"
        )
    if len(numbers.shape) != 2:
        raise ValueError(
            f"numbers must be 2D tensor (batch, atoms), got shape {tuple(numbers.shape)}"
        )
    if len(charges.shape) != 1:
        raise ValueError(
            f"charges must be 1D tensor (batch,), got shape {tuple(charges.shape)}"
        )


def print_stats(state: dict[str, Any], patience: int) -> None:
    """Print the optimization status.

    Outputs the current optimization progress including total structures,
    converged structures, dropped structures (oscillating), and active structures.

    Args:
        state: Optimization state dictionary containing:
            - numbers: Atomic numbers tensor, shape (batch, n_atoms)
            - converged_mask: Boolean convergence mask, shape (batch,)
            - oscillating_count: Oscillation counter tensor, shape (batch, 1)
        patience: Number of steps without force decrease before dropping a structure.
    """
    numbers = state['numbers']
    num_total = numbers.size()[0]
    num_converged_dropped = torch.sum(state['converged_mask']).to('cpu')
    oscillating_count = state['oscillating_count'].to('cpu').reshape(-1, ) >= patience
    num_dropped = torch.sum(oscillating_count)
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    print("Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i" %
          (num_total, num_converged, num_dropped, num_active), flush=True)


def n_steps(
    state: dict[str, Any],
    n: int,
    opttol: float,
    patience: int,
    energy_tol: float = 1e-4,
    energy_patience: int = 3,
) -> None:
    """Run n optimization steps for each input structure.

    Only non-converged structures are modified at each step. n_steps does not
    change input conformer order.

    Uses multiple convergence criteria:
    1. Force convergence: maximum force below opttol
    2. Oscillation detection: drops structures that don't improve for patience steps
    3. Energy stability: converges if energy stable for energy_patience steps

    Args:
        state: Optimization state dictionary containing:
            - numbers: Atomic numbers, shape (batch, n_atoms)
            - charges: Molecular charges, shape (batch,)
            - coord: Coordinates, shape (batch, n_atoms, 3)
            - nn: Neural network model wrapper (EnForce_ANI)
            - converged_mask: Boolean convergence mask, shape (batch,)
            - fmax: Maximum force per molecule, shape (batch,)
            - energy: Energy per molecule, shape (batch,)
        n: Maximum number of optimization steps.
        opttol: Force convergence tolerance in eV/Angstrom.
        patience: Number of steps without force decrease before dropping a structure
            as oscillating.
        energy_tol: Energy convergence threshold in eV (default 1e-4 eV = ~0.002 kcal/mol).
        energy_patience: Number of steps energy must be stable before considering
            converged (default 3).
    """
    numbers = state['numbers']
    charges = state['charges']
    coord = state['coord']

    # Validate input state tensors before processing
    _validate_state(state)

    optimizer = FIRE(coord)

    # The following two terms are used to detect oscillating conformers
    smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999,
                                  dtype=torch.float).to(coord.device)
    oscillating_count0 = torch.tensor(np.zeros((len(coord), 1)),
                                      dtype=torch.float).to(coord.device)

    # Energy-based convergence tracking
    prev_energy = torch.full((len(coord),), float('inf'), dtype=torch.double, device=coord.device)
    energy_stable_count = torch.zeros(len(coord), dtype=torch.long, device=coord.device)

    state["oscillating_count"] = oscillating_count0

    for istep in tqdm(range(1, (n + 1), 1)):
        not_converged = ~ state['converged_mask']  # Essential tracker handle, size fixed
        # Stop optimization if all structures converged.
        if not not_converged.any():
            break

        coord = state['coord'][not_converged]  # Subset coordinates, size=not_converged.
        numbers = state['numbers'][not_converged]
        charges = state['charges'][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscillating_count = state["oscillating_count"][not_converged]
        prev_e_subset = prev_energy[not_converged]
        energy_stable_subset = energy_stable_count[not_converged]

        coord.requires_grad_(True)
        e, f = state['nn'].forward_batched(coord, numbers,
                                           charges)  # Key step to calculate all energies and forces.
        coord.requires_grad_(False)

        coord = optimizer(coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[
            0]  # Tensor, Norm is the length of each vector. Here it returns the maximum force length for each conformer. Size (100)
        not_converged_post1 = fmax > opttol

        # Update smallest_fmax for each molecule
        fmax_reduced = fmax.reshape(-1, 1) < smallest_fmax
        fmax_reduced = fmax_reduced.reshape(-1, )
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        # Reduce count to 0 for reducing; raise count for non-reducing
        oscillating_count[fmax_reduced] = 0
        fmax_not_reduced = ~fmax_reduced
        oscillating_count += fmax_not_reduced.reshape(-1, 1)
        not_oscillating = oscillating_count < patience
        not_oscillating = not_oscillating.reshape(-1, )

        # Energy-based convergence: check if energy change is below threshold
        e_double = e.detach().to(torch.double)
        energy_change = torch.abs(e_double - prev_e_subset)
        energy_stable = energy_change < energy_tol
        # Increment count where energy is stable, reset where not
        energy_stable_subset = torch.where(energy_stable, energy_stable_subset + 1, torch.zeros_like(energy_stable_subset))
        # Consider converged if energy stable for energy_patience steps AND force is reasonable (< 10x opttol)
        energy_converged = (energy_stable_subset >= energy_patience) & (fmax < opttol * 10)

        # Combine all convergence criteria
        not_converged_post = not_converged_post1 & not_oscillating & ~energy_converged

        optimizer.clean(not_converged_post)  # Subset v, a in FIRE for next optimization

        state['converged_mask'][
            not_converged] = ~ not_converged_post  # Update converged_mask, so that converged structures will not be updated in future steps.
        state['fmax'][
            not_converged] = fmax  # Update fmax for conformers that are optimized in this iteration
        state['energy'][
            not_converged] = e.detach().to(state['energy'].dtype)  # Update energy for conformers that are optimized in this iteration
        state['coord'][
            not_converged] = coord  # Update coordinates for conformers that are optimized in this iteration
        smallest_fmax0[not_converged] = smallest_fmax  # Update smallest_fmax for each conformer
        state["oscillating_count"][
            not_converged] = oscillating_count  # Update counts for continuous no reduction in fmax
        prev_energy[not_converged] = e_double  # Update previous energy for next iteration
        energy_stable_count[not_converged] = energy_stable_subset  # Update energy stability count

        # Print stats every 10% of steps (avoid division by zero for small n)
        if n >= 10 and (istep % (n // 10)) == 0:
            print_stats(state, patience)

    if istep == (n):
        print("Reaching maximum optimization step:   ", end="")
    else:
        print(f"Optimization finished at step {istep}:   ", end="")
    print_stats(state, patience)
