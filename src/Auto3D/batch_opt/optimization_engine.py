# src/Auto3D/batch_opt/optimization_engine.py
"""Optimization loop for batch geometry optimization.

This module contains the main optimization loop (n_steps) and status reporting
(print_stats) functions extracted from batchopt.py for better modularity.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from Auto3D.batch_opt.fire_optimizer import FIRE
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


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


def optimization_counts(state: dict[str, Any], patience: int) -> tuple[int, int, int, int]:
    """Return (total, converged, dropped, active) structure counts from state.

    ``converged`` excludes structures dropped for oscillation; ``active`` is the
    number still being optimized. Performs a small host-device sync (sum of two
    boolean masks), so callers gate how often they invoke it.
    """
    num_total = int(state['numbers'].size()[0])
    num_converged_dropped = int(torch.sum(state['converged_mask']).to('cpu'))
    num_dropped = int(torch.sum(state['oscillating_count'].to('cpu') >= patience))
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    return num_total, num_converged, num_dropped, num_active


def print_stats(state: dict[str, Any], patience: int) -> None:
    """Print the optimization status.

    Outputs the current optimization progress including total structures,
    converged structures, dropped structures (oscillating), and active structures.

    Args:
        state: Optimization state dictionary containing:
            - numbers: Atomic numbers tensor, shape (batch, n_atoms)
            - converged_mask: Boolean convergence mask, shape (batch,)
            - oscillating_count: Oscillation counter tensor, shape (batch,)
        patience: Number of steps without force decrease before dropping a structure.
    """
    num_total, num_converged, num_dropped, num_active = optimization_counts(state, patience)
    logger.info("Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i" %
          (num_total, num_converged, num_dropped, num_active))


def n_steps(
    state: dict[str, Any],
    n: int,
    opttol: float,
    patience: int,
    atom_mask: torch.Tensor | None = None,
    progress_cb: Callable[[dict], None] | None = None,
) -> None:
    """Run n optimization steps for each input structure.

    Only non-converged structures are modified at each step. n_steps does not
    change input conformer order.

    A structure leaves the active set for one of two reasons:
    1. Force convergence: maximum force at or below opttol
    2. Oscillation detection: drops structures that don't improve for patience steps

    There is deliberately no energy-stability criterion. One existed until
    4.0.0 but could never fire: it required ``fmax < opttol``, which is
    exactly the condition under which the force criterion has already stopped
    the structure, so the term was the identity of ``&`` at every element (see
    ``test_convergence_outcome_never_depends_on_energy_stability``).

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
        atom_mask: Boolean mask, shape (batch, n_atoms), True for real atoms and
            False for padded (ghost) atom slots. Forces on padded slots are
            zeroed before the force-convergence reduction so convergence does
            not depend on how the model treats padded atoms. Stored into
            ``state['atom_mask']`` and subset alongside ``numbers``/``coord``
            each step. Deriving this mask from a species sentinel value
            (``numbers == species_pad``) breaks whenever a custom NNP's
            padding value collides with a real species index -- the exact
            convention Auto3D itself uses for ANI2xt, where 0 is hydrogen
            (audit C13). Defaults to None, which is a no-op (every atom
            treated as real) for unpadded batches or any caller that omits it.
    """
    numbers = state['numbers']
    charges = state['charges']
    coord = state['coord']

    if atom_mask is None:
        atom_mask = torch.ones_like(numbers, dtype=torch.bool)
    state['atom_mask'] = atom_mask

    # Validate input state tensors before processing
    _validate_state(state)

    optimizer = FIRE(coord)

    # The following two terms are used to detect oscillating conformers
    smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999,
                                  dtype=torch.float).to(coord.device)
    # Integer step counter: only ever incremented by a bool mask and compared
    # to `patience`, so use torch.long rather than float for a quantity that is
    # conceptually an integer count.
    oscillating_count0 = torch.zeros(len(coord), dtype=torch.long,
                                     device=coord.device)

    state["oscillating_count"] = oscillating_count0

    istep = 0  # Initialize in case loop doesn't execute (n=0)
    for istep in tqdm(range(1, (n + 1), 1)):
        not_converged = ~ state['converged_mask']  # Essential tracker handle, size fixed
        # Stop optimization if all structures converged. The all-converged check
        # `not not_converged.any()` forces a GPU->CPU sync, so throttle it to
        # every 10 steps. `not_converged` itself is still recomputed every step
        # because the loop body subsets the batch with it below.
        if istep % 10 == 0 and not not_converged.any():
            break

        coord = state['coord'][not_converged]  # Subset coordinates, size=not_converged.
        # On non-throttle steps we may reach here after every molecule has
        # converged (the .any() break only runs every 10 steps). Subsetting then
        # yields a zero-length batch; bail out before the (empty) NN call rather
        # than feeding an empty batch through the model. `.shape[0]` is host-side
        # tensor metadata, so this guard adds no host-device sync.
        if coord.shape[0] == 0:
            break
        numbers = state['numbers'][not_converged]
        charges = state['charges'][not_converged]
        atom_mask_subset = state['atom_mask'][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]
        oscillating_count = state["oscillating_count"][not_converged]

        coord.requires_grad_(True)
        e, f = state['nn'].forward_batched(coord, numbers,
                                           charges)  # Key step to calculate all energies and forces.
        coord.requires_grad_(False)

        # Zero forces on padded atom slots so convergence is independent of how
        # the model treats ghost atoms. atom_mask is True for real atoms.
        # Deriving this from a sentinel value (numbers == species_pad) broke
        # for any model whose species_pad collides with a real index (audit
        # C13). Use the loop-local `atom_mask_subset` (state['atom_mask'][not_converged])
        # so the mask aligns with the current batch of f. Masking before the
        # optimizer step also keeps padded atoms from drifting.
        pad_mask = ~atom_mask_subset.unsqueeze(-1)
        f = f.masked_fill(pad_mask, 0.0)
        # Detach the optimizer output so the next step starts from a leaf tensor.
        # Production adapters return detached forces, so coord never tracks grad
        # across steps; detaching here makes the loop robust to NNPs whose forces
        # still carry a grad graph (otherwise the next requires_grad_ would error).
        coord = optimizer(coord, f).detach()
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
        oscillating_count += fmax_not_reduced
        not_oscillating = oscillating_count < patience

        # Combine the convergence criteria. An `& ~energy_converged` term stood
        # here until 4.0.0; `energy_converged` required `fmax < opttol` while
        # `not_converged_post1` is `fmax > opttol`, so the term was the identity
        # of `&` wherever it was consulted and false-dominated elsewhere -- it
        # could never change an outcome, including at the `fmax == opttol`
        # boundary where both comparisons are false (audit M1).
        not_converged_post = not_converged_post1 & not_oscillating

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

        # Print stats every 10% of steps (avoid division by zero for small n)
        if n >= 10 and (istep % (n // 10)) == 0:
            print_stats(state, patience)

        # Emit a live-progress event for the CLI display. Reuses the istep % 10
        # sync cadence; only active when a callback was supplied (i.e. interactive
        # `auto3d run`), so the default/library path adds nothing. Guarded so a
        # progress hiccup can never abort the optimization.
        if progress_cb is not None and istep % 10 == 0:
            try:
                total, converged, dropped, active = optimization_counts(state, patience)
                progress_cb({"step": istep, "total": total, "converged": converged,
                             "dropped": dropped, "active": active})
            except Exception:
                pass

    # Final event so the display reflects the converged end state.
    if progress_cb is not None:
        try:
            total, converged, dropped, active = optimization_counts(state, patience)
            progress_cb({"step": istep, "total": total, "converged": converged,
                         "dropped": dropped, "active": active})
        except Exception:
            pass

    # Energy and fmax stored during the loop are evaluated at the pre-step
    # geometry, while the stored coordinates are post-step (the loop always takes
    # one FIRE step after measuring forces). Recompute both once at the final
    # geometry so state['energy'] and state['fmax'] correspond to the reported
    # state['coord']. The adapters differentiate internally for forces, so grad
    # must be enabled; the forces were previously discarded.
    final_coord = state['coord'].detach().clone().requires_grad_(True)
    e_final, f_final = state['nn'].forward_batched(final_coord, state['numbers'], state['charges'])
    state['energy'] = e_final.detach().to(state['energy'].dtype)
    # Zero padded-atom force slots before the reduction, matching the in-loop
    # convergence check, so reported fmax is independent of how the model treats
    # ghost atoms. atom_mask is True for real atoms (audit C13).
    f_final = f_final.detach().masked_fill(~state['atom_mask'].unsqueeze(-1), 0.0)
    state['fmax'] = f_final.norm(dim=-1).max(dim=-1)[0].to(state['fmax'].dtype)

    if istep == (n):
        logger.info("Reaching maximum optimization step:")
    else:
        logger.info(f"Optimization finished at step {istep}:")
    print_stats(state, patience)
