# src/Auto3D/batch_opt/optimization_engine.py
"""Optimization loop for batch geometry optimization.

This module contains the main optimization loop (n_steps) and status reporting
(print_stats) functions extracted from batchopt.py for better modularity.

Host-device synchronization
---------------------------
``n_steps`` runs up to 2000 iterations per bucket, so a per-step
host-device serialization point costs 2000 of them. Every subset read and write
in the loop therefore goes through **one** ``torch.nonzero`` per step, whose
int64 result feeds ``index_select`` (reads) and ``index_copy_`` (writes) --
neither of which synchronizes.

The loop used to subset with boolean masks instead, which on CUDA is 18 syncs
per step: ATen has to ``nonzero()`` each mask and copy the element count to the
host to size the output, once per masked read and once per masked write, six of
each here plus four more inside ``FIRE.clean``. Reusing a single ``nonzero``
result across all twelve gathers and scatters takes that to **2 per step** --
one for the active subset, one inside the step for ``FIRE.clean``, whose mask is
indexed within the active subset rather than the full batch.

Two, not zero: ``nonzero`` is *itself* the sync, which is precisely why boolean
masking synced. The win is amortizing one over twelve uses, not eliminating it.
``tests/test_optimization_engine_indexing.py`` counts this on CPU (the sync is a
property of the operator, not the device) and fails if boolean-mask indexing
reappears.

``index_copy_`` is stricter than masked assignment about dtype: mismatched
writes raise rather than cast. Every write below therefore casts explicitly to
the destination dtype. That is not merely tidiness -- it is what lets a custom
NNP return float64 forces, which the boolean-mask loop rejected with
``"Index put requires the source and destination dtypes match"`` whenever two or
more molecules reduced their force in the same step (with exactly one, the value
had ``numel() == 1``, hit ATen's ``masked_fill_`` fast path and silently cast,
so the crash was batch-size dependent).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch

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
    coord = state["coord"]
    numbers = state["numbers"]
    charges = state["charges"]

    if len(coord.shape) != 3:
        raise ValueError(
            f"coord must be 3D tensor (batch, atoms, 3), got shape {tuple(coord.shape)}"
        )
    if len(numbers.shape) != 2:
        raise ValueError(
            f"numbers must be 2D tensor (batch, atoms), got shape {tuple(numbers.shape)}"
        )
    if len(charges.shape) != 1:
        raise ValueError(f"charges must be 1D tensor (batch,), got shape {tuple(charges.shape)}")


def optimization_counts(state: dict[str, Any], patience: int) -> tuple[int, int, int, int]:
    """Return (total, converged, dropped, active) structure counts from state.

    ``converged`` excludes structures dropped for oscillation; ``active`` is the
    number still being optimized. Performs a small host-device sync (sum of two
    boolean masks), so callers gate how often they invoke it.
    """
    num_total = int(state["numbers"].size()[0])
    num_converged_dropped = int(torch.sum(state["converged_mask"]).to("cpu"))
    num_dropped = int(torch.sum(state["oscillating_count"].to("cpu") >= patience))
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
    logger.info(
        "Total 3D structures: %i  Converged: %i   Dropped(Oscillating): %i    Active: %i"
        % (num_total, num_converged, num_dropped, num_active)
    )


def _emit_progress(
    state: dict[str, Any],
    patience: int,
    progress_cb: Callable[[dict], None] | None,
    istep: int,
) -> None:
    """Emit one live-progress event, if a callback was supplied.

    Only active when the caller passed a callback (i.e. interactive
    ``auto3d run``), so the default library path adds nothing -- including the
    two host-device syncs ``optimization_counts`` performs. Wrapped so a
    progress-display hiccup can never abort an optimization that is otherwise
    fine.

    Args:
        state: Optimization state dictionary.
        patience: Oscillation patience, needed to split converged from dropped.
        progress_cb: Callback, or None to do nothing.
        istep: Step number to report.
    """
    if progress_cb is None:
        return
    try:
        total, converged, dropped, active = optimization_counts(state, patience)
        progress_cb(
            {
                "step": istep,
                "total": total,
                "converged": converged,
                "dropped": dropped,
                "active": active,
            }
        )
    except Exception:
        pass


class _StepResult(NamedTuple):
    """Per-molecule results for one step, ordered by position in the active subset.

    Every field has leading dimension ``active_idx.numel()``, so
    ``_scatter_back`` can write all six with the same index.

    Attributes:
        coord: Post-step coordinates, ``(active, n_atoms, 3)``.
        energy: Energy at the *pre*-step geometry, ``(active,)``.
        fmax: Maximum force at the pre-step geometry, ``(active,)``.
        still_active: True for molecules that remain in the active set, i.e.
            neither force-converged nor dropped as oscillating, ``(active,)``.
        smallest_fmax: Running per-molecule force minimum, ``(active, 1)``.
        oscillating_count: Steps since that minimum last improved, ``(active,)``.
    """

    coord: torch.Tensor
    energy: torch.Tensor
    fmax: torch.Tensor
    still_active: torch.Tensor
    smallest_fmax: torch.Tensor
    oscillating_count: torch.Tensor


def _step_active_subset(
    state: dict[str, Any],
    optimizer: FIRE,
    active_idx: torch.Tensor,
    smallest_fmax0: torch.Tensor,
    opttol: float,
    patience: int,
) -> _StepResult:
    """Take one FIRE step over the still-active molecules.

    Gathers the active rows with ``index_select`` (no sync), evaluates the model,
    applies the force-convergence and oscillation criteria, steps the optimizer
    and subsets the optimizer's own state. Mutates ``optimizer`` in place;
    everything else is returned for ``_scatter_back`` to write.

    Args:
        state: Optimization state dictionary. Read, not written.
        optimizer: The FIRE optimizer, whose per-molecule state is subset here.
        active_idx: int64 row indices of still-active molecules, ascending.
        smallest_fmax0: Full-batch running force minimum, ``(batch, 1)``.
        opttol: Force convergence tolerance in eV/Angstrom.
        patience: Steps without force decrease before dropping as oscillating.

    Returns:
        A ``_StepResult`` aligned with ``active_idx``.
    """
    coord = state["coord"].index_select(0, active_idx)
    numbers = state["numbers"].index_select(0, active_idx)
    charges = state["charges"].index_select(0, active_idx)
    atom_mask_subset = state["atom_mask"].index_select(0, active_idx)
    smallest_fmax = smallest_fmax0.index_select(0, active_idx)
    oscillating_count = state["oscillating_count"].index_select(0, active_idx)

    coord.requires_grad_(True)
    # atom_mask goes to the model too, not just to the force reduction below:
    # an adapter that has to flatten a padded batch (AIMNet2) needs to know
    # which slots are real, and deriving that from a species sentinel is what
    # audit C13 forbids.
    e, f = state["nn"].forward_batched(
        coord,
        numbers,
        charges,
        atom_mask=atom_mask_subset,
    )  # Key step to calculate all energies and forces.
    coord.requires_grad_(False)

    # Zero forces on padded atom slots so convergence is independent of how the
    # model treats ghost atoms. atom_mask is True for real atoms. Deriving this
    # from a sentinel value (numbers == species_pad) broke for any model whose
    # species_pad collides with a real index (audit C13). Masking before the
    # optimizer step also keeps padded atoms from drifting.
    pad_mask = ~atom_mask_subset.unsqueeze(-1)
    f = f.masked_fill(pad_mask, 0.0)
    # Norm is the length of each force vector; the max over atoms is the
    # per-molecule convergence measure.
    fmax = f.norm(dim=-1).max(dim=-1)[0]

    # The force-convergence test runs BEFORE the FIRE step, and must stay there.
    not_converged_post1 = fmax > opttol
    # Detach the optimizer output so the next step starts from a leaf tensor.
    # Production adapters return detached forces, so coord never tracks grad
    # across steps; detaching here makes the loop robust to NNPs whose forces
    # still carry a grad graph (otherwise the next requires_grad_ would error).
    stepped = optimizer(coord, f).detach()
    # A structure that has just met the force criterion keeps the geometry its
    # force was measured at; only the ones still moving take the step.
    #
    # The convergence test above and the step used to run in the other order,
    # so every structure took one more FIRE step *after* the force that
    # declared it converged. `Converged` then described the geometry before
    # that step while the reported `fmax` (recomputed at the end of n_steps)
    # described the geometry after it -- and `batchopt` writes both onto the
    # same record. A consumer filtering on `fmax <= opt_tol` and one filtering
    # on `Converged == "True"` got different sets from one file. Measured on a
    # hermetic harmonic potential before this change: fmax up to 6.9x the
    # tolerance beside `Converged=True`. The discrepancy grows with stiffness,
    # so a soft test case hides it entirely -- which is how it survived being
    # looked at twice.
    #
    # Structures leaving the active set as oscillating are stepped like any
    # other: they are reported `Converged=False`, so no consistency is claimed
    # for them, and n_steps' end-of-function recompute makes their `fmax` match
    # their coordinates regardless.
    coord = torch.where(not_converged_post1.view(-1, 1, 1), stepped, coord)

    # Update the running force minimum and the oscillation counter. Both are
    # per-molecule elementwise updates that need no index at all, so they are
    # `torch.where` rather than masked assignment.
    #
    # `torch.where`, NOT `torch.minimum`: `<` is False for NaN, so the masked
    # assignment this replaced *kept* the previous smallest_fmax when a
    # molecule's force went NaN, whereas `minimum` propagates the NaN and would
    # poison that molecule's oscillation tracking for the rest of the run.
    # `_validate_outputs` should make NaN unreachable through the adapters, but
    # "should be unreachable" is not a reason to change semantics.
    fmax_col = fmax.reshape(-1, 1)
    fmax_reduced = (fmax_col < smallest_fmax).reshape(
        -1,
    )
    smallest_fmax = torch.where(fmax_reduced.unsqueeze(-1), fmax_col, smallest_fmax)
    # Reduced -> reset to 0; not reduced -> increment. This is one `where` in
    # place of "zero the reduced entries, then add ~reduced", whose ordering was
    # load-bearing: a reduced molecule was zeroed and then incremented by False.
    oscillating_count = torch.where(
        fmax_reduced, torch.zeros_like(oscillating_count), oscillating_count + 1
    )
    not_oscillating = oscillating_count < patience

    # Combine the convergence criteria. An `& ~energy_converged` term stood
    # here until 3.0.0; `energy_converged` required `fmax < opttol` while
    # `not_converged_post1` is `fmax > opttol`, so the term was the identity
    # of `&` wherever it was consulted and false-dominated elsewhere -- it
    # could never change an outcome, including at the `fmax == opttol`
    # boundary where both comparisons are false (audit M1).
    still_active = not_converged_post1 & not_oscillating

    # Second nonzero of the step: `still_active` is indexed within the active
    # subset, not the full batch, so it cannot reuse `active_idx`. FIRE.clean
    # takes an int64 index precisely so this is one sync instead of four.
    optimizer.clean(torch.nonzero(still_active, as_tuple=True)[0])

    return _StepResult(
        coord=coord,
        energy=e.detach(),
        fmax=fmax,
        still_active=still_active,
        smallest_fmax=smallest_fmax,
        oscillating_count=oscillating_count,
    )


def _scatter_back(
    state: dict[str, Any],
    active_idx: torch.Tensor,
    smallest_fmax0: torch.Tensor,
    result: _StepResult,
) -> None:
    """Write one step's results back into the full-batch state tensors.

    All six writes reuse ``active_idx``, so they cost no synchronization.
    ``index_copy_`` requires the source dtype to match the destination exactly
    (it will not cast, in either direction), so each source is cast at the call
    site -- which is also the only reason a custom NNP returning float64 forces
    works here.

    Args:
        state: Optimization state dictionary, mutated in place.
        active_idx: int64 row indices the results correspond to.
        smallest_fmax0: Full-batch running force minimum, mutated in place.
        result: The values returned by ``_step_active_subset``.
    """
    # Converged structures are excluded from every subsequent step.
    state["converged_mask"].index_copy_(
        0, active_idx, (~result.still_active).to(state["converged_mask"].dtype)
    )
    state["fmax"].index_copy_(0, active_idx, result.fmax.to(state["fmax"].dtype))
    state["energy"].index_copy_(0, active_idx, result.energy.to(state["energy"].dtype))
    state["coord"].index_copy_(0, active_idx, result.coord.to(state["coord"].dtype))
    smallest_fmax0.index_copy_(0, active_idx, result.smallest_fmax.to(smallest_fmax0.dtype))
    state["oscillating_count"].index_copy_(
        0, active_idx, result.oscillating_count.to(state["oscillating_count"].dtype)
    )


def _recompute_final_energy_and_fmax(state: dict[str, Any]) -> None:
    """Re-evaluate energy and fmax at the final reported geometry.

    Energy and fmax stored during the loop are evaluated at the *pre*-step
    geometry, while the stored coordinates are post-step (the loop always takes
    one FIRE step after measuring forces). Recompute both once at the end so
    ``state['energy']`` and ``state['fmax']`` correspond to the reported
    ``state['coord']``. The adapters differentiate internally for forces, so
    grad must be enabled; the forces themselves were previously discarded.

    Args:
        state: Optimization state dictionary, mutated in place.
    """
    final_coord = state["coord"].detach().clone().requires_grad_(True)
    e_final, f_final = state["nn"].forward_batched(
        final_coord,
        state["numbers"],
        state["charges"],
        atom_mask=state["atom_mask"],
    )
    state["energy"] = e_final.detach().to(state["energy"].dtype)
    # Zero padded-atom force slots before the reduction, matching the in-loop
    # convergence check, so reported fmax is independent of how the model treats
    # ghost atoms. atom_mask is True for real atoms (audit C13).
    f_final = f_final.detach().masked_fill(~state["atom_mask"].unsqueeze(-1), 0.0)
    state["fmax"] = f_final.norm(dim=-1).max(dim=-1)[0].to(state["fmax"].dtype)


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
    3.0.0 but could never fire: it required ``fmax < opttol``, which is
    exactly the condition under which the force criterion has already stopped
    the structure, so the term was the identity of ``&`` at every element (see
    ``test_convergence_outcome_never_depends_on_energy_stability``).

    The per-step work is split across ``_step_active_subset`` (gather, model,
    criteria, FIRE step) and ``_scatter_back`` (the writes), both driven by a
    single ``torch.nonzero`` of the active set -- see the module docstring for
    why that matters and what it costs.

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
        progress_cb: Optional callback invoked every 10 steps and once at the
            end with a dict of step/total/converged/dropped/active counts.
    """
    numbers = state["numbers"]
    coord = state["coord"]

    if atom_mask is None:
        atom_mask = torch.ones_like(numbers, dtype=torch.bool)
    state["atom_mask"] = atom_mask

    # Validate input state tensors before processing
    _validate_state(state)

    optimizer = FIRE(coord)

    # The following two terms are used to detect oscillating conformers
    smallest_fmax0 = torch.tensor(np.ones((len(coord), 1)) * 999, dtype=torch.float).to(
        coord.device
    )
    # Integer step counter: only ever compared to `patience` and reset to zero,
    # so use torch.long rather than float for a quantity that is conceptually
    # an integer count.
    state["oscillating_count"] = torch.zeros(len(coord), dtype=torch.long, device=coord.device)

    istep = 0  # Initialize in case loop doesn't execute (n=0)
    # Plain range, not tqdm. A bar over `range(1, n+1)` measures the *step
    # budget*, which is not the work: a run that converges at step 300 of 2000
    # showed 15% and then disappeared, and a run where nothing converged
    # marched confidently to 100%. It also wrote carriage returns into stderr
    # unconditionally -- tqdm only auto-disables on `disable=None`, so every
    # redirected log and CI transcript collected the control characters too.
    # Real progress is reported by `print_stats` (every 10% of the budget) and
    # by the `progress_cb` events below, both of which carry converged/active/
    # dropped counts rather than a step ratio.
    for istep in range(1, n + 1):
        not_converged = ~state["converged_mask"]  # Essential tracker handle, size fixed
        # Stop optimization if all structures converged. The all-converged check
        # `not not_converged.any()` forces a GPU->CPU sync, so throttle it to
        # every 10 steps. `not_converged` itself is still recomputed every step
        # because the loop body subsets the batch with it below.
        if istep % 10 == 0 and not not_converged.any():
            break

        # THE one sync of the gather half of the step. Its int64 result is
        # reused by six index_select reads and six index_copy_ writes; the
        # boolean-mask spelling would have paid a nonzero() for each of the
        # twelve. Indices come out ascending, so gathered row order -- and
        # therefore every downstream reduction -- is unchanged.
        active_idx = torch.nonzero(not_converged, as_tuple=True)[0]
        # On non-throttle steps we may reach here after every molecule has
        # converged (the .any() break only runs every 10 steps). Stepping then
        # would feed a zero-length batch through the model, so bail out first.
        # `numel()` on the nonzero result is host-side metadata that the nonzero
        # already made available, so this guard adds no further sync.
        if active_idx.numel() == 0:
            break

        result = _step_active_subset(state, optimizer, active_idx, smallest_fmax0, opttol, patience)
        _scatter_back(state, active_idx, smallest_fmax0, result)

        # Print stats every 10% of steps (avoid division by zero for small n)
        if n >= 10 and (istep % (n // 10)) == 0:
            print_stats(state, patience)

        # Live-progress event for the CLI display, on the same 10-step cadence
        # as the throttled sync above.
        if istep % 10 == 0:
            _emit_progress(state, patience, progress_cb, istep)

    # Final event so the display reflects the converged end state.
    _emit_progress(state, patience, progress_cb, istep)

    _recompute_final_energy_and_fmax(state)

    if istep == (n):
        logger.info("Reaching maximum optimization step:")
    else:
        logger.info(f"Optimization finished at step {istep}:")
    print_stats(state, patience)
