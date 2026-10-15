"""FIRE (Fast Inertial Relaxation Engine) optimizer for geometry optimization.

This module provides the FIRE optimizer class for molecular geometry optimization.
FIRE is an efficient method for finding energy minima, combining aspects of
molecular dynamics and conjugate gradient methods.

Implementation based on:
    Bitzek, Erik, Pekka Koskinen, Franz Gaehler, Michael Moseler, and Peter
    Gumbsch. "Structural relaxation made simple." Physical Review Letters 97,
    no. 17 (2006): 170201.

The five adaptation constants are constructor arguments whose defaults are the
set every Auto3D release has run; the named sets live in
``Auto3D.foundation.constants.FIRE_CONSTANT_SETS`` and are selected by name
(``OptimizationConfig.fire_constants``, ``n_steps(..., fire_constants=...)``).
Nothing selects a non-default set in a shipped run: the second set exists so
``benchmarks/measure_fire_constants.py`` can measure the two against each
other before any default moves.

Example:
    >>> import torch
    >>> from Auto3D.engines.batch_opt.fire_optimizer import FIRE
    >>> coord = torch.randn(10, 20, 3)  # 10 molecules, 20 atoms each
    >>> optimizer = FIRE(coord)
    >>> forces = torch.randn(10, 20, 3)
    >>> new_coord = optimizer(coord, forces)
"""

from __future__ import annotations

import torch


@torch.jit.script
class FIRE:
    """FIRE optimizer for batch molecular geometry optimization.

    A general optimization algorithm using the Fast Inertial Relaxation Engine,
    which combines a semi-implicit (symplectic) Euler integration step --
    the same one ASE's own FIRE implements -- with adaptive time stepping.
    This implementation supports batch processing of multiple molecules
    simultaneously.

    The algorithm adjusts the time step and mixing parameter dynamically
    based on whether the optimization is making progress (forces aligned
    with velocities) or needs to be reset (forces opposing velocities).

    Two named sets of the five adaptation constants exist
    (``Auto3D.foundation.constants.FIRE_CONSTANT_SETS``), and the defaults
    below are one of them:

    * ``auto3d`` -- ``dt_max 0.1, maxstep 0.1, finc 1.5, fdec 0.7,
      astart 0.1``, with the progress test applied on the first step. This is
      the set every release has run and the default here.
    * ``bitzek2006`` -- ``dt_max 1.0, maxstep 0.2, finc 1.1, fdec 0.5,
      astart 0.1``, the constants of Bitzek et al. 2006 as ASE implements
      them, with ``skip_first_progress_test=True`` because ASE runs no
      progress test on the first step (the velocity is still zero there, so
      the dot product of velocity and force is zero and the step would count
      as "not progressing").

    ``dt_init`` is a separate argument from ``dt_max``: the two are equal in
    the ``auto3d`` set but not in general, so a different ``dt_max`` has to
    come with its own starting time step rather than inherit it.

    Attributes:
        dt_max: Maximum time step (default: 0.1).
        Nmin: Minimum number of steps before increasing time step (default: 5).
        maxstep: Maximum atomic displacement per step (default: 0.1).
        finc: Factor for increasing time step (default: 1.5).
        fdec: Factor for decreasing time step (default: 0.7).
        astart: Starting value for mixing parameter (default: 0.1).
        fa: Factor for decreasing mixing parameter (default: 0.99).
        skip_first_progress_test: Whether a molecule's first step leaves
            ``dt``, ``a`` and ``Nsteps`` untouched (default: False).
        v: Velocities for each atom, shape (batch, n_atoms, 3).
        Nsteps: Number of consecutive successful steps per molecule.
        dt: Current time step per molecule.
        a: Current mixing parameter per molecule.
        first_step: True per molecule until that molecule has taken a step,
            shape (batch,).
    """

    def __init__(
        self,
        coord: torch.Tensor,
        dt_max: float = 0.1,
        maxstep: float = 0.1,
        finc: float = 1.5,
        fdec: float = 0.7,
        astart: float = 0.1,
        dt_init: float = 0.1,
        skip_first_progress_test: bool = False,
    ) -> None:
        """Initialize FIRE optimizer.

        Every argument but ``coord`` is one constant of a named set; the
        defaults are the ``auto3d`` set, so ``FIRE(coord)`` is exactly the
        optimizer every release has run. A caller selecting a set spells it
        ``FIRE(coord, **FIRE_CONSTANT_SETS[name])``.

        Args:
            coord: Initial coordinates, shape (batch, n_atoms, 3).
                   Used to initialize internal state tensors with proper
                   shape and device.
            dt_max: Maximum time step.
            maxstep: Maximum atomic displacement per step.
            finc: Factor for increasing the time step.
            fdec: Factor for decreasing the time step.
            astart: Starting value for the mixing parameter, and the value it
                resets to on a non-progressing step.
            dt_init: Starting time step, which is not necessarily ``dt_max``.
            skip_first_progress_test: Leave ``dt``, ``a`` and ``Nsteps``
                untouched on a molecule's first step, as ASE's FIRE does.
        """
        # FIRE parameters, from the selected constant set
        self.dt_max: float = dt_max
        self.Nmin: int = 5
        self.maxstep: float = maxstep
        self.finc: float = finc
        self.fdec: float = fdec
        self.astart: float = astart
        self.fa: float = 0.99
        self.skip_first_progress_test: bool = skip_first_progress_test

        # State tensors initialized based on input shape
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], dt_init, device=coord.device)
        self.a = torch.full(coord.shape[:1], astart, device=coord.device)
        # Per molecule, not one scalar for the batch: molecules enter and leave
        # the active set at different steps, and `clean` subsets this alongside
        # the rest of the state, so each row's own first step is the one held.
        self.first_step = torch.ones(coord.shape[0], dtype=torch.bool, device=coord.device)

    def __call__(self, coord: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Move atoms based on forces using the FIRE algorithm.

        Performs one optimization step by:
        1. Computing velocity-force dot product to determine progress
        2. Mixing velocity with force direction if making progress
        3. Adjusting time step and mixing parameter adaptively
        4. Computing displacement with a semi-implicit (symplectic) Euler update
        5. Clamping maximum displacement to maxstep

        Args:
            coord: Current coordinates of atoms, shape (batch, n_atoms, 3).
            forces: Forces on each atom, shape (batch, n_atoms, 3).
                   Should be in consistent units with coords (e.g., eV/Angstrom).

        Returns:
            New coordinates moved based on input forces, shape (batch, n_atoms, 3).
        """
        # Compute dot product of velocities and forces for each molecule
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        progressing = vf > 0.0  # Molecules making progress (v aligned with f)

        # Per-molecule FIRE rule (Bitzek et al. 2006): dt/a/Nsteps adaptation
        # for a molecule depends only on that molecule's OWN progressing
        # history, never on whether its batchmates happen to be progressing
        # on the same step. Earlier revisions gated `speedup`/`nsteps_inc` on
        # a whole-batch `progressing.all()` reduction, which is a defect, not
        # a batching optimization: for a batch of one, `.all()` is a
        # tautology of `progressing` itself, so `speedup` (`progressing &
        # ~all_progressing & ...`) was always False and a solo molecule's dt
        # never left dt_init -- verified on a CPU harmonic-potential probe
        # where dt drops to 0.07 at step 1 (dt_max=0.1, fdec=0.7) and stays
        # frozen there for 60 steps, i.e. FIRE's inertial acceleration never
        # runs and what executes is damped MD at a fixed small step. Worse,
        # once any batchmate stalls, `all_progressing` goes permanently
        # False for the whole batch, so `nsteps_inc` (`progressing &
        # (all_progressing | past_nmin)`) fell back to requiring
        # `past_nmin`, and a molecule whose own Nsteps had not yet passed
        # Nmin got its counter frozen at 0 forever -- a molecule's
        # trajectory (and therefore its converged conformer's energy) would
        # depend on which unrelated molecules shared its batch. Removing
        # `all_progressing` fixes both: `speedup` and `nsteps_inc` below are
        # now plain per-molecule predicates.
        past_nmin = self.Nsteps > self.Nmin
        speedup = progressing & past_nmin

        # ASE runs no progress test on the first step (v is zero, so v.f is
        # zero): under that rule dt, a and Nsteps keep their values this step.
        hold = (
            self.first_step if self.skip_first_progress_test else torch.zeros_like(self.first_step)
        )

        # Velocity mix toward normalized force direction (computed for every
        # molecule; selected below). Matches the original mixing expression.
        a3 = self.a.unsqueeze(-1).unsqueeze(-1)
        v_norm = self.v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        f_norm = forces.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        # Clamp the force-norm denominator: a molecule can be "progressing"
        # (v.f > 0) while its float32 force-norm underflows to 0 (forces so small
        # their squares round to zero, e.g. ~1e-24), which makes forces / f_norm
        # produce inf/NaN that is then selected into self.v and corrupts that
        # conformer for the rest of the run. clamp_min is a no-op for any real
        # force norm (>> 1e-30) and only guards the degenerate near-zero limit.
        v_mixed = (1.0 - a3) * self.v + a3 * v_norm * forces / f_norm.clamp_min(1e-30)

        # v: mixed where progressing, reset to zero otherwise.
        prog3 = progressing.unsqueeze(-1).unsqueeze(-1)
        self.v = torch.where(prog3, v_mixed, torch.zeros_like(self.v))

        # dt:
        #   progressing & speedup        -> (dt * finc).clamp(max=dt_max)
        #   progressing & not speedup    -> unchanged
        #   not progressing              -> dt * fdec  (reset)
        dt_speedup = (self.dt * self.finc).clamp(max=self.dt_max)
        dt_prog = torch.where(speedup, dt_speedup, self.dt)
        dt_new = torch.where(progressing, dt_prog, self.dt * self.fdec)

        # a: same selection structure as dt.
        a_prog = torch.where(speedup, self.a * self.fa, self.a)
        a_new = torch.where(progressing, a_prog, torch.full_like(self.a, self.astart))

        # Nsteps:
        #   progressing     -> Nsteps + 1
        #   not progressing -> 0  (reset)
        nsteps_inc = progressing
        nsteps_prog = torch.where(nsteps_inc, self.Nsteps + 1, self.Nsteps)
        nsteps_new = torch.where(progressing, nsteps_prog, torch.zeros_like(self.Nsteps))

        # The three updates above are what this step would do under the
        # progress test; `hold` is what keeps a first step from applying them.
        # It is all-False unless the selected set asks for ASE's rule, so the
        # default path computes exactly the values it always did.
        self.dt = torch.where(hold, self.dt, dt_new)
        self.a = torch.where(hold, self.a, a_new)
        self.Nsteps = torch.where(hold, self.Nsteps, nsteps_new)
        self.first_step = torch.zeros_like(self.first_step)

        # Semi-implicit (symplectic) Euler update
        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v = self.v + dt * forces
        dr = dt * self.v

        # Clamp maximum displacement per molecule
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        dr = dr * (self.maxstep / normdr).clamp(max=1.0)

        return coord + dr

    def clean(self, idx: torch.Tensor) -> None:
        """Subset optimizer state to keep only the molecules at ``idx``.

        Used to drop molecules that have left the active set, so subsequent
        steps do less work.

        Takes an **int64 index**, not a boolean mask. Four boolean-mask reads
        here were four host-device synchronizations per optimization step on
        CUDA (ATen expands each mask via ``nonzero()`` and copies the element
        count to the host to size the output). ``index_select`` with an index
        the caller already computed is sync-free, so the caller pays one
        ``nonzero`` and this method pays none. Over 2000 steps per bucket that
        is 8000 serialization points removed from a loop that does no host work.

        Args:
            idx: int64 tensor of row indices to keep, shape ``(kept,)``. Must be
                int64: a boolean mask would *not* fail loudly here -- ``index_select``
                would reinterpret ``tensor([True, False])`` as indices ``[1, 0]``
                and silently permute the optimizer state -- so the dtype is
                checked instead.

        Raises:
            ValueError: ``idx`` is not int64.
        """
        if idx.dtype != torch.long:
            raise ValueError(
                "FIRE.clean expects an int64 tensor of row indices to keep, not "
                "a boolean mask. Pass torch.nonzero(mask, as_tuple=True)[0]. "
                "A bool tensor would be silently reinterpreted as indices."
            )
        self.v = self.v.index_select(0, idx)
        self.Nsteps = self.Nsteps.index_select(0, idx)
        self.dt = self.dt.index_select(0, idx)
        self.a = self.a.index_select(0, idx)
        # Subset on every path, not only when the first-step rule is in use:
        # this tensor is one of the operands of the `torch.where` calls in the
        # step, so a stale full-batch shape here would not stay inert.
        self.first_step = self.first_step.index_select(0, idx)
