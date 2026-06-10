"""FIRE (Fast Inertial Relaxation Engine) optimizer for geometry optimization.

This module provides the FIRE optimizer class for molecular geometry optimization.
FIRE is an efficient method for finding energy minima, combining aspects of
molecular dynamics and conjugate gradient methods.

Implementation based on:
    Guenole, Julien, et al. "Assessment and optimization of the fast inertial
    relaxation engine (fire) for energy minimization in atomistic simulations
    and its implementation in lammps." Computational Materials Science 175 (2020): 109584.

Example:
    >>> import torch
    >>> from Auto3D.batch_opt.fire_optimizer import FIRE
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
    which combines velocity Verlet integration with adaptive time stepping.
    This implementation supports batch processing of multiple molecules
    simultaneously.

    The algorithm adjusts the time step and mixing parameter dynamically
    based on whether the optimization is making progress (forces aligned
    with velocities) or needs to be reset (forces opposing velocities).

    Attributes:
        dt_max: Maximum time step (default: 0.1).
        Nmin: Minimum number of steps before increasing time step (default: 5).
        maxstep: Maximum atomic displacement per step (default: 0.1).
        finc: Factor for increasing time step (default: 1.5).
        fdec: Factor for decreasing time step (default: 0.7).
        astart: Starting value for mixing parameter (default: 0.1).
        fa: Factor for decreasing mixing parameter (default: 0.99).
        v: Velocities for each atom, shape (batch, n_atoms, 3).
        Nsteps: Number of consecutive successful steps per molecule.
        dt: Current time step per molecule.
        a: Current mixing parameter per molecule.
    """

    def __init__(self, coord: torch.Tensor) -> None:
        """Initialize FIRE optimizer.

        Args:
            coord: Initial coordinates, shape (batch, n_atoms, 3).
                   Used to initialize internal state tensors with proper
                   shape and device.
        """
        # Default FIRE parameters
        self.dt_max: float = 0.1
        self.Nmin: int = 5
        self.maxstep: float = 0.1
        self.finc: float = 1.5
        self.fdec: float = 0.7
        self.astart: float = 0.1
        self.fa: float = 0.99

        # State tensors initialized based on input shape
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], 0.1, device=coord.device)
        self.a = torch.full(coord.shape[:1], 0.1, device=coord.device)

    def __call__(self, coord: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Move atoms based on forces using the FIRE algorithm.

        Performs one optimization step by:
        1. Computing velocity-force dot product to determine progress
        2. Mixing velocity with force direction if making progress
        3. Adjusting time step and mixing parameter adaptively
        4. Computing displacement with velocity Verlet-like update
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

        # Branchless reformulation of the original four if/elif blocks. The
        # original branched on whole-batch reductions (w_vf.all()/.any()),
        # each forcing a GPU->CPU sync. We instead keep `all_progressing` as a
        # 0-d on-device boolean tensor and select per-molecule with torch.where,
        # so no host-device sync occurs. The math is identical per molecule.
        #
        # The original is genuinely batch-dependent: a progressing molecule in
        # the "all progress" path (Case 1) only mixes its velocity and bumps
        # Nsteps (dt/a untouched), whereas in the "some progress" path (Case 2)
        # the dt/a/Nsteps speed-up applies only to progressing molecules past
        # Nmin. `all_progressing` is therefore required to reproduce Case 1 vs
        # Case 2 exactly.
        all_progressing = progressing.all()  # 0-d bool tensor (no .item() sync)
        past_nmin = self.Nsteps > self.Nmin
        # Speed-up applies only in the "some progress" branch (not all_progressing)
        # to progressing molecules that have exceeded Nmin steps.
        speedup = progressing & (~all_progressing) & past_nmin

        # Velocity mix toward normalized force direction (computed for every
        # molecule; selected below). Matches the original mixing expression.
        a3 = self.a.unsqueeze(-1).unsqueeze(-1)
        v_norm = self.v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        f_norm = forces.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        v_mixed = (1.0 - a3) * self.v + a3 * v_norm * forces / f_norm

        # v: mixed where progressing, reset to zero otherwise.
        prog3 = progressing.unsqueeze(-1).unsqueeze(-1)
        self.v = torch.where(prog3, v_mixed, torch.zeros_like(self.v))

        # dt:
        #   progressing & speedup        -> (dt * finc).clamp(max=dt_max)
        #   progressing & not speedup    -> unchanged
        #   not progressing              -> dt * fdec  (reset)
        dt_speedup = (self.dt * self.finc).clamp(max=self.dt_max)
        dt_prog = torch.where(speedup, dt_speedup, self.dt)
        self.dt = torch.where(progressing, dt_prog, self.dt * self.fdec)

        # a: same selection structure as dt.
        a_prog = torch.where(speedup, self.a * self.fa, self.a)
        self.a = torch.where(progressing, a_prog,
                             torch.full_like(self.a, self.astart))

        # Nsteps:
        #   progressing & (all_progressing | past_nmin) -> Nsteps + 1
        #   progressing & else                          -> unchanged
        #   not progressing                             -> 0  (reset)
        nsteps_inc = progressing & (all_progressing | past_nmin)
        nsteps_prog = torch.where(nsteps_inc, self.Nsteps + 1, self.Nsteps)
        self.Nsteps = torch.where(progressing, nsteps_prog,
                                  torch.zeros_like(self.Nsteps))

        # Velocity Verlet-like update
        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v = self.v + dt * forces
        dr = dt * self.v

        # Clamp maximum displacement per molecule
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        dr = dr * (self.maxstep / normdr).clamp(max=1.0)

        return coord + dr

    def clean(self, mask: torch.Tensor) -> bool:
        """Subset optimizer state to keep only specified molecules.

        This method is used to remove converged molecules from the optimization
        batch, reducing computation in subsequent steps.

        Args:
            mask: Boolean tensor of shape (batch,). True values indicate
                  molecules to keep in the optimization.

        Returns:
            Always returns True to indicate success.
        """
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True
