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
        w_vf = vf > 0.0  # Mask for molecules making progress (v aligned with f)

        # Case 1: All molecules making progress
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            # Cache norms to avoid redundant computation
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            # Mix velocity with normalized force direction
            self.v = (1.0 - a) * v + a * v_norm * f / f_norm
            self.Nsteps += 1

        # Case 2: Some molecules making progress
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            # Cache norms to avoid redundant computation
            v_norm = v.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            f_norm = f.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
            self.v[w_vf] = (1.0 - a) * v + a * v_norm * f / f_norm

            # Increase time step and decrease mixing for molecules with enough steps
            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        # Handle molecules NOT making progress (need reset)
        w_vf = ~w_vf

        # Case 3: All molecules need reset
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = self.astart
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0

        # Case 4: Some molecules need reset
        elif w_vf.any():
            self.v[w_vf] = 0.0
            self.a[w_vf] = self.astart
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = 0

        # Velocity Verlet-like update
        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v

        # Clamp maximum displacement per molecule
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)

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
