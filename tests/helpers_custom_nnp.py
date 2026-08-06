"""Scriptable custom-NNP stubs for tests.

torch.jit.script needs source access, so these live in a real module rather
than being defined inside a test function.
"""
from __future__ import annotations

import torch


class ScriptableNNP(torch.nn.Module):
    """Contract-conforming and TorchScript-safe.

    The padding values are INSTANCE attributes: TorchScript carries those into
    the archive, but silently drops plain class attributes.
    """

    def __init__(self):
        super().__init__()
        self.coord_pad: float = 0.0
        self.species_pad: int = -1

    def forward(
        self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
    ) -> torch.Tensor:
        return (coords ** 2).sum(dim=(1, 2))


class StochasticNNP(torch.nn.Module):
    """Contract-conforming, but its output depends on ``self.training``.

    Exists to pin the eval-mode guarantee: ``torch.jit.save`` records the
    module's ``training`` flag, so an archive saved by a careless author while
    still in train mode keeps dropout live at inference. A model whose energy
    changes between identical calls makes the FIRE loop chase noise, and the
    run reports ``Converged=False`` with no way to tell it from a floppy
    molecule.
    """

    def __init__(self):
        super().__init__()
        self.coord_pad: float = 0.0
        self.species_pad: int = -1
        self.drop = torch.nn.Dropout(0.5)

    def forward(
        self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
    ) -> torch.Tensor:
        return self.drop(coords ** 2).sum(dim=(1, 2))


class ClassAttrOnlyNNP(torch.nn.Module):
    """Padding values as class attributes only -- lost by torch.jit.save."""

    coord_pad: float = 0.0
    species_pad: int = -1

    def forward(
        self, species: torch.Tensor, coords: torch.Tensor, charges: torch.Tensor
    ) -> torch.Tensor:
        return (coords ** 2).sum(dim=(1, 2))
