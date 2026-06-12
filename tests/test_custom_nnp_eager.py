"""A custom NNP saved as an eager nn.Module (torch.save) must load and run.

Modern AIMNet2-based models are no longer torch.jit.script-able, so the
custom-NNP adapter must accept eager modules, not only TorchScript archives.
This needs no torchani and no aimnet -- a trivial analytic energy module suffices.
"""
from __future__ import annotations

import torch


class _TinyNNP(torch.nn.Module):
    """E = sum(coord^2) over real atoms; ignores charges. Follows the custom-NNP
    contract: forward(species, coords, charges) -> energies, plus coord_pad /
    species_pad attributes."""

    coord_pad = 0.0
    species_pad = -1

    def forward(self, species, coords, charges):
        mask = (species != self.species_pad).unsqueeze(-1)
        return (coords * mask).pow(2).sum(dim=(1, 2))


def test_custom_eager_module_loads_and_runs(tmp_path):
    from Auto3D.models.adapter import CustomModelAdapter

    path = tmp_path / "tiny_eager.pt"
    torch.save(_TinyNNP(), path)  # eager module, NOT torch.jit.script

    adapter = CustomModelAdapter(str(path), torch.device("cpu"))
    # The adapter is invoked as forward(coords, species, charges) -- coords first
    # (EnForce_ANI.forward delegates self.model.forward(coord, numbers, charges)).
    coords = torch.randn(1, 3, 3)
    species = torch.tensor([[1, 6, -1]])  # last atom is padding
    charges = torch.zeros(1)
    e, f = adapter.forward(coords, species, charges)
    assert e.shape == (1,) and torch.isfinite(e).all()
    assert f.shape == coords.shape and torch.isfinite(f).all()
