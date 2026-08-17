# tests/helpers_adapter.py
"""One conforming :class:`~Auto3D.models.contract.ModelAdapter` double.

``EnForce_ANI.__init__`` gates its first argument against the adapter contract
(``Auto3D.models.contract.ModelAdapter``), and ``pad_from_mols`` reads the
species convention *and* both padding sentinels off the same object. Before
this module every test that needed "something adapter-shaped" grew its own
duck-typed class declaring only the members that particular test happened to
exercise, so the gate could not be tightened without six unrelated files going
red -- and the cheapest way to make them green again would have been to weaken
the gate.

Everything here is a plain Python object: no ``nn.Module``, no weights, no
device traffic, and nothing is loaded or downloaded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class FakeAdapter:
    """A minimal object that satisfies the whole ``ModelAdapter`` contract.

    Args:
        coord_pad: Coordinate fill value reported to the padder.
        species_pad: Species fill value reported to the padder.
        species_map: Optional atomic-number -> model-species mapping applied by
            :meth:`to_species`. ``None`` means the identity, which is what every
            adapter except ``ANI2xtAdapter`` does.
        energy_fn: Optional ``(coords, species, charges) -> energies``. The
            default is ``sum(coords**2)`` per molecule, whose gradient is
            analytic (``2*coords``), so a caller can check forces without any
            model.
        hessian: What :meth:`analytic_hessian` returns. ``None`` -- the default,
            and every in-tree adapter's answer except AIMNet2's -- means "no
            native second derivative, differentiate ``energy`` instead".
    """

    def __init__(
        self,
        coord_pad: float = 0.0,
        species_pad: int = -1,
        species_map: dict[int, int] | None = None,
        energy_fn=None,
        hessian: torch.Tensor | None = None,
    ) -> None:
        self.coord_pad = coord_pad
        self.species_pad = species_pad
        self.species_map = species_map
        self._energy_fn = energy_fn
        self._hessian = hessian
        #: Recorded ``(coords_dtype, species, charges)`` per forward/energy call.
        self.calls: list[dict] = []

    # -- the species half of the contract ---------------------------------
    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        if self.species_map is None:
            return list(atomic_numbers)
        return [self.species_map[int(z)] for z in atomic_numbers]

    # -- the numerical half -----------------------------------------------
    def _energies(self, coords: torch.Tensor, species, charges) -> torch.Tensor:
        if self._energy_fn is not None:
            return self._energy_fn(coords, species, charges)
        return coords.pow(2).sum(dim=(1, 2))

    def forward(self, coords, species, charges, atom_mask=None):
        self.calls.append({"dtype": coords.dtype, "atom_mask": atom_mask, "kind": "forward"})
        coords = coords if coords.requires_grad else coords.detach().requires_grad_(True)
        energy = self._energies(coords, species, charges)
        grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
        return energy, -grad

    def energy(self, coords, species, charges, atom_mask=None):
        """Energies at the dtype of ``coords`` -- deliberately no downcast."""
        self.calls.append({"dtype": coords.dtype, "atom_mask": atom_mask, "kind": "energy"})
        return self._energies(coords, species, charges)

    def analytic_hessian(self, coords, species, charges):
        """The Hessian capability, ``None`` unless one was supplied."""
        self.calls.append({"dtype": coords.dtype, "kind": "analytic_hessian"})
        return self._hessian

    def to_double(self) -> None:
        """Recorded, not performed: there are no weights here to upcast.

        This double computes its energies from ``coords`` alone, so it is already
        dtype-preserving and an upcast has nothing to act on. Recording the call
        still matters -- it is how a test checks that the fp64 request reached the
        adapter at all.
        """
        self.calls.append({"kind": "to_double"})


class AdapterModuleMixin:
    """Makes an ``nn.Module`` test double satisfy ``ModelAdapter``.

    For doubles that must be real ``nn.Module``s (because the code under test
    reads ``.parameters()``, or ASE needs a module) and therefore cannot simply
    be :class:`FakeAdapter`. Mix in FIRST so these defaults are found before
    ``nn.Module``'s attribute machinery::

        class _Stub(AdapterModuleMixin, nn.Module):
            def forward(self, coords, species, charges, atom_mask=None): ...

    The values match ``BaseModelAdapter``'s own defaults. ``species_pad = -1``
    specifically: it can be neither a real atomic number nor a 0-based species
    index, so it cannot collide the way ``0`` did (audit C13).
    """

    coord_pad: float = 0.0
    species_pad: int = -1

    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        return list(atomic_numbers)

    def energy(self, coords, species, charges, atom_mask=None):
        return self.forward(coords, species, charges, atom_mask)[0]

    def analytic_hessian(self, coords, species, charges):
        """No native second derivative -- ``BaseModelAdapter``'s own default."""
        return None

    def to_double(self) -> None:
        """Upcast whatever this double registered, matching the real adapters.

        These doubles ARE ``nn.Module``s, so unlike :class:`FakeAdapter` there may
        genuinely be parameters to promote. ``self.double()`` rather than
        ``self.model.double()`` because a mixed-in double need not wrap a module
        at all -- several declare their forward inline and hold no ``.model``.
        """
        self.double()


def padded_batch(n_mols: int = 2, n_atoms: int = 3):
    """Tensors shaped like :func:`Auto3D.batch_opt.padding.pad_from_mols`."""
    coords = torch.zeros(n_mols, n_atoms, 3)
    species = torch.ones(n_mols, n_atoms, dtype=torch.long)
    charges = torch.zeros(n_mols)
    atom_mask = torch.ones(n_mols, n_atoms, dtype=torch.bool)
    return coords, species, charges, atom_mask
