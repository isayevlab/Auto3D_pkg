"""The two torchani-family adapters: ANI2xt (indexed species) and ANI2x (periodic table).

Cut from this package's single-file predecessor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from Auto3D.engines.models.adapter.base import BaseModelAdapter
from Auto3D.engines.models.adapter.compiling import _try_compile
from Auto3D.engines.models.ani2xt import ANI2xt, element_indices, self_atomic_energies
from Auto3D.engines.models.species import to_ani2xt_species
from Auto3D.foundation.constants import HARTREE_TO_EV
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


class ANI2xtAdapter(BaseModelAdapter):
    """Adapter for ANI2xt model.

    ANI2xt is a retrained version of ANI with improved performance.
    Uses indexed species (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6).

    ``compile_model=True`` compiles the per-element network evaluation only; the
    AEV computer stays eager because compiling it corrupts the energies (P-C1,
    2026-09-21). Before this change, ``compile_model=True`` compiled all of
    ``forward``, including torchani's AEV computer -- that compiled
    *successfully* and returned energies off by hundreds of eV with no error
    raised (P-C1, 2026-09-21). Now only ``_atom_energies_fn`` -- the
    per-element loop, separately rewritten (M7) to be free of the
    data-dependent branch that used to make it uncompilable on its own -- is
    compiled (``tests/test_ani2xt_atom_energies.py``). Whether that is a
    wall-clock win, and by how much, is a GPU measurement this repository does
    not make; see ``benchmarks/bench_optimization_perf.py``. No speedup figure
    is claimed here because none has been measured.
    """

    def __init__(self, device: torch.device, compile_model: bool = False) -> None:
        """Initialize ANI2xt adapter.

        Args:
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        model = ANI2xt(device)
        num_elements = len(model.networks)
        energy_shifts = model.energy_shifts
        super().__init__(model, device, coord_pad=0.0, species_pad=-1, compile_model=compile_model)
        # Precompute-and-pass plumbing for ANI2xt.forward. Both helpers are pure
        # functions of `species`, and both have to be called from *outside*
        # ANI2xt.forward for it to be compilable at all: element_indices has a
        # data-dependent output shape, and a graph break inside forward's
        # per-element loop makes Dynamo skip the whole frame rather than split
        # it, which is why compile_model=True used to produce zero subgraphs for
        # this model. Bound here rather than imported at module scope because
        # models -> batch_opt is the one deliberate back-edge and must stay
        # inside a method (see the deferred ANI2xt import above).
        self._element_indices = element_indices
        self._self_atomic_energies = self_atomic_energies
        self._num_elements = num_elements
        self._energy_shifts = energy_shifts

    def _compile(self, model: nn.Module) -> nn.Module:
        # Compile only the per-element MLP evaluation. Compiling the whole
        # module (which includes torchani's AEVComputer) gives energies off by
        # hundreds of eV with no error raised (P-C1). The AEV stays eager.
        model._atom_energies_fn = _try_compile(model._atom_energies_fn)
        return model

    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        """Remap atomic numbers to ANI2xt's 0-based network indices.

        ANI2xt is built with ``periodic_table_index=False`` everywhere, so its
        ``forward`` expects H=0, C=1, N=2, O=3, F=4, S=5, Cl=6 -- not atomic
        numbers. The remap lives on the adapter (rather than in a name-keyed free
        function the caller had to remember to invoke) so it cannot be omitted at
        one call site and applied at another; that omission is audit findings
        C3/C4, where thermo and the CLI health check silently scored a different
        molecule than the one submitted.

        Raises:
            ValueError: An atomic number outside ANI2xt's element set.
        """
        return to_ani2xt_species(atomic_numbers)

    def _energy_graph(self, coords, species, charges, atom_mask=None):
        """ANI2xt's float64 totals; ``charges`` and ``atom_mask`` are unused (its padding index is -1)."""
        return self._call_model(species, coords)

    def _call_model(self, species: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Invoke ``ANI2xt.forward`` with the species-only terms precomputed.

        ``element_indices`` collapses seven ``nonzero`` calls -- seven
        host-device synchronizations per forward on CUDA -- into a single host
        readback of a fixed-size count vector, and ``self_atomic_energies``
        removes a seven-iteration Python loop that recomputed a constant. Doing
        both out here rather than inside ``forward`` is also what leaves
        ``forward`` free of data-dependent ops, so ``compile_model=True`` has a
        frame it can actually compile.

        Not cached across calls: the optimization loop gathers a fresh
        ``species`` tensor for the still-active subset on every step, so there is
        no object whose identity could key a cache, and a content-keyed cache
        would cost the comparison it saves.

        Falls back to the plain two-argument call when the helpers are absent,
        which happens whenever ``self.model`` is not a real ``ANI2xt`` -- several
        tests bypass ``__init__`` and substitute a toy quadratic model so they
        can exercise the *real* ``forward``/``energy`` without loading weights or
        importing torchani. The precompute is an optimization, not part of the
        model contract, so it degrades rather than breaking.
        """
        helper = getattr(self, "_element_indices", None)
        if helper is None:
            return self.model(species, coords)
        elem_index = helper(species, self._num_elements)
        self_energies = self._self_atomic_energies(species, self._energy_shifts, self._num_elements)
        return self.model(species, coords, elem_index=elem_index, self_energies=self_energies)


class ANI2xAdapter(BaseModelAdapter):
    """Adapter for ANI2x model from TorchANI.

    ANI2x uses periodic table indexing for species.
    Requires torchani to be installed.

    ``compile_model=True`` is ignored with a warning: torch.compile of
    torchani's AEV path corrupts energies (P-C1, 2026-09-21).
    """

    def __init__(self, device: torch.device, compile_model: bool = False) -> None:
        """Initialize ANI2x adapter.

        Args:
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        import torchani

        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        super().__init__(model, device, coord_pad=0.0, species_pad=-1, compile_model=compile_model)

    def _compile(self, model: nn.Module) -> nn.Module:
        # torchani's ANI2x has no separable network module at the top level
        # (children: neighborlist, energy_shifter, species_converter,
        # potentials), and compiling the whole model corrupts energies by
        # thousands of eV (P-C1, 2026-09-21). Refuse rather than risk it.
        logger.warning(
            "compile_model=True is not supported for ANI2x; running eager. "
            "torch.compile of torchani's AEV path produces wrong energies."
        )
        self._compiled = False
        return model

    def _model_inputs(self, coords, charges):
        # torchani's weights are float32; the cast is in forward's hook only, so
        # energy() still answers a float64 request in float64 (issue #5).
        return coords.float(), charges.float()

    def _energy_graph(self, coords, species, charges, atom_mask=None):
        """torchani's energies in eV at the dtype it produces: its float32 self-energy
        buffer, so a total above |E| ~ 2e4 eV is quantized at 2-4e-3 eV (torchani 2.8.4)."""
        return self.model((species, coords)).energies * HARTREE_TO_EV
