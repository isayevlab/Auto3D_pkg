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
    AEV computer stays eager because compiling it would corrupt the energies:
    torchani's AEV computer compiles *successfully* under ``torch.compile``,
    but returns energies off by hundreds of eV with no error raised. Only
    ``_atom_energies_fn`` -- the per-element loop, free of a data-dependent
    branch that would otherwise make it uncompilable on its own -- is
    compiled (``tests/test_ani2xt_atom_energies.py``). Whether that is a
    wall-clock win, and by how much, is a GPU measurement this repository does
    not make; see ``benchmarks/bench_optimization_perf.py``. No speedup figure
    is claimed here because none has been measured.
    """

    #: eV (0.25 kcal/mol), five times the floor of the 2026-10-16 measurement
    #: this value was taken from (2.04e-03 eV), rounded up to two significant
    #: figures; a second sample that day measured 2.10e-03 eV, which this
    #: value still stands 5.2x above. See :class:`BaseModelAdapter` for what
    #: the attribute is, and
    #: ``benchmarks/results-notes/2026-10-16-batch-noise.md`` for the
    #: measurement. The difference from the shipped fallback (0.01 eV) is
    #: within the rounding, so deduplication on this engine is effectively
    #: unchanged by it.
    duplicate_energy_tol: float = 0.011

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
        # it, which is why compile_model=True would otherwise produce zero
        # subgraphs for this model. Bound as instance attributes here (both
        # are already imported at module scope, above) so `_call_model`'s `getattr(self,
        # "_element_indices", None)` can tell a real ANI2xt apart from the toy
        # models several tests substitute by bypassing __init__ -- see
        # `_call_model`'s docstring.
        self._element_indices = element_indices
        self._self_atomic_energies = self_atomic_energies
        self._num_elements = num_elements
        self._energy_shifts = energy_shifts

    def _compile(self, model: nn.Module) -> nn.Module:
        # Compile only the per-element MLP evaluation. Compiling the whole
        # module (which includes torchani's AEVComputer) gives energies off by
        # hundreds of eV with no error raised. The AEV stays eager.
        #
        # Attempted, not landed: each distinct element-presence
        # pattern across a run 0/1-specializes `element_indices`'s seven
        # per-element row-index tensors, so past
        # `torch._dynamo.config.recompile_limit` of those patterns the frame
        # silently falls back to eager (measured: total 9, ok 8).
        # `torch._dynamo.mark_dynamic` on `aev_flat` and each `elem_index[i]`
        # does not fix this on torch 2.9.1+cu128: with seven simultaneously
        # varying, frequently 0/1-sized dims, Dynamo's own 0/1 specialization
        # recompiles anyway ("to avoid specialization try
        # torch._dynamo.mark_unbacked"); `mark_unbacked` instead raises
        # `GuardOnDataDependentSymNode: Eq(u7, 0) | Eq(u7, 1)` inside
        # Inductor's constant-folding pass; and stacking
        # `torch.fx.experimental._config.backed_size_oblivious = True` with
        # `use_duck_shape = False` and
        # `torch._inductor.config.joint_graph_constant_folding = False` only
        # trades that failure for a broadcast-lowering guard on the same
        # size-1 question. No config combination tried removes the
        # specialization. The compile below is unchanged from before this was
        # attempted -- no mark_dynamic wrapper is installed, rather than
        # landing a closure whose docstring claims a fix that does not hold.
        model._atom_energies_fn = _try_compile(model._atom_energies_fn)
        return model

    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        """Remap atomic numbers to ANI2xt's 0-based network indices.

        ANI2xt's species mapping is fixed to atomic-number order with no
        periodic-table option, so its ``forward`` expects H=0, C=1, N=2, O=3,
        F=4, S=5, Cl=6 -- not atomic numbers. The remap lives on the adapter
        (rather than in a name-keyed free function the caller had to
        remember to invoke) so it cannot be omitted at one call site and
        applied at another, where thermo and the CLI health check could
        then silently score a different molecule than the one submitted.

        Raises:
            ValueError: An atomic number outside ANI2xt's element set.
        """
        return to_ani2xt_species(atomic_numbers)

    def _energy_graph(self, coords, species, charges, atom_mask=None, mults=None):
        """ANI2xt's float64 totals; ``charges``, ``atom_mask`` and ``mults`` are unused
        (its padding index is -1, and it has no spin input)."""
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
    torchani's AEV path corrupts energies.
    """

    #: eV (0.92 kcal/mol), five times the floor of the 2026-10-16 measurement
    #: this value was taken from (7.81e-03 eV), rounded up to two significant
    #: figures. See :class:`BaseModelAdapter` for what the attribute is, and
    #: ``benchmarks/results-notes/2026-10-16-batch-noise.md`` for the
    #: measurement.
    #:
    #: This engine's total energy is itself a float32 quantity, quantized at
    #: 2-4e-3 eV above |E| ~ 2e4 eV, so its floor counts those steps rather
    #: than measuring where the optimizer stopped: the sample behind this
    #: value saw two steps, and a second sample that day saw five
    #: (1.95e-02 eV, verified as one minimum reached twice at 0.10 A
    #: heavy-atom RMSD), which this value covers by only 2.05x. The step
    #: doubles with each doubling of |E|, so the value is calibrated for
    #: molecules up to the |E| that bench set reaches, 3.9e4 eV (41 atoms for
    #: a sulfur-containing molecule there; |E| per atom depends on
    #: composition). Above that, the same number of steps is a larger energy.
    duplicate_energy_tol: float = 0.040

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
        # thousands of eV. Refuse rather than risk it.
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

    def _energy_graph(self, coords, species, charges, atom_mask=None, mults=None):
        """torchani's energies in eV at the dtype it produces: its float32 self-energy
        buffer, so a total above |E| ~ 2e4 eV is quantized at 2-4e-3 eV (torchani 2.8.4).

        ``charges``, ``atom_mask`` and ``mults`` are unused: ANI2x has no charge
        and no spin input, and its padding index is -1."""
        return self.model((species, coords)).energies * HARTREE_TO_EV
