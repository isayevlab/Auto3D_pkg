# src/Auto3D/models/adapter.py
"""Implementations of the adapter contract, one per NNP backend.

The contract itself -- :class:`Auto3D.models.contract.ModelAdapter` -- lives in
:mod:`Auto3D.models.contract`, next to the custom-NNP contract it is so easily
confused with. This module holds only implementations.

Layering: :mod:`Auto3D.models` is a leaf. It imports ``torch``,
``Auto3D.constants``, ``Auto3D.exceptions`` and its own submodules, and nothing
else from Auto3D. There is exactly one deliberate back-edge into
``Auto3D.batch_opt`` (``ANI2xtAdapter.__init__``'s deferred ``ANI2xt`` import);
see the comment there before moving it.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from Auto3D.constants import HARTREE_TO_EV
from Auto3D.exceptions import NumericalError
from Auto3D.models.ani2xt import ANI2xt, element_indices, self_atomic_energies
from Auto3D.models.loading import load_custom_nnp
from Auto3D.models.species import to_ani2xt_species


def _try_compile(model: nn.Module, mode: str = "default") -> nn.Module:
    """Attempt to compile a model with torch.compile.

    Args:
        model: The model to compile.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            Defaults to "default" and the model is compiled with dynamic=True.
            The optimization batch shrinks every step as conformers converge, and
            bucketing produces variable sub-batch sizes, so shapes are not static.
            "reduce-overhead" (CUDA graphs) requires static shapes and would
            trigger constant recompilation/guard failures here (review finding
            #24); dynamic default mode avoids that.

    Returns:
        The compiled model. Compilation is lazy, so this returns an
        ``OptimizedModule`` immediately and any Dynamo/Inductor failure surfaces
        at the **first forward** -- which happens inside the FIRE step loop,
        far from here.

        That is why there is no try/except around the call. An earlier version
        had one and its docstring promised "the original model if compilation
        fails"; it could never deliver that, because nothing fails here. The
        fallback is `suppress_errors` below, which is the mechanism that
        actually degrades to eager at the point of failure.
    """
    # Opting in to compilation opts in to falling back rather than crashing
    # mid-optimization: without this a graph break Inductor cannot handle takes
    # down a run that was already thousands of steps in.
    torch._dynamo.config.suppress_errors = True
    return torch.compile(model, mode=mode, fullgraph=False, dynamic=True)


def _raise_for_energy(energy: torch.Tensor) -> None:
    """Raise the NaN/Inf diagnosis for a known-non-finite energy tensor.

    Split out so the energy-only path
    (:func:`validate_energies`, reached from
    :meth:`Auto3D.batch_opt.model_wrapper.EnForce_ANI.energy_batched`) and the
    energy-and-forces path (:func:`_validate_outputs`) emit the SAME message for
    the same defect instead of two near-identical copies. Returns normally if
    the energy is finite after all, leaving the caller to decide what that means.
    """
    if torch.isnan(energy).any():
        nan_count = torch.isnan(energy).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} energy value(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(energy).any():
        inf_count = torch.isinf(energy).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} energy value(s). "
            "This may indicate atomic clashes or numerical overflow."
        )


def validate_energies(energy: torch.Tensor) -> None:
    """Reject a non-finite energy on a path that computed no forces.

    ``forward``'s :func:`_validate_outputs` used to be the only NaN gate a
    single-point energy passed through, so an energy-only path that skipped it
    would turn ``auto3d energy``'s exit-5 diagnosis into an SDF full of ``nan``.

    Args:
        energy: Energy tensor, shape (batch,).

    Raises:
        NumericalError: If NaN or Inf values are detected.
    """
    # One combined reduction (one host-device sync) on the happy path, for the
    # same reason as _validate_outputs below.
    if bool(torch.isfinite(energy).all()):
        return
    _raise_for_energy(energy)


def _validate_outputs(energy: torch.Tensor, forces: torch.Tensor) -> None:
    """Validate model outputs for numerical stability.

    Checks for NaN and Inf values in energy and force tensors, raising
    an exception if numerical instability is detected.

    Args:
        energy: Energy tensor from model forward pass.
        forces: Force tensor from model forward pass.

    Raises:
        NumericalError: If NaN or Inf values are detected.
    """
    # This runs on every NN forward, i.e. every FIRE step. Each `.any()`/`.item()`
    # on a CUDA tensor is a host-device sync that serializes the stream, so the
    # happy path (finite outputs) does a SINGLE combined reduction. The detailed,
    # additionally-synchronizing NaN/Inf breakdown is computed only on the rare
    # failure branch, where one extra sync is irrelevant.
    if bool(torch.isfinite(energy).all() & torch.isfinite(forces).all()):
        return

    _raise_for_energy(energy)
    if torch.isnan(forces).any():
        nan_count = torch.isnan(forces).sum().item()
        raise NumericalError(
            f"NaN detected in {nan_count} force component(s). "
            "This may indicate problematic molecular geometries."
        )
    if torch.isinf(forces).any():
        inf_count = torch.isinf(forces).sum().item()
        raise NumericalError(
            f"Inf detected in {inf_count} force component(s). "
            "This may indicate atomic clashes or numerical overflow."
        )


class BaseModelAdapter(ABC, nn.Module):
    """Implementation base for Auto3D's adapters. NOT the contract.

    The contract is :class:`Auto3D.models.contract.ModelAdapter`, and that -- not
    this class -- is what every signature that wants "an adapter" annotates.
    This distinction is the point: production has always accepted structural
    implementations (test doubles, and anything a downstream user writes), so
    annotating the ABC while accepting the Protocol is exactly what made the
    Protocol decorative. The one place this class legitimately appears as a type
    is ``ModelFactory._adapters``, a registry of Auto3D's OWN classes.

    Provides common functionality for all NNP model adapters including:
    - Model storage and device management
    - Padding value configuration
    - Gradient disabling for model parameters (weights are frozen)
    - Optional torch.compile() for performance optimization
    - Concrete ``to_species`` (identity) and ``energy`` defaults, so a subclass
      satisfies the contract by implementing ``forward`` alone

    Note on torch.inference_mode():
        This class CANNOT use torch.inference_mode() or torch.no_grad() in forward
        methods because force calculations require computing gradients of energy
        with respect to atomic coordinates via torch.autograd.grad(). Model parameters
        have requires_grad=False (frozen weights), but coordinates must have
        requires_grad=True for force computation. All autograd.grad() calls use
        create_graph=False to avoid building second-order gradient graphs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        coord_pad: float = 0.0,
        species_pad: int = -1,
        compile_model: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            model: The underlying neural network model.
            device: Target device for computations.
            coord_pad: Fill value for unused coordinate slots.
            species_pad: Fill value for unused species slots. Defaults to -1 to
                agree with ``Auto3D.batch_opt.padding.pad_from_mols`` and with
                the documented custom-NNP convention; the previous default of 0
                collided with ANI2xt's hydrogen index, so the two layers
                disagreed about which slots were padding. Every adapter below
                passes this explicitly, so the default only applies to
                third-party subclasses -- for which -1 is the safe value,
                because it can never be a real atomic number or a 0-based
                species index.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        super().__init__()
        self.device = device
        self.coord_pad = coord_pad
        self.species_pad = species_pad
        self._compiled = False

        # Disable gradients for model parameters (inference mode)
        for p in model.parameters():
            p.requires_grad_(False)

        # Optionally compile the model
        if compile_model:
            model = _try_compile(model)
            self._compiled = True

        self.model = model

    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        """Identity: this model consumes raw atomic numbers.

        Correct for AIMNet2, for ANI2x (constructed with
        ``periodic_table_index=True``), and for every custom NNP -- a custom
        model declares its own ``species_pad`` and receives atomic numbers, so
        remapping them here would silently feed every third-party model
        different species indices than its author tested against. ANI2xt is the
        sole override; see :meth:`ANI2xtAdapter.to_species`.
        """
        return list(atomic_numbers)

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies only, graph-connected, at the dtype of ``coords``.

        The default takes ``forward``'s first output. That is safe only for
        adapters whose ``forward`` is already dtype-preserving and does not
        mutate ``coords.requires_grad``; ``ANI2xtAdapter``, ``ANI2xAdapter`` and
        ``CustomModelAdapter`` each override it for exactly that reason (the
        latter two call ``coords.float()``, which would turn an fp64 caller's
        request into an fp32 answer with no error).

        No ``no_grad`` here, deliberately: a caller differentiating this (a
        Hessian) needs the graph, and a caller that does not want it wraps its
        own call site.
        """
        return self.forward(coords, species, charges, atom_mask)[0]

    def analytic_hessian(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor | None:
        """No native second derivative: differentiate :meth:`energy` instead.

        ``None`` is the right default for ANI2xt, ANI2x and every custom NNP --
        all plain ``nn.Module``s with the whole energy in the autograd graph, so
        ``torch.autograd.functional.hessian`` of :meth:`energy` is exact for
        them. :class:`AIMNet2Adapter` is the sole override, because its energy
        pipeline includes external D3 and Coulomb modules that differentiating
        the bare module would drop.

        See :meth:`Auto3D.models.contract.ModelAdapter.analytic_hessian`: this
        must never be used to swallow a failed native Hessian into ``None``.
        """
        return None

    @abstractmethod
    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms and False
                for padded slots, threaded through from
                :func:`Auto3D.batch_opt.padding.pad_from_mols`. ``None`` means
                "every slot is a real atom" and is correct only for an
                unpadded batch. Subclasses that need to know which slots are
                padding must use THIS mask, never a comparison against
                ``self.species_pad`` (audit C13).

        Returns:
            Tuple of (energies, forces) where energies has shape (batch,)
            and forces has shape (batch, n_atoms, 3). Units: eV.
        """
        ...


class AIMNet2Adapter(BaseModelAdapter):
    """Adapter for AIMNet2 models served by the `aimnet` package.

    Models are resolved by registry name/alias (e.g. 'aimnet2',
    'aimnet2-2025', 'aimnet2-nse') and auto-downloaded + sha256-validated
    into ~/.cache/aimnet on first use. Supports charged molecules and the
    full AIMNet2 element set.

    The optimizer feeds a padded (B, N, 3) batch. AIMNet2 does not tolerate
    padding atoms (species 0 at the origin yields NaN), so this adapter
    flattens real atoms and uses the calculator's ragged `mol_idx` batching,
    then scatters forces back into the padded (B, N, 3) layout (padded slots
    receive zero force).
    """

    def __init__(
        self,
        model_name: str = "aimnet2",
        device: torch.device | None = None,
        compile_model: bool = False,
    ) -> None:
        """Initialize the AIMNet2 adapter.

        Args:
            model_name: aimnet registry name/alias.
            device: Target device.
            compile_model: Forwarded to AIMNet2Calculator (torch.compile).
        """
        from aimnet.calculators import AIMNet2Calculator

        if device is None:
            device = torch.device("cpu")
        self.model_name = model_name
        calc = AIMNet2Calculator(model_name, device=device, compile_model=compile_model)
        super().__init__(calc.model, device, coord_pad=0.0, species_pad=0, compile_model=False)
        self._calc = calc

    def analytic_hessian(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """AIMNet2's native analytic Hessian, through the FULL energy pipeline.

        The external D3 dispersion and Coulomb modules are part of that
        pipeline. Differentiating this adapter's ``.model`` instead silently
        drops them (D3 is attractive at bonding range), stiffening every bond
        and shifting C-H stretches up by ~4%, ~130 cm-1, with nothing in the
        output signalling it -- which is why this override exists rather than
        letting the base class's ``None`` send AIMNet2 down the autograd path.

        This method replaced the ``calculator`` property this class used to
        publish purely so ``Auto3D.ASE.thermo._load_hessian_model`` could hand
        the raw ``AIMNet2Calculator`` back to a caller that then dispatched on
        ``isinstance(model, AIMNet2Calculator)``. The capability now lives on the
        contract, so the third-party type no longer appears in Auto3D's control
        flow and ``_load_hessian_model`` has one return type instead of two.

        Args:
            coords: (1, n_atoms, 3). fp32 in practice -- whole-graph fp64
                through AIMNet2 would be false precision, so unlike the ANI /
                custom autograd path this one is not upcast.
            species: atomic numbers, (1, n_atoms). Identity-mapped by
                :meth:`BaseModelAdapter.to_species`.
            charges: molecular charge, (1,). Passed to the calculator exactly as
                received (no dtype coercion): the calculator prepares its own
                input tensors, and casting here would change the numbers this
                path has always produced.

        Returns:
            Hessian in eV/A^2, shape ``(n_atoms, 3, n_atoms, 3)`` as aimnet
            returns it.
        """
        result = self._calc(
            {"coord": coords, "numbers": species, "charge": charges},
            hessian=True,
        )
        return result["hessian"]

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies (eV) via ``forward``; the base default, made explicit.

        Written out rather than inherited so the dtype reasoning is on the record
        for the one adapter where it differs. ``forward`` returns float64
        energies whatever it was fed -- an UPCAST, so there is no silent
        precision loss to guard against (the hazard the other two overrides
        exist for), and whole-graph fp64 through AIMNet2 would be false
        precision regardless. Routed through ``forward`` (hence the calculator's
        ``forces=True`` path) because that is the route the calculator
        guarantees stays connected to ``coord`` in the autograd graph.
        """
        return self.forward(coords, species, charges, atom_mask)[0]

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies (eV) and forces (eV/A) for a padded batch.

        Args:
            coords: (batch, n_atoms, 3); padded slots at coord_pad.
            species: atomic numbers (batch, n_atoms); padded slots at
                species_pad. The VALUE is never inspected here -- see below.
            charges: molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms, as
                returned by :func:`Auto3D.batch_opt.padding.pad_from_mols`.
                Required for a padded batch. ``None`` means every slot is a
                real atom, which is what the unpadded single-molecule callers
                (``ASE/thermo.py``'s ASE Calculator, ``auto3d models test``)
                want.

        Returns:
            (energy[batch], forces[batch, n_atoms, 3]) in eV and eV/A. Padded
            atom slots have zero force.

        The real-atom mask is the caller's explicit ``atom_mask``, NEVER
        ``species != self.species_pad``. This adapter's ``species_pad`` is 0
        and it consumes raw atomic numbers, so the sentinel comparison deleted
        atomic number 0 -- an R-group/dummy ``*`` atom, which
        ``utils.validation._requires_aimnet`` routes to precisely this engine.
        For ``*CCO`` the padder reported 9 real atoms and this adapter scored
        8: the energy belonged to a different species, and the dummy atom got
        exactly zero force and stayed frozen for the whole optimization. That
        is the collision class ``padding.pad_from_mols`` documents (audit C13).
        """
        b, n = species.shape[0], species.shape[1]
        if atom_mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=species.device)
        else:
            mask = atom_mask.to(device=species.device, dtype=torch.bool)
        coord_flat = coords[mask]  # (M, 3)
        numbers_flat = species[mask]  # (M,)
        mol_idx = torch.arange(b, device=species.device).unsqueeze(1).expand(b, n)[mask]  # (M,)

        result = self._calc(
            {
                "coord": coord_flat,
                "numbers": numbers_flat,
                "charge": charges.to(coord_flat.dtype),
                "mol_idx": mol_idx,
            },
            forces=True,
        )
        energy = result["energy"].reshape(-1).to(torch.double)  # (B,)
        forces_flat = result["forces"].reshape(-1, 3)  # (M, 3)

        forces = torch.zeros(b, n, 3, dtype=forces_flat.dtype, device=forces_flat.device)
        forces[mask] = forces_flat
        _validate_outputs(energy, forces)
        return energy, forces


class ANI2xtAdapter(BaseModelAdapter):
    """Adapter for ANI2xt model.

    ANI2xt is a retrained version of ANI with improved performance.
    Uses indexed species (H=0, C=1, N=2, O=3, F=4, S=5, Cl=6).

    ``compile_model=True`` compiles ``ANI2xt.forward``. Until this change it compiled
    *nothing*: ``forward``'s per-element loop contained a data-dependent branch
    (``if mask.any():``), and a graph break inside a loop gives Dynamo nowhere to
    place a resume point, so it skipped the frame -- measured as **zero**
    compiled subgraphs. The loop is now free of data-dependent ops and compiles
    to one subgraph (``tests/test_ani2xt_atom_energies.py``). Whether that is a
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

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies (eV) with no ``requires_grad_`` mutation of ``coords``.

        ``forward`` calls ``coords.requires_grad_(True)`` because it must
        differentiate to get forces. ``energy`` cannot: an autograd-Hessian
        caller hands in a NON-LEAF tensor, and ``requires_grad_`` on a non-leaf
        raises. Energies come out float64 (see ``ANI2xt.forward``); coords are
        consumed at whatever dtype they arrive in.
        """
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

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2xt.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Indexed atomic species (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2xt.
            atom_mask: Accepted for interface uniformity and deliberately
                unused: ANI consumes ``species_pad = -1`` as its own dummy-atom
                index, so the model itself skips padded slots. -1 is not a
                sentinel this adapter compares against, and it can never
                collide with a real 0-based species index.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        coords = coords.requires_grad_(True)
        energy = self._call_model(species, coords)
        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords], create_graph=False)[0]
        forces = -grad
        _validate_outputs(energy, forces)
        return energy, forces


class ANI2xAdapter(BaseModelAdapter):
    """Adapter for ANI2x model from TorchANI.

    ANI2x uses periodic table indexing for species.
    Requires torchani to be installed.

    ``compile_model=True`` compiles the torchani model. No speedup figure is
    claimed: none has been measured for this path, and the ~1.25x these
    docstrings used to assert had no measurement behind it.
    ``benchmarks/bench_optimization_perf.py`` is what produces one.
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

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies (eV) at the dtype of ``coords`` -- NO float32 downcast.

        This override exists solely to keep that promise. ``forward`` calls
        ``coords.float()`` for compatibility with torchani's float32 weights;
        inheriting ``BaseModelAdapter.energy`` (which is ``forward(...)[0]``)
        would therefore turn an fp64 caller's request into an fp32 answer with no
        error, no warning, and no way to notice -- the caller that wants fp64 is
        computing a Hessian, and it would silently get an fp32 one. Feeding the
        model the dtype it was handed pushes that choice back to the caller,
        which is the layer that also has to promote the model's weights
        (``.double()``) for it to be meaningful.
        """
        return self.model((species, coords)).energies * HARTREE_TO_EV

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using ANI2x.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers (batch, n_atoms).
            charges: Molecular charges (batch,) - not used by ANI2x.
            atom_mask: Accepted for interface uniformity and deliberately
                unused: torchani consumes ``species_pad = -1`` as its own
                dummy-atom index, which can never collide with a real atomic
                number.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        # Convert to float32 for ANI2x (it uses float32 internally)
        input_dtype = coords.dtype
        coords_f32 = coords.float().requires_grad_(True)

        energy = self.model((species, coords_f32)).energies * HARTREE_TO_EV
        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords_f32], create_graph=False)[0]
        forces = -grad

        _validate_outputs(energy, forces)
        # Convert back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for user-provided custom NNP models.

    Custom models implement the contract defined in
    ``Auto3D.models.contract`` (:class:`~Auto3D.models.contract.CustomNNP`):
    - ``forward(species, coords, charges) -> energies`` -- species FIRST, and
      energies only. This adapter derives forces from the returned energy by
      autograd, so the model must not return them.
    - ``coord_pad`` and ``species_pad`` attributes. Both are REQUIRED; a missing
      one is rejected at load rather than silently defaulted.

    Note the argument order is the reverse of this adapter's own
    ``forward(coords, species, charges)``, which is Auto3D's internal
    :class:`ModelAdapter` interface and returns ``(energies, forces)``.
    ``load_custom_nnp`` rejects a model that confuses the two.

    The model file may be EITHER a TorchScript archive
    (``torch.jit.script(m).save(path)``) OR an eager nn.Module saved with
    ``torch.save(m, path)``; the adapter auto-detects. Eager loading is required
    because modern AIMNet2-based models are no longer torch.jit.script-able.

    Note: if your model pads batches, use a non-zero ``species_pad`` -- some
    backends (e.g. AIMNet2) produce NaN on species-0 padded atoms.

    Note: Custom models have limited torch.compile() benefits.

    Note: inputs are cast to float32 before the forward pass. If your NNP
    requires float64 precision (e.g. for very small energy differences),
    wrap it to upcast internally, as Auto3D will feed it float32 coordinates.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        compile_model: bool = False,
    ) -> None:
        """Initialize custom model adapter.

        Args:
            model_path: Path to the TorchScript model file.
            device: Target device for computations.
            compile_model: Whether to apply torch.compile() for optimization.
        """
        # Accept either a TorchScript archive or an eager nn.Module checkpoint
        # (shared load contract -- see Auto3D.models.loading.load_custom_nnp).
        # load_custom_nnp validates the contract, so coord_pad/species_pad are
        # guaranteed present here. Reading them directly (rather than through
        # getattr defaults that disagreed with BaseModelAdapter's) is what keeps
        # one padding value in play instead of two.
        model = load_custom_nnp(model_path, device)
        # TorchScript archives are already a compiled graph, so `torch.compile`
        # has nothing to add; an eager `nn.Module` -- which `load_custom_nnp`
        # also accepts, and which every AIMNet2-derived custom model is -- does
        # benefit. Honouring the flag only in that case is what stops
        # `compile_model=True` from being silently ignored on the one adapter a
        # user supplies the model for.
        compile_custom = compile_model and not isinstance(model, torch.jit.ScriptModule)
        if compile_model and not compile_custom:
            warnings.warn(
                "compile_model=True ignored: a TorchScript archive is already a "
                "compiled graph. Save the model eagerly (torch.save) to use "
                "torch.compile.",
                stacklevel=2,
            )
        super().__init__(
            model, device, model.coord_pad, model.species_pad, compile_model=compile_custom
        )

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies (eV) at the dtype of ``coords`` -- NO float32 downcast.

        Same reason as :meth:`ANI2xAdapter.energy`: ``forward`` casts coords and
        charges to float32 (documented in the class docstring), so inheriting the
        ``forward(...)[0]`` default would silently answer an fp64 request in fp32.
        ``charges`` follows ``coords``' dtype so a model that indexes or
        concatenates the two does not hit a mismatch.

        The published contract is ``forward(species, coords, charges)`` -- species
        FIRST -- and that order is the user's, not this adapter's; see
        :class:`Auto3D.models.contract.CustomNNP`.
        """
        return self.model(species, coords, charges.to(coords.dtype))

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces using custom model.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers or indexed species (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Accepted for interface uniformity and NOT forwarded:
                the published custom-NNP contract
                (:class:`Auto3D.models.contract.CustomNNP`) is
                ``forward(species, coords, charges)``, so a user model
                identifies its own padding from the ``species_pad`` value it
                declared. Choose a ``species_pad`` that cannot collide with a
                real species index (-1 is always safe); see the class
                docstring.

        Returns:
            Tuple of (energies, forces) in eV units.
        """
        # Intentional downcast to float32 for compatibility with most NNP
        # models (e.g., ANI2x). This silently loses precision for fp64 models;
        # such models should upcast internally (see class docstring).
        input_dtype = coords.dtype
        coords_f32 = coords.float().requires_grad_(True)
        charges_f32 = charges.float()

        energy = self.model(species, coords_f32, charges_f32)

        # create_graph=False (default) avoids building second-order gradient graph
        grad = torch.autograd.grad([energy.sum()], [coords_f32], create_graph=False)[0]
        forces = -grad

        _validate_outputs(energy, forces)
        # Convert output back to input dtype for consistency
        return energy.to(input_dtype), forces.to(input_dtype)
