"""The forward template every in-tree adapter shares, and nothing else.

Renamed from this package's single-file predecessor,
``Auto3D/engines/models/adapter.py`` (``git mv`` to this file; the other five
modules were then cut out of it).

Not the contract: :class:`Auto3D.engines.models.contract.ModelAdapter` -- see
:class:`BaseModelAdapter`'s own docstring for the distinction. This module
holds only that one class; each backend (:mod:`.aimnet2`, :mod:`.ani`,
:mod:`.custom`) imports it.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from Auto3D.engines.models.adapter.compiling import (
    _try_compile,
    _warn_if_compile_fell_back_to_eager,
)
from Auto3D.engines.models.adapter.validation import _validate_outputs
from Auto3D.engines.models.contract import spin_kwargs
from Auto3D.foundation.constants import DEFAULT_DUPLICATE_ENERGY_TOL


class BaseModelAdapter(ABC, nn.Module):
    """Implementation base for Auto3D's adapters. NOT the contract.

    The contract is :class:`Auto3D.engines.models.contract.ModelAdapter`, and that -- not
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
    - Concrete ``to_species`` (identity), ``energy`` and ``forward`` -- nothing
      here is abstract. Two supported shapes: (1) override :meth:`_energy_graph`
      only (and :meth:`_model_inputs` if the backend computes in float32);
      ``energy`` and ``forward`` are the one shared tail built on top of that
      hook. (2) override :meth:`forward` wholesale -- for a subclass that needs
      to compute forces itself (``AIMNet2Adapter``) -- and also
      :meth:`_energy_graph`, returning ``forward``'s first output, if
      :meth:`energy` is needed too; the hooks are one-directional, so nothing
      here falls back from one to the other. A subclass that supplies neither,
      itself or through a parent, is refused at class definition (see
      :meth:`__init_subclass__`); the full rule, including what the overrides
      must not call, is stated once in :meth:`_energy_graph`.

    Note on torch.inference_mode():
        This class CANNOT use torch.inference_mode() or torch.no_grad() in forward
        methods because force calculations require computing gradients of energy
        with respect to atomic coordinates via torch.autograd.grad(). Model parameters
        have requires_grad=False (frozen weights), but coordinates must have
        requires_grad=True for force computation. All autograd.grad() calls use
        create_graph=False to avoid building second-order gradient graphs.
    """

    #: eV. The duplicate-energy tolerance conformers from THIS backend are
    #: deduplicated with: how far two optimizations of one minimum can land
    #: apart at the convergence gate, which is a property of the backend's
    #: kernels and of where its forces leave that gate. The orchestrator reads
    #: it (through
    #: :func:`Auto3D.engines.models.contract.duplicate_energy_tol_of`) and
    #: passes it down to the conformer filter, which is a lower layer and
    #: cannot ask which engine ran.
    #:
    #: Every adapter shipped here inherits this value, the shared
    #: ``DEFAULT_DUPLICATE_ENERGY_TOL``, and none overrides it: the
    #: 2026-10-16 measurement of the three bundled engines
    #: (``benchmarks/results-notes/2026-10-16-batch-noise.md``) found floors
    #: that moved between two runs an hour apart, and the per-engine values it
    #: computed for AIMNet2 and ANI2x (0.050 and 0.098 eV) were 1.9 and 3.8
    #: times kT at 298 K, where the rotamer pairs the energy half of the
    #: duplicate test protects live; the third (ANI2xt, 0.011 eV) is within
    #: rounding of this constant. The attribute is the
    #: hook for an adapter that HAS a measurement to apply -- a custom NNP, or
    #: a future backend -- and overriding it changes only that adapter's runs.
    duplicate_energy_tol: float = DEFAULT_DUPLICATE_ENERGY_TOL

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass that supplies neither hook, at class-definition time.

        Before this check, a half-implemented subclass (forgot both
        :meth:`_energy_graph` and :meth:`forward`) would construct cleanly --
        ``forward`` is no longer ``@abstractmethod`` -- and only fail on its
        first real ``forward()``/``energy()`` call, deep inside the FIRE loop
        or the single-point-energy path. The test is identity against this
        class's own two functions, resolved through the subclass's MRO: a
        class anywhere in the chain that overrode either hook makes the
        subclass valid, so a grandchild of a working adapter that adds only
        unrelated methods passes, while a class whose ``forward`` AND
        ``_energy_graph`` are still exactly the base bodies is refused. A
        ``cls.__dict__`` check would wrongly reject that grandchild.
        """
        super().__init_subclass__(**kwargs)
        if (
            cls.forward is BaseModelAdapter.forward
            and cls._energy_graph is BaseModelAdapter._energy_graph
        ):
            raise TypeError(f"{cls.__name__} must define forward or _energy_graph")

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
                agree with ``Auto3D.engines.batch_opt.padding.pad_from_mols`` and with
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
        # How many suppressed frame compilations
        # _warn_if_compile_fell_back_to_eager has already reported for this
        # adapter. A running count, NOT a "checked yet" flag: Dynamo can fall
        # back long after the first forward (the recompile limit -- see that
        # method's docstring), so a one-shot gate spent its single observation
        # on a frame that always compiles cleanly and then stayed silent
        # forever (T-2).
        self._compile_suppressed_seen = 0

        # Disable gradients for model parameters (inference mode)
        for p in model.parameters():
            p.requires_grad_(False)

        # Optionally compile the model
        if compile_model:
            # Set BEFORE calling the hook: ANI2xAdapter's override resets
            # ``_compiled`` back to False when it refuses to compile, which
            # only takes effect if this assignment runs first.
            self._compiled = True
            model = self._compile(model)
            # Snapshot BEFORE this adapter's first forward, so the fallback
            # check reads a DELTA scoped to ITS OWN compilation rather than
            # the raw cumulative counter (issue #23). Both
            # ``suppress_errors`` and ``torch._dynamo.utils.counters`` are
            # process-global (see the containment comment on ``_try_compile``
            # for the first half of that fact) -- without this snapshot, an
            # in-process caller that already compiled and fell back once for
            # some earlier adapter would have that stale failure blamed on
            # the NEXT compiled adapter's unrelated first forward. A plain
            # ``dict`` copy, not a reference: ``counters["frames"]`` is a
            # live ``Counter`` that keeps mutating after this line.
            self._compile_frame_stats_before = dict(torch._dynamo.utils.counters.get("frames", {}))

        self.model = model

    def _compile(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile for this backend. Default: the whole module.

        Overridden where whole-module compilation is numerically wrong
        (ANI2xt: only the per-element networks compile safely; ANI2x: nothing
        does, see ANI2xAdapter). Called once from __init__ before the model is
        stored, so a subclass may compile a sub-component and return the same
        module object.
        """
        return _try_compile(model)

    def _warn_if_compile_fell_back_to_eager(self) -> None:
        """Log each time a NEW compiled frame silently fell back to eager.

        The bound entry point tests and
        :func:`Auto3D.engines.model_factory._rebaseline_compile_fallback_counters`
        name; the mechanism, the DELTA against ``_compile_frame_stats_before``,
        and the caveats on reading ``torch._dynamo.utils.counters`` (including
        the ``nvalchemiops`` helper aimnet calls, ``prepare_batch_idx_ptr``
        -- reached through ``neighbor_list`` -- compiling even when
        ``compile_model=False``) live on the module-level function this
        delegates to.
        """
        _warn_if_compile_fell_back_to_eager(self)

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

    def _model_inputs(
        self, coords: torch.Tensor, charges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What the backend is fed in :meth:`forward`; identity unless the backend is float32.

        A float32 backend (torchani's ANI2x, a custom NNP) overrides this to
        ``coords.float(), charges.float()``. The cast lives HERE and not in
        :meth:`energy` on purpose: ``energy`` must answer at the caller's dtype
        (a Hessian caller hands in float64), so it feeds the backend the
        tensors it was given. Only :meth:`forward`, whose caller is the FIRE
        loop with float32 coordinates, goes through this hook.
        """
        return coords, charges

    def _energy_graph(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        mults: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The backend's energies, graph-connected to ``coords``, at the dtype it produces.

        One-directional hook: this base body never calls :meth:`forward`, so
        there is no path back into the template from here. Every in-tree
        adapter but AIMNet2 overrides this with its one backend call;
        :meth:`forward` and :meth:`energy` are built on top of it.
        :class:`AIMNet2Adapter`, whose calculator computes forces itself,
        overrides :meth:`forward` wholesale instead and supplies its own
        one-line ``_energy_graph`` returning ``forward``'s first output --
        the same shape its ``energy`` would otherwise be as a conditional
        default, just stated here directly instead.

        The rule for subclasses, stated once here: override this hook, or
        override :meth:`forward` wholesale (and also this hook if ``energy``
        is needed). Neither override may call the other through the template:
        a ``forward`` override must not call ``super().forward()``, and a
        ``_energy_graph`` override must not call ``self.forward()`` unless
        ``forward`` is also overridden, because the base ``forward`` calls this
        hook and the cycle recurses with no diagnosis.

        ``mults`` is the spin multiplicity per molecule. A backend that READS it
        must accept the keyword; one that has no spin input accepts and ignores
        it, because ``energy`` and ``forward`` above pass it on whenever their
        own caller supplied one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _energy_graph (the backend's "
            "energy, graph-connected to coords); an override of forward must "
            "compute the energy itself and must not call super().forward(), "
            "which re-enters this hook."
        )

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        *,
        mults: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Energies only, graph-connected; never narrower than ``coords``' dtype.

        No silent downcast: a backend whose ``_model_inputs`` casts to float32
        for :meth:`forward` is bypassed here, so a float64 request gets a
        float64 answer rather than an fp32 one with no error. A backend that
        computes wider may still return wider -- :class:`AIMNet2Adapter`
        returns float64 energies whatever dtype it was handed, since its
        ``_energy_graph`` routes through its own ``forward``.

        This bypasses :meth:`forward` for two reasons, not because
        ``requires_grad_`` would raise on a non-leaf tensor (it would not: a
        non-leaf tensor's ``requires_grad`` already reads ``True``, so
        ``requires_grad_(True)`` on one is a no-op). First, routing through
        :meth:`forward` would run ``coords`` through :meth:`_model_inputs`,
        whose float32 cast is exactly the silent downcast the previous
        paragraph refuses. Second, :meth:`forward` always pays for one
        ``torch.autograd.grad`` call to compute forces that a pure energy
        request never asked for. No ``no_grad`` either: a Hessian caller
        needs the graph; a caller that does not wraps its own call site.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers or indexed species (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms.
            mults: spin multiplicity per molecule, (batch,), or None when every
                molecule is a singlet; backends without a spin input ignore it.
        """
        return self._energy_graph(coords, species, charges, atom_mask, **spin_kwargs(mults))

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        *,
        mults: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies and forces: the one tail every backend shares.

        Inputs pass through :meth:`_model_inputs` (a float32 backend casts
        there), ``coords`` is marked for autograd, :meth:`_energy_graph` is
        evaluated, forces are the negative gradient, :func:`_validate_outputs`
        refuses NaN or Inf, and the compile fallback check runs. Only the
        FORCES are cast back to the input dtype. The energy is returned at the
        dtype the backend produced (issue #5): ``forward``'s caller is the FIRE
        loop, which stores it as ``E_tot``, and casting a float64 backend's
        energy to float32 there quantized conformer energies at ~0.002 eV --
        enough to reorder two conformers 0.02 kcal/mol apart and to make
        ``auto3d energy`` (which goes through :meth:`energy`) disagree with the
        pipeline on the same geometry. Forces are consumed once, by the
        optimizer step, so rounding them costs nothing analogous.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Atomic numbers or indexed species (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms, threaded
                through from :func:`Auto3D.engines.batch_opt.padding.pad_from_mols`
                to the backend; most backends ignore it because their padding
                index is their own dummy-atom sentinel.
            mults: spin multiplicity per molecule, (batch,), or None when every
                molecule is a singlet; backends without a spin input ignore it.

        Returns:
            Tuple of (energies, forces); energies (batch,), forces
            (batch, n_atoms, 3), both in eV units.
        """
        input_dtype = coords.dtype
        coords_in, charges_in = self._model_inputs(coords, charges)
        coords_in = coords_in.requires_grad_(True)
        energy = self._energy_graph(coords_in, species, charges_in, atom_mask, **spin_kwargs(mults))
        # create_graph=False (default): no second-order graph.
        grad = torch.autograd.grad([energy.sum()], [coords_in], create_graph=False)[0]
        forces = -grad
        _validate_outputs(energy, forces)
        self._warn_if_compile_fell_back_to_eager()
        return energy, forces.to(input_dtype)

    def analytic_hessian(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        *,
        mults: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """No native second derivative: differentiate :meth:`energy` instead.

        ``None`` is the right default for ANI2xt, ANI2x and every custom NNP --
        all plain ``nn.Module``s with the whole energy in the autograd graph, so
        ``torch.autograd.functional.hessian`` of :meth:`energy` is exact for
        them. :class:`AIMNet2Adapter` is the sole override, because its energy
        pipeline includes external D3 and Coulomb modules that differentiating
        the bare module would drop.

        See :meth:`Auto3D.engines.models.contract.ModelAdapter.analytic_hessian`: this
        must never be used to swallow a failed native Hessian into ``None``.

        ``mults`` is accepted here, where it is unused, so a spin-aware
        subclass can use it without the Hessian caller having to ask which
        adapter it holds: the capability query and the spin input travel
        together or not at all.
        """
        return None

    def to_double(self) -> None:
        """Promote the wrapped module's weights to float64, in place.

        ``self.model.double()`` rather than ``self.double()``, although this
        class is itself an ``nn.Module`` and the two coincide for every adapter
        that registers no other child. They are not guaranteed to: ``Module.double``
        recurses into every registered submodule, so a subclass that holds a second
        one would have it upcast too. This is the exact operation
        ``Auto3D.entry.ASE.thermo`` performed before the call moved onto the contract,
        and keeping it exact is what makes "no reported frequency moves" a claim
        rather than a hope.
        """
        self.model.double()
