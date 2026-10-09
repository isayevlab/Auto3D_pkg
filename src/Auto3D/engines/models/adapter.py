"""Implementations of the adapter contract, one per NNP backend.

The contract itself -- :class:`Auto3D.engines.models.contract.ModelAdapter` -- lives in
:mod:`Auto3D.engines.models.contract`, next to the custom-NNP contract it is so easily
confused with. This module holds only implementations.

Layering: :mod:`Auto3D.engines.models` is a leaf. It imports ``torch``,
``Auto3D.foundation.constants``, ``Auto3D.foundation.exceptions``,
``Auto3D.foundation.utils.logging_config`` (L0, same as the other two -- for the
one-time compile-fallback diagnostic below, issue #23) and its own submodules,
and nothing else from Auto3D. There is exactly one deliberate back-edge into
``Auto3D.engines.batch_opt`` (``ANI2xtAdapter.__init__``'s deferred ``ANI2xt`` import);
see the comment there before moving it.
"""

from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from Auto3D.engines.models.ani2xt import ANI2xt, element_indices, self_atomic_energies
from Auto3D.engines.models.availability import require_aimnet
from Auto3D.engines.models.loading import load_custom_nnp
from Auto3D.engines.models.species import to_ani2xt_species
from Auto3D.foundation.constants import HARTREE_TO_EV
from Auto3D.foundation.exceptions import NumericalError
from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)


def _try_compile(model: Callable[..., Any] | nn.Module, mode: str = "default") -> Any:
    """Attempt to compile a model with torch.compile.

    Args:
        model: The model to compile. An ``nn.Module`` for
            :meth:`BaseModelAdapter._compile` (whole-module compilation), but a
            plain **function** for :meth:`ANI2xtAdapter._compile`, which
            compiles ``ANI2xt._atom_energies_fn`` and leaves the AEV computer
            eager -- ``torch.compile`` accepts either, and the annotation has
            to say so.
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            Defaults to "default" and the model is compiled with dynamic=True.
            The optimization batch shrinks every step as conformers converge, and
            bucketing produces variable sub-batch sizes, so shapes are not static.
            "reduce-overhead" (CUDA graphs) requires static shapes and would
            trigger constant recompilation/guard failures here (review finding
            #24); dynamic default mode avoids that.

    Returns:
        The compiled model, of whatever kind ``torch.compile`` hands back for
        the input (an ``OptimizedModule`` for a module, a callable wrapper for
        a function) -- hence ``Any``, which is the only annotation that is not
        wrong for one of the two call sites. Compilation is lazy, so this
        returns immediately and any Dynamo/Inductor failure surfaces
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
    #
    # Containment (issue #23): this is a `torch._dynamo.config` attribute --
    # PROCESS-GLOBAL, not scoped to this model or even to this adapter class.
    # `create_model(..., compile_model=True)` is called inside a
    # `multiprocessing.get_context("spawn")` worker in production
    # (``Auto3D.orchestration.workflow_workers``), which is a fresh
    # interpreter that exits when the worker does, so the setting cannot
    # outlive that one optimization job. It is NOT contained the same way for
    # an in-process caller -- the Python API (``smiles2mols``, ``calc_spe``,
    # ``opt_geometry``, ``calc_thermo``, or any direct
    # ``create_model(..., compile_model=True)`` call) runs in the caller's own
    # interpreter, so this flip persists for the rest of that process and
    # silently changes how every OTHER ``torch.compile`` call in it handles a
    # failure, Auto3D's or not.
    torch._dynamo.config.suppress_errors = True
    return torch.compile(model, mode=mode, fullgraph=False, dynamic=True)


def _raise_for_energy(energy: torch.Tensor) -> None:
    """Raise the NaN/Inf diagnosis for a known-non-finite energy tensor.

    Split out so the energy-only path
    (:func:`validate_energies`, reached from
    :meth:`Auto3D.engines.batch_opt.model_wrapper.EnForce_ANI.energy_batched`) and the
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
      here falls back from one to the other. A subclass that defines neither is
      refused at class definition (see :meth:`__init_subclass__`).

    Note on torch.inference_mode():
        This class CANNOT use torch.inference_mode() or torch.no_grad() in forward
        methods because force calculations require computing gradients of energy
        with respect to atomic coordinates via torch.autograd.grad(). Model parameters
        have requires_grad=False (frozen weights), but coordinates must have
        requires_grad=True for force computation. All autograd.grad() calls use
        create_graph=False to avoid building second-order gradient graphs.
    """

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

        ``_try_compile`` sets ``torch._dynamo.config.suppress_errors = True``
        so a graph break Inductor cannot handle degrades to eager rather than
        crashing an optimization thousands of steps in -- but a silent
        degrade is itself a defect (issue #23): nothing told a caller that
        ``compile_model=True`` bought nothing for an entire run.

        ``torch._dynamo.utils.counters["frames"]`` is Dynamo's own
        bookkeeping: ``"total"`` increments for every frame it attempts,
        ``"ok"`` only for one that completes without raising back up to the
        `except` clause that does the actual suppressing (a normal partial
        graph break -- a resume point Dynamo places itself -- still counts as
        ``"ok"``; only a fully suppressed failure does not). ``total > ok`` is
        therefore exactly "a suppressed error fired at least once", not
        merely "there was a graph break" -- and both counts are read as a
        DELTA against the snapshot ``__init__`` took right after compiling
        (``_compile_frame_stats_before``), because the counter itself is
        process-global: without the delta, one adapter's stale fallback would
        get blamed on the next compiled adapter's unrelated first forward.

        Called on every forward for each compilable adapter (``ANI2xtAdapter``,
        ``ANI2xAdapter``, ``CustomModelAdapter`` -- inherited from
        :meth:`BaseModelAdapter.forward` since each backend collapsed to
        overriding only ``_energy_graph``; ``AIMNet2Adapter`` never reaches
        ``_try_compile`` at all, since it keeps its own ``forward`` and its
        ``compile_model`` goes to ``AIMNet2Calculator`` instead), and it logs
        once per *increase* in the suppressed count (``_compile_suppressed_seen``).

        This used to be gated to a single check, on the premise that whether a
        compiled frame falls back "cannot change after the first observation".
        That premise is false, and measurably so (T-2): Dynamo 0/1-specializes
        the seven per-element index tensors ``ANI2xt`` builds, so each distinct
        element-presence pattern is a fresh recompile, and once
        ``torch._dynamo.config.recompile_limit`` is reached the frame runs
        eager for the rest of the process (measured on CPU: total 9, ok 8).
        The one allowed observation, meanwhile, was spent on the
        compiled-vs-eager probe forward ``create_model`` runs at construction
        -- a frame that by construction compiles cleanly -- so the real
        fallback, hundreds of steps later, was never reported at all. The
        factory re-baselines ``_compile_frame_stats_before`` and this count
        after that probe, so the probe's own frame is excluded from these
        deltas.

        The per-forward cost is two dict lookups and a handful of integer
        subtractions -- no device traffic and no host-device sync -- which is
        what makes reading it every step affordable in the hottest loop in the
        codebase.

        ``getattr`` with a default, not direct attribute access: several
        tests construct an adapter by bypassing ``BaseModelAdapter.__init__``
        entirely and substituting a toy module (the same reason
        ``ANI2xtAdapter._call_model`` reads ``self._element_indices`` through
        ``getattr`` rather than directly), so neither ``_compiled`` nor
        ``_compile_frame_stats_before`` may exist. An adapter with no
        ``_compiled`` was never compiled, so "nothing to check" is the right
        answer, not an ``AttributeError``; one with ``_compiled`` set by hand
        but no snapshot (a test asserting this method's behavior directly)
        falls back to treating the counter's absolute value as the delta,
        which is exactly right when nothing else in the process has compiled
        yet.
        """
        if not getattr(self, "_compiled", False):
            return
        before: dict[str, int] = getattr(self, "_compile_frame_stats_before", {})
        after: dict[str, int] = torch._dynamo.utils.counters.get("frames", {})
        total = after.get("total", 0) - before.get("total", 0)
        suppressed = total - (after.get("ok", 0) - before.get("ok", 0))
        if suppressed > getattr(self, "_compile_suppressed_seen", 0):
            self._compile_suppressed_seen = suppressed
            logger.warning(
                "torch.compile suppressed %d of %d frame compilation(s) for this "
                "adapter and fell back to eager execution for them "
                "(suppress_errors=True, see _try_compile); compile_model=True "
                "may not be providing its intended benefit.",
                suppressed,
                total,
            )

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
    ) -> torch.Tensor:
        """The backend's energies, graph-connected to ``coords``, at the dtype it produces.

        One-directional hook: this base body never calls :meth:`forward`, so
        there is no path back into the template from here. Every in-tree
        adapter but AIMNet2 overrides this with its one backend call;
        :meth:`forward` and :meth:`energy` are built on top of it.
        :class:`AIMNet2Adapter`, whose calculator computes forces itself,
        overrides :meth:`forward` wholesale instead and supplies its own
        one-line ``_energy_graph`` returning ``forward``'s first output --
        the same shape its ``energy`` used to be, just stated here instead of
        as a conditional default.

        The rule for subclasses, stated once here: override this hook, or
        override :meth:`forward` wholesale (and also this hook if ``energy``
        is needed). Neither override may call the other through the template:
        a ``forward`` override must not call ``super().forward()``, and a
        ``_energy_graph`` override must not call ``self.forward()`` unless
        ``forward`` is also overridden, because the base ``forward`` calls this
        hook and the cycle recurses with no diagnosis.
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
        """
        return self._energy_graph(coords, species, charges, atom_mask)

    def forward(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
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
                index is their own dummy-atom sentinel (audit C13).

        Returns:
            Tuple of (energies, forces); energies (batch,), forces
            (batch, n_atoms, 3), both in eV units.
        """
        input_dtype = coords.dtype
        coords_in, charges_in = self._model_inputs(coords, charges)
        coords_in = coords_in.requires_grad_(True)
        energy = self._energy_graph(coords_in, species, charges_in, atom_mask)
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

    Defines no ``energy`` override: ``forward`` returns float64 energies
    whatever it was fed -- an UPCAST, so there is no silent precision loss to
    guard against. The hazard the other backends' ``_model_inputs`` float32
    casts would create for ``energy`` -- an fp64 request answered in fp32
    with no error -- does not arise for AIMNet2, whose energy is an fp64
    upcast, and whole-graph fp64 through AIMNet2 would be false precision
    regardless. Its own ``_energy_graph`` returns ``forward``'s first output
    (hence the calculator's ``forces=True`` path), which is the route the
    calculator guarantees stays connected to ``coord`` in the autograd graph.
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

        Raises:
            DependencyError: The ``aimnet`` package is not installed.
        """
        require_aimnet()
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
        publish purely so ``Auto3D.entry.ASE.thermo._load_hessian_model`` could hand
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

    def to_double(self) -> None:
        """Refuse: AIMNet2 has no meaningful whole-graph fp64 form.

        Two independent reasons, either sufficient. Whole-graph fp64 through
        AIMNet2 is false precision -- the network is trained and evaluated in
        fp32 -- and this adapter never needs the upcast anyway, because
        :meth:`analytic_hessian` above means it is never differentiated by
        autograd. ``Auto3D.entry.ASE.thermo._load_hessian_model`` accordingly upcasts
        the ANI and custom-model branches and not this one.

        The mechanism would also be wrong, not merely unnecessary: ``self.model``
        is ``self._calc.model``, the same object, so upcasting it here mutates the
        module underneath an ``AIMNet2Calculator`` that prepares its own fp32
        input tensors.

        Inheriting :meth:`BaseModelAdapter.to_double` would make all of that a
        silent, working-looking call. This raise is the same judgment
        :meth:`analytic_hessian` documents for ``None``: a member that cannot
        honestly do what it says must say so rather than appear to comply.
        """
        raise NotImplementedError(
            "AIMNet2 has no fp64 form: it is trained and evaluated in float32, "
            "and its Hessian is analytic, so it is never differentiated by "
            "autograd and never needs the upcast. Use analytic_hessian instead."
        )

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
                returned by :func:`Auto3D.engines.batch_opt.padding.pad_from_mols`.
                Required for a padded batch, and now ENFORCED rather than merely
                documented -- see ``Raises`` below. ``None`` means every slot is
                a real atom, which is what the unpadded single-molecule callers
                (``ASE/thermo.py``'s ASE Calculator, ``auto3d models test``)
                want.

        Returns:
            (energy[batch], forces[batch, n_atoms, 3]) in eV and eV/A. Padded
            atom slots have zero force.

        Raises:
            ValueError: ``atom_mask`` is ``None`` on a batch of two or more
                molecules that contains a slot equal to ``species_pad`` (0).
                Such a slot is either padding or a dummy atom and this adapter
                cannot tell which, so both readings are refused instead of one
                being guessed: treating it as padding deletes a real atom
                (audit C13), and treating it as an atom feeds AIMNet2 a ghost
                at the origin, which returns NaN. A caller with a padded batch
                has the mask -- the padder returned it alongside the other three
                tensors. This is the one place the sentinel VALUE is read, and it
                is read only to refuse; the mask that drives the arithmetic below
                is still never derived from it.

                "Two or more molecules" is a deliberate concession, not an
                oversight, and dropping the ``> 1`` clause would break working
                callers. A SINGLE molecule may legitimately contain a species-0
                dummy ``*`` atom -- that is the whole R-group case audit C13 is
                about -- and whether such a molecule is scored at all is the
                engine policy's decision (``models.policy._requires_aimnet``
                routes it here on purpose), not this method's. Four production
                paths hand exactly that over unmasked at B == 1
                (``ASE/thermo/calculator.py``, ``ASE/thermo/driver.py``,
                ``ASE/thermo/vibrations.py``, ``cli/commands/models.py``), and
                their coverage is slow-marked, so the fast tier would not report
                the breakage.

                The check is also value-based, so it is narrower than "a padded
                batch without a mask is refused" and must not be read as that.
                It cannot see a hand-built batch padded with any value other than
                ``species_pad``, nor a padded batch exactly one molecule wide
                (which ``pad_from_mols`` cannot produce -- it pads to the widest
                molecule -- but ``PaddedBatch.sub(slice(i, i + 1))`` with the mask
                dropped by hand can). Nothing stronger is available from the
                tensors alone without putting a sentinel comparison back into the
                arithmetic, which is what audit C13 forbids.

        The real-atom mask is the caller's explicit ``atom_mask``, NEVER
        ``species != self.species_pad``. This adapter's ``species_pad`` is 0
        and it consumes raw atomic numbers, so the sentinel comparison deleted
        atomic number 0 -- an R-group/dummy ``*`` atom, which
        ``models.policy._requires_aimnet`` routes to precisely this engine.
        For ``*CCO`` the padder reported 9 real atoms and this adapter scored
        8: the energy belonged to a different species, and the dummy atom got
        exactly zero force and stayed frozen for the whole optimization. That
        is the collision class ``padding.pad_from_mols`` documents (audit C13).
        """
        if atom_mask is None and species.shape[0] > 1 and bool((species == self.species_pad).any()):
            raise ValueError(
                "AIMNet2Adapter needs atom_mask for a padded batch: a slot equal "
                "to species_pad (0) is either padding or a dummy atom, and this "
                "adapter cannot tell which without the mask pad_from_mols "
                "returns. An unpadded batch may omit it."
            )
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

    def _energy_graph(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``forward``'s first output: the only way to reach AIMNet2's energy.

        This does not recurse: ``forward`` above is self-contained (it never
        calls ``_energy_graph``), so this one-line hook just states, on the
        subclass that needs it, the fallback the base class's default used to
        apply conditionally. The energy it returns is an fp64 upcast (see the
        class docstring), not a dtype-preserving pass-through.
        """
        return self.forward(coords, species, charges, atom_mask)[0]


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


class CustomModelAdapter(BaseModelAdapter):
    """Adapter for user-provided custom NNP models.

    Custom models implement the contract defined in
    ``Auto3D.engines.models.contract`` (:class:`~Auto3D.engines.models.contract.CustomNNP`):
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
        # (shared load contract -- see Auto3D.engines.models.loading.load_custom_nnp).
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

    def _model_inputs(self, coords, charges):
        # Documented downcast for float32 custom models; an fp64 model upcasts
        # internally (class docstring). energy() does not pass through here.
        return coords.float(), charges.float()

    def _energy_graph(self, coords, species, charges, atom_mask=None):
        """The published contract is ``forward(species, coords, charges)`` -- species FIRST;
        ``charges`` follows ``coords``' dtype so a model that concatenates them does not mismatch."""
        return self.model(species, coords, charges.to(coords.dtype))
