# src/Auto3D/models/contract.py
"""Both model contracts, in one file, next to what enforces each of them.

Auto3D has two distinct and *permanently* separate model interfaces. Conflating
them is the whole reason this module exists, and the reason both declarations
live here rather than one file apart: they are forty lines from each other and
cannot be read separately.

* **Contract A -- the custom-NNP contract** (:class:`CustomNNP`). What a *user*
  implements and hands to Auto3D as ``optimizing_engine=/path/to/model.pt``.
  Auto3D calls it as ``model(species, coords, charges) -> energies`` and
  differentiates the returned energy with respect to ``coords`` to obtain forces
  (:meth:`Auto3D.models.adapter.CustomModelAdapter.forward`). The model returns
  energies only; it must not return forces. Enforced by
  :func:`validate_custom_nnp` at load time.

* **Contract B -- the adapter interface** (:class:`ModelAdapter`). Internal to
  Auto3D: ``forward(coords, species, charges, atom_mask=None) -> (energies,
  forces)``. Only Auto3D's own adapters implement it (via
  :class:`Auto3D.models.adapter.BaseModelAdapter`, which supplies working
  defaults for everything but ``forward``). Users never do. Enforced by
  :func:`missing_adapter_members`, which
  :class:`Auto3D.batch_opt.model_wrapper.EnForce_ANI` consults on construction.

Note that the two take ``species`` and ``coords`` in *opposite* order, and that
this is deliberate rather than an accident awaiting cleanup: Contract A's order
is published (``CHANGELOG.md``, ``docs/source/howto/custom_nnp.rst``) and
changing it would break every working third-party model. A model written against
the adapter interface silently computes an energy from transposed tensors and
then fails deep inside ``torch.autograd.grad``, so :func:`validate_custom_nnp`
rejects that shape at load time instead.
"""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from Auto3D.exceptions import ModelLoadError

#: Human-readable form of the contract, quoted in every rejection message.
EXPECTED_SIGNATURE = "forward(self, species, coords, charges) -> energies"


def _protocol_data_members(proto: type) -> tuple[str, ...]:
    """Names annotated in a Protocol's own class body, in declaration order.

    Data members land in ``__annotations__``; methods do not. Implemented by
    hand because the stdlib alternatives are unavailable or unstable here:
    ``typing.get_protocol_members`` is 3.13+, and ``__protocol_attrs__`` is a
    CPython implementation detail that also folds in the methods.

    ``from __future__ import annotations`` turns the annotation *values* into
    strings; the *keys* are unaffected, and only keys are used here.
    """
    return tuple(proto.__annotations__)


def _protocol_members(proto: type) -> tuple[str, ...]:
    """Every member a Protocol declares: data members then public methods."""
    data = _protocol_data_members(proto)
    methods = tuple(
        name
        for name, value in vars(proto).items()
        if not name.startswith("_") and callable(value) and name not in data
    )
    return data + methods


#: Attributes every custom NNP must define. They are the padding *fill* values
#: Auto3D writes into the batched tensors; ``species_pad`` in particular decides
#: what lands in the species tensor's unused slots, so guessing it is unsafe.
#:
#: DERIVED from :class:`CustomNNP`'s own annotations rather than retyped beside
#: them, so the two cannot drift. See the warning above ``CustomNNP``: this makes
#: the Protocol's annotations load-bearing.
#: (Assigned below the class, which must exist first.)
REQUIRED_ATTRIBUTES: tuple[str, ...]

# Parameter-name vocabulary used only to detect a *transposed* forward. Names
# outside these sets are not an error -- they just make the argument order
# unknowable, so the order check is skipped rather than guessed.
_SPECIES_NAMES = frozenset(
    {"species", "numbers", "atomic_numbers", "atomicnumbers", "z", "elements"}
)
_COORDS_NAMES = frozenset(
    {"coords", "coord", "coordinates", "positions", "pos", "xyz"}
)
_CHARGES_NAMES = frozenset({"charges", "charge", "q"})


# WARNING -- the annotated members of this class are load-bearing, not
# documentation. REQUIRED_ATTRIBUTES is derived from them (below), and
# validate_custom_nnp skips the `forward` signature check entirely for a
# TorchScript RecursiveScriptModule, so for every archive in the wild those
# annotations are the ONLY gate. Adding a third annotated field here instantly
# rejects every existing custom NNP that does not carry it: that is a BREAKING
# change and must be released as one. `test_custom_nnp_contract.py::
# test_customnnp_data_members_are_exactly_the_two_padding_values` pins the set
# so it cannot happen by accident.
#
# Deliberately NOT @runtime_checkable, unlike ModelAdapter below. A runtime
# Protocol check tests attribute *presence* only, and torch installs
# `Module.forward = _forward_unimplemented` on every nn.Module, so the single
# most common real failure -- a saved module with no forward of its own -- would
# pass isinstance() and then raise NotImplementedError inside the optimization
# loop. `_check_forward_signature` already handles that case correctly and by
# hand, and a bare boolean cannot carry the diagnosis validate_custom_nnp emits.
# So `isinstance(x, CustomNNP)` raises TypeError, which is the honest answer to
# "can I check this at runtime?" -- no; call validate_custom_nnp.
class CustomNNP(Protocol):
    """Protocol a user-supplied NNP must satisfy to be used as an engine.

    Attributes:
        coord_pad: Fill value written into unused coordinate slots (typically 0).
        species_pad: Fill value written into unused species slots (typically -1).
            Prefer a value that cannot collide with a real species index of your
            model -- ``-1`` is safe for both atomic numbers and 0-based indices.

    Example:
        >>> class MyNNP(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.coord_pad = 0.0
        ...         self.species_pad = -1
        ...
        ...     def forward(self, species, coords, charges):
        ...         # -> energies, shape (batch,), in eV. Forces are obtained by
        ...         # Auto3D via autograd; do not return them.
        ...         return self.energy(species, coords, charges)

    Note:
        Set ``coord_pad``/``species_pad`` in ``__init__`` (or list them in
        ``__constants__``) rather than as bare class attributes if you save the
        model with ``torch.jit.save``: TorchScript does not carry plain class
        attributes into the archive, so they would be absent after loading.
    """

    coord_pad: float
    """Fill value for unused coordinate slots in batched tensors."""

    species_pad: int
    """Fill value for unused species slots in batched tensors."""

    def forward(
        self,
        species: torch.Tensor,
        coords: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energies for a padded batch of molecules.

        Args:
            species: Atomic numbers (or the model's own species indices),
                shape (batch, max_atoms). Unused slots hold ``species_pad``.
            coords: Atomic coordinates in Angstrom, shape (batch, max_atoms, 3).
                Unused slots hold ``coord_pad``.
            charges: Total molecular charge per molecule, shape (batch,).

        Returns:
            Energies, shape (batch,), in eV. Auto3D differentiates this with
            respect to ``coords`` to obtain forces, so the returned tensor must
            stay connected to ``coords`` in the autograd graph.
        """
        ...


REQUIRED_ATTRIBUTES = _protocol_data_members(CustomNNP)


@runtime_checkable
class ModelAdapter(Protocol):
    """Contract B: the interface Auto3D's own model adapters present.

    Every consumer inside Auto3D -- the optimizer, the single-point-energy path,
    the ASE calculator, the CLI health check -- talks to a model through exactly
    these members, and this is the type they annotate. Implementations live in
    :mod:`Auto3D.models.adapter`; :class:`~Auto3D.models.adapter.BaseModelAdapter`
    supplies working defaults for everything except ``forward``, so an in-tree
    adapter satisfies this by inheritance.

    Note the argument order is the REVERSE of :class:`CustomNNP`
    (``species`` first there, ``coords`` first here) and that this one returns
    ``(energies, forces)`` while a custom NNP returns energies only. See the
    module docstring.

    ``device`` is deliberately NOT part of this contract. Nothing outside an
    adapter reads it (``BaseModelAdapter`` keeps ``self.device`` as an
    implementation detail), and requiring it would make every legitimate
    structural implementation -- including test doubles that never touch a
    device -- non-conforming for no benefit.

    This Protocol IS ``@runtime_checkable``, and unlike before it is actually
    consulted: :func:`missing_adapter_members` is called by
    :class:`Auto3D.batch_opt.model_wrapper.EnForce_ANI`. Be clear about what that
    buys, because overselling it is how the decorator became decorative in the
    first place: a presence check catches a **category error** (a raw
    ``nn.Module``, a third-party calculator, an engine-name string), NOT a
    contract violation. Presence is not arity, and a ``MagicMock`` satisfies it
    trivially. Never use ``issubclass`` against this Protocol -- it raises
    ``TypeError`` for any Protocol with data members.
    """

    coord_pad: float
    """Fill value the batch padder writes into unused coordinate slots."""

    species_pad: int
    """Fill value the batch padder writes into unused species slots."""

    def to_species(self, atomic_numbers: Sequence[int]) -> list[int]:
        """Convert atomic numbers into this model's own species convention.

        The species convention is a property of the *model*, so it lives on the
        same object that supplies ``species_pad``. That is what makes it
        impossible for the remap and the padding sentinel to come from two
        different sources and contradict each other -- the shape of audit
        findings C3/C4, where a name-keyed converter and an adapter-supplied pad
        disagreed about which slots were padding.

        Args:
            atomic_numbers: Atomic numbers, one per atom.

        Returns:
            Species values in the model's convention. The identity for every
            engine except ANI2xt, which uses 0-based network indices.
        """
        ...

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
            species: Species in this adapter's own convention (batch, n_atoms),
                i.e. the output of :meth:`to_species`.
            charges: Molecular charges (batch,).
            atom_mask: Boolean (batch, n_atoms), True for real atoms and False
                for padded slots, as returned by
                :func:`Auto3D.batch_opt.padding.pad_from_mols`. Required from
                any caller that passes a PADDED batch; ``None`` means every
                slot holds a real atom. An adapter must never re-derive this by
                comparing ``species`` against ``species_pad`` (audit C13).

        Returns:
            Tuple of (energies, forces) where energies has shape (batch,)
            and forces has shape (batch, n_atoms, 3). Units: eV.
        """
        ...

    def energy(
        self,
        coords: torch.Tensor,
        species: torch.Tensor,
        charges: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute energies only, graph-connected and at the caller's dtype.

        Two properties are part of the contract and neither is optional:

        * **No internal ``no_grad``/``inference_mode``.** The result must stay
          connected to ``coords`` so a caller can differentiate it (a Hessian).
          A caller that wants no graph wraps its own call site.
        * **Dtype-preserving.** ``forward`` downcasts to float32 in two adapters
          for compatibility with float32 NNP weights. ``energy`` must not: an
          fp64 caller that silently receives an fp32 result gets no error and no
          warning, only a wrong number.

        Args:
            coords: Atomic coordinates (batch, n_atoms, 3).
            species: Species in this adapter's own convention (batch, n_atoms).
            charges: Molecular charges (batch,).
            atom_mask: As for :meth:`forward`.

        Returns:
            Energies, shape (batch,), in eV.
        """
        ...


def missing_adapter_members(obj: Any) -> list[str]:
    """Members :class:`ModelAdapter` requires that ``obj`` does not provide.

    Derived from the Protocol, never hand-listed, so widening
    :class:`ModelAdapter` widens the rejection message in the same edit.

    Args:
        obj: The candidate adapter.

    Returns:
        Missing member names in declaration order; empty if ``obj`` structurally
        conforms. Remember this is a presence check: an empty list means "not a
        category error", not "correct".
    """
    return [name for name in _protocol_members(ModelAdapter) if not hasattr(obj, name)]


def _classify(name: str) -> str | None:
    """Map a parameter name to ``'species'``/``'coords'``/``'charges'``, or None."""
    lowered = name.lower()
    if lowered in _SPECIES_NAMES:
        return "species"
    if lowered in _COORDS_NAMES:
        return "coords"
    if lowered in _CHARGES_NAMES:
        return "charges"
    return None


def _check_forward_signature(model: Any, source: str) -> None:
    """Reject a ``forward`` that cannot be called as ``(species, coords, charges)``.

    Skips silently when the signature is not introspectable or the parameter
    names carry no ordering information -- a false rejection would break a
    working model, which is worse than a missed diagnosis.
    """
    # `getattr(model, "forward", None) is None` never fires for an nn.Module:
    # torch supplies `Module.forward = _forward_unimplemented(*input)`, which is
    # a real callable, so the attribute always exists. Worse, its signature is
    # VAR_POSITIONAL, so the `*args` early-return below would then ACCEPT it --
    # a saved module with valid padding attributes and no forward of its own
    # would load clean here and raise NotImplementedError deep inside the
    # optimization loop, which is exactly the failure this validator exists to
    # move to load time. Compare the class attribute against Module's own to
    # tell "inherited the stub" from "defined a real forward".
    # ScriptModule is excluded and the access is guarded: a
    # RecursiveScriptModule's class attribute is a `_CachedForward` descriptor
    # that raises AttributeError on this lookup, and a scripted module always
    # has a real forward anyway (it could not have been scripted otherwise).
    # Rejecting one here would break every valid TorchScript archive.
    if isinstance(model, torch.nn.Module) and not isinstance(
        model, torch.jit.ScriptModule
    ):
        try:
            inherits_stub = type(model).forward is torch.nn.Module.forward
        except AttributeError:
            inherits_stub = False
        if inherits_stub:
            raise ModelLoadError(
                f"Custom NNP at {source} defines no forward method of its own "
                f"(it inherits torch.nn.Module's placeholder, which raises "
                f"NotImplementedError when called). Expected {EXPECTED_SIGNATURE}."
            )

    forward = getattr(model, "forward", None)
    if forward is None:
        raise ModelLoadError(
            f"Custom NNP at {source} defines no forward method. "
            f"Expected {EXPECTED_SIGNATURE}."
        )

    try:
        signature = inspect.signature(forward)
        parameters = list(signature.parameters.values())
    except (ValueError, TypeError):
        # A TorchScript RecursiveScriptModule's forward is a pybind11 builtin
        # with no Python signature, so inspect.signature raises ValueError. Skip
        # the signature check for such models rather than reject a valid
        # TorchScript archive; the REQUIRED_ATTRIBUTES check still applies.
        return

    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        # forward(*args) accepts any arity and names nothing; nothing to check.
        return

    positional = [
        p
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    # Render the signature the way the interpreter does, NOT by comma-joining
    # parameter names. Joining names drops `*`, `/` and defaults, so a
    # keyword-only forward was refused with text identical to the text it was
    # being asked for ("has forward(species, coords, charges) ... Expected
    # forward(self, species, coords, charges)") -- a message that shows the
    # author nothing wrong. The marker IS the explanation.
    # The return annotation is dropped so the rendering stays comparable with
    # EXPECTED_SIGNATURE's parameter list; everything else -- markers, defaults,
    # parameter annotations -- is kept exactly as written.
    rendered = str(signature.replace(return_annotation=inspect.Signature.empty))
    if len(positional) < 3 or len(required) > 3:
        raise ModelLoadError(
            f"Custom NNP at {source} has forward{rendered}, which cannot be "
            f"called with three positional arguments. Expected "
            f"{EXPECTED_SIGNATURE}, returning energies of shape (batch,) in eV."
        )

    classified = [_classify(p.name) for p in positional[:3]]
    if None in classified:
        # Unrecognized parameter names: the order is unknowable, so accept.
        return
    if classified != ["species", "coords", "charges"]:
        raise ModelLoadError(
            f"Custom NNP at {source} has forward{rendered}, but Auto3D calls a "
            f"custom NNP as {EXPECTED_SIGNATURE} -- species first, coords "
            f"second. Note this is the opposite order from Auto3D's internal "
            f"ModelAdapter interface (coords, species, charges), which returns "
            f"(energies, forces); a custom NNP returns energies only, and Auto3D "
            f"derives forces from them by autograd. Reorder the parameters or "
            f"wrap the model."
        )


def validate_custom_nnp(model: Any, source: str) -> None:
    """Check a freshly loaded custom NNP against :class:`CustomNNP`.

    Args:
        model: The loaded module.
        source: Path the model came from, for error messages.

    Raises:
        ModelLoadError: If a required padding attribute is missing, or if
            ``forward`` demonstrably cannot be called as
            ``(species, coords, charges)``.
    """
    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(model, name)]
    if missing:
        raise ModelLoadError(
            f"Custom NNP at {source} is missing required attribute(s): "
            f"{', '.join(missing)}. Auto3D writes these values into the unused "
            f"slots of the batched coordinate and species tensors, so they "
            f"cannot be guessed -- a wrong species_pad silently changes which "
            f"atoms your model treats as padding. Define them in __init__ "
            f"(e.g. self.coord_pad = 0.0; self.species_pad = -1). If you save "
            f"with torch.jit.save, note that TorchScript drops plain class "
            f"attributes: set them on the instance or list them in "
            f"__constants__."
        )
    _check_forward_signature(model, source)
