# src/Auto3D/models/contract.py
"""The single definition of the custom-NNP contract, next to what enforces it.

Auto3D has two distinct model interfaces, and conflating them is the whole
reason this module exists:

* **The custom-NNP contract (this module).** What a *user* implements and hands
  to Auto3D as ``optimizing_engine=/path/to/model.pt``. Auto3D calls it as
  ``model(species, coords, charges) -> energies`` and differentiates the
  returned energy with respect to ``coords`` to obtain forces
  (:meth:`Auto3D.models.adapter.CustomModelAdapter.forward`). The model returns
  energies only; it must not return forces.

* **The adapter interface** (:class:`Auto3D.models.adapter.ModelAdapter`).
  Internal to Auto3D: ``forward(coords, species, charges) -> (energies,
  forces)``. Only Auto3D's own adapters implement it. Users never do.

Note that the two take ``species`` and ``coords`` in *opposite* order. A model
written against the adapter interface silently computes an energy from
transposed tensors and then fails deep inside ``torch.autograd.grad``, so
:func:`validate_custom_nnp` rejects that shape at load time instead.
"""
from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

import torch

from Auto3D.exceptions import ModelLoadError

#: Human-readable form of the contract, quoted in every rejection message.
EXPECTED_SIGNATURE = "forward(self, species, coords, charges) -> energies"

#: Attributes every custom NNP must define. They are the padding *fill* values
#: Auto3D writes into the batched tensors; ``species_pad`` in particular decides
#: what lands in the species tensor's unused slots, so guessing it is unsafe.
REQUIRED_ATTRIBUTES = ("coord_pad", "species_pad")

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


@runtime_checkable
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
        parameters = list(inspect.signature(forward).parameters.values())
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
    if len(positional) < 3 or len(required) > 3:
        rendered = ", ".join(p.name for p in parameters) or "<no parameters>"
        raise ModelLoadError(
            f"Custom NNP at {source} has forward({rendered}), which cannot be "
            f"called with three positional arguments. Expected "
            f"{EXPECTED_SIGNATURE}, returning energies of shape (batch,) in eV."
        )

    classified = [_classify(p.name) for p in positional[:3]]
    if None in classified:
        # Unrecognized parameter names: the order is unknowable, so accept.
        return
    if classified != ["species", "coords", "charges"]:
        rendered = ", ".join(p.name for p in positional[:3])
        raise ModelLoadError(
            f"Custom NNP at {source} has forward({rendered}), but Auto3D calls a "
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
