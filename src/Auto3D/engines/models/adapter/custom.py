"""The user-supplied custom-NNP adapter: ``CustomNNP`` wrapped as a ``ModelAdapter``.

Cut from this package's single-file predecessor.
"""

from __future__ import annotations

import warnings

import torch

from Auto3D.engines.models.adapter.base import BaseModelAdapter
from Auto3D.engines.models.loading import load_custom_nnp
from Auto3D.foundation.exceptions import ModelLoadError


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
        self._model_path = model_path
        model = load_custom_nnp(model_path, device)
        # TorchScript archives are already a compiled graph, so `torch.compile`
        # has nothing to add; an eager `nn.Module` -- which `load_custom_nnp`
        # also accepts, and which every AIMNet2-derived custom model is -- does
        # benefit. Honoring the flag only in that case is what stops
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

    def _energy_graph(self, coords, species, charges, atom_mask=None, mults=None):
        """The published contract is ``forward(species, coords, charges)`` -- species FIRST;
        ``charges`` follows ``coords``' dtype so a model that concatenates them does not mismatch.

        ``atom_mask`` and ``mults`` are unused: the published custom-NNP
        contract has three inputs, so a custom model is never handed a spin
        multiplicity and evaluates every molecule the way it always did.

        The returned shape is checked here, on every call, rather than with a
        synthetic forward at load time: executing the user's model has
        side effects this adapter cannot bound -- a real custom NNP's
        calculator can lazily build persistent per-instance state (adaptive
        neighbor lists, D3/Coulomb buffers) sized off whatever it is first
        called with, so a degenerate load-time probe geometry would leave
        that state mis-sized for every real molecule that followed. Checking here costs
        one shape comparison per call and catches the same defect a probe
        was meant to -- a bare scalar, say, which would otherwise sum over the
        batch and silently move every molecule by one shared, wrong force --
        before autograd ever sees it, and before the model has been asked to
        answer for any geometry but the caller's own.
        """
        energies = self.model(species, coords, charges.to(coords.dtype))
        batch = coords.shape[0]
        if tuple(energies.shape) != (batch,):
            raise ModelLoadError(
                f"Custom NNP at {self._model_path} must return one energy per "
                f"molecule, shape ({batch},); got {tuple(energies.shape)}."
            )
        return energies
