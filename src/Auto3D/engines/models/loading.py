"""Shared loader for user-provided custom NNP model files.

A single definition of the "TorchScript archive OR eager nn.Module" load
contract, used by every entry point that accepts a custom model path
(model_factory/CustomModelAdapter, input validation, and the thermo Hessian
path) so the accepted formats and error behavior stay identical everywhere.
"""

from __future__ import annotations

import torch

from Auto3D.engines.models.contract import validate_custom_nnp
from Auto3D.foundation.exceptions import ModelLoadError


def load_custom_nnp(
    model_path: str,
    device: torch.device,
    *,
    double: bool = False,
) -> torch.nn.Module:
    """Load a custom NNP from a file as a TorchScript archive or eager nn.Module.

    Tries ``torch.jit.load`` first (legacy TorchScript archives); falls back to
    ``torch.load`` for eager ``nn.Module`` checkpoints saved with
    ``torch.save`` -- modern AIMNet2-based models are no longer
    ``torch.jit.script``-able. Always returns an ``nn.Module`` on ``device``;
    any failure (corrupt file, or a payload that is not an ``nn.Module`` such as
    a bare ``state_dict``) is raised as :class:`~Auto3D.foundation.exceptions.ModelLoadError`.

    ``weights_only=False`` is required to deserialize a whole ``nn.Module`` and
    executes code from the file; these are trusted, user-supplied local paths
    the caller explicitly selected as the optimizing engine.

    Whatever the format, the loaded module is checked against the custom-NNP
    contract (:func:`Auto3D.engines.models.contract.validate_custom_nnp`) before it is
    returned, so a model with the wrong ``forward`` argument order or missing
    ``coord_pad``/``species_pad`` is refused here with a message naming the
    expected signature -- rather than producing a nonsense energy and failing
    deep inside ``torch.autograd.grad`` many steps later. The signature check
    cannot see the return VALUE, so a smoke forward on a one-molecule,
    two-atom probe batch runs after it, refusing a model that returns
    anything other than one energy per molecule -- a bare scalar, say, which
    would otherwise sum over the batch and move every molecule by one shared,
    wrong force deep inside the optimization loop.

    Args:
        model_path: Path to the model file.
        device: Target device for the loaded model.
        double: If True, cast the returned module to float64 before the smoke
            forward, so the probe exercises the model at the dtype it will
            actually run at.

    Returns:
        The loaded model as an ``nn.Module`` on ``device``.

    Raises:
        ModelLoadError: If the file cannot be loaded as either supported form,
            if the loaded module violates the custom-NNP contract, or if its
            smoke forward does not return one energy per molecule.
    """
    try:
        model: torch.nn.Module = torch.jit.load(model_path, map_location=device)
    except RuntimeError:
        # Not a TorchScript archive -> try an eager nn.Module checkpoint.
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:  # corrupt / unpicklable file
            raise ModelLoadError(
                f"Custom NNP at {model_path} could not be loaded as a TorchScript "
                f"archive or an eager nn.Module: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(model, torch.nn.Module):
            raise ModelLoadError(
                f"Custom NNP at {model_path} did not deserialize to an nn.Module "
                f"(got {type(model).__name__})."
            )
        model = model.to(device)
    # Both branches, not just the eager one: `torch.jit.save` records the
    # module's `training` flag, so a *scripted* archive saved before its author
    # called .eval() keeps dropout and batchnorm live at inference. That makes
    # the energy differ between identical calls, which FIRE cannot converge
    # against -- the run spends its whole step budget and reports
    # Converged=False, indistinguishable from a genuinely floppy molecule.
    #
    # This does not rescue a *traced* archive. `torch.jit.trace` bakes the
    # training-time branch into the graph as a constant, so `.eval()` clears
    # the flag while the recorded dropout keeps firing -- `model.training` is
    # then False and the energy is still stochastic. Nothing here can detect
    # that; a model author must call `.eval()` before tracing.
    model = model.eval()
    validate_custom_nnp(model, model_path)
    if double:
        model = model.double()

    # Smoke-test the loaded forward: a model that satisfies the attribute and
    # arity checks above can still return the wrong SHAPE -- a bare scalar
    # summed over every molecule in the batch, say -- which the signature
    # check cannot see (it inspects parameter names, never the return value)
    # and which only breaks later, deep inside the FIRE loop's per-molecule
    # bookkeeping, far from the model that caused it. Two carbons so a model
    # whose per-element branch only covers one species still runs; a zero
    # charge and padding-free coordinates so nothing here depends on the
    # caller's own padding convention. The probe reads the dtype off the
    # model's own parameters -- already settled by the ``double`` cast above,
    # or by whatever dtype the checkpoint was saved at -- rather than
    # imposing float32 unconditionally, which would be a cast the caller
    # never asked for.
    try:
        probe_dtype = next(model.parameters()).dtype
    except StopIteration:
        probe_dtype = torch.float64 if double else torch.float32
    species_probe = torch.tensor([[6, 6]], dtype=torch.long, device=device)
    coords_probe = torch.zeros((1, 2, 3), dtype=probe_dtype, device=device)
    charges_probe = torch.zeros((1,), dtype=probe_dtype, device=device)
    with torch.no_grad():
        energies = model(species_probe, coords_probe, charges_probe)
    if tuple(energies.shape) != (1,):
        raise ModelLoadError(
            f"Custom NNP at {model_path} must return one energy per molecule, "
            f"shape (batch,); got {tuple(energies.shape)}."
        )
    return model
