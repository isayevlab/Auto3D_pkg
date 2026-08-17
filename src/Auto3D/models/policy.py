"""What a model requires of its host and of its molecules.

Two preconditions that belong to the model layer rather than to generic
helpers, and that every caller of a model has to satisfy: is the device the
caller asked for actually there, and can the engine they chose represent the
molecules they handed it. Both were in ``utils/validation.py`` alongside input
and configuration parsing, which made ``utils`` -- a leaf by intent -- depend on
the model package, and made a property calculator import a module named for
validating command-line input in order to ask whether a GPU exists.
"""

from __future__ import annotations

import torch
from rdkit import Chem

from Auto3D.constants import BUILTIN_ANI_MODELS
from Auto3D.exceptions import ConfigurationError, GPUError
from Auto3D.models.species import ANI2XT_INDEX

#: Elements ANI2x/ANI2xt were trained on. AIMNET (and any aimnet registry
#: model) and a custom NNP path are not restricted to this set.
ANI_ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})

# The gate above and ANI2xt's network-index table must cover the same elements.
# Asserted rather than derived, deliberately: they are the same seven numbers but
# not the same fact -- this set is what ANI2x AND ANI2xt were trained on, while
# ANI2XT_INDEX is one engine's 0-based index order. Defining either in terms of
# the other would record a provenance that is not true and would stop being a
# check. Same construction as model_factory's BUILTIN_ANI_MODELS assert.
assert frozenset(ANI2XT_INDEX) == ANI_ELEMENTS, (
    "ANI_ELEMENTS (what check_engine_supports_molecules admits) and "
    "ANI2XT_INDEX (what to_ani2xt_species can remap) have drifted apart: "
    f"{sorted(ANI_ELEMENTS ^ frozenset(ANI2XT_INDEX))}"
)


def check_gpu_requested(use_gpu: bool) -> None:
    """Raise if GPU was requested but no CUDA device is visible.

    Single source of truth for Auto3D's GPU policy: **fatal, not a silent
    fallback**. Before this function existed, ``use_gpu=True`` on a CPU-only
    box produced three different behaviors depending on the entry point
    (M23):

    - ``main()`` (via ``WorkflowOrchestrator._validate_input`` ->
      ``check_valid_configuration``) raised ``ConfigurationError``, which
      shows the CLI's "run 'auto3d config init'" hint -- unrelated to a GPU
      problem.
    - ``smiles2mols`` reached ``check_input``'s own inline check and raised
      ``GPUError`` (the right exception, right hint), but with different
      wording than ``check_valid_configuration``'s message.
    - ``auto3d energy``/``optimize``/``thermo`` (``calc_spe``/
      ``opt_geometry``/``calc_thermo``) never checked at all: they fell back
      to CPU through ``model_factory.get_device`` with no error and no
      warning.

    A scripted user who set ``use_gpu=True`` and silently got CPU has no way
    to know their "GPU" results were actually computed on CPU -- possibly
    orders of magnitude slower than they assumed, with no signal anything
    was wrong. This function is called as the *first* check everywhere GPU
    use is decided (``check_input``, ``check_valid_configuration``, and the
    ``auto3d energy``/``optimize``/``thermo`` CLI commands in
    ``cli/commands/properties.py``, which call the API functions directly
    and never go through ``check_input``/``check_valid_configuration``), so
    it fails fast -- before any worker is forked and before any compute is
    spent -- with the same exception type and the same "--no-gpu" hint
    regardless of entry point.

    Args:
        use_gpu: The ``use_gpu`` option requested by the caller.

    Raises:
        GPUError: `use_gpu` is True and `torch.cuda.is_available()` is False.
    """
    if use_gpu and not torch.cuda.is_available():
        raise GPUError(
            "No cuda device was detected, but use_gpu=True was requested. "
            "Pass --no-gpu on the CLI (or set use_gpu=False in the Python "
            "API) to run on CPU."
        )


def _requires_aimnet(mol: Chem.Mol) -> bool:
    """True if `mol` cannot be represented by ANI2x/ANI2xt.

    A molecule needs AIMNET when it carries an element outside ANI_ELEMENTS or
    a nonzero net formal charge. Single implementation of this test --
    check_smi_format and check_sdf_format used to each inline it as their own
    copy of the identical {1, 6, 7, 8, 9, 16, 17} literal, which is exactly
    how the two would silently drift apart (C11).
    """
    elements = {a.GetAtomicNum() for a in mol.GetAtoms()}
    charge = Chem.rdmolops.GetFormalCharge(mol)
    return (not elements.issubset(ANI_ELEMENTS)) or charge != 0


def check_engine_supports_molecules(
    mols: Chem.Mol | list[Chem.Mol], optimizing_engine: str
) -> None:
    """Raise if `optimizing_engine` cannot represent every molecule in `mols`.

    ANI2x/ANI2xt can only represent uncharged molecules built from
    {H, C, N, O, F, S, Cl}. A charged or out-of-set species handed to either
    is silently evaluated as a different, neutral, in-set species -- tens of
    kcal/mol wrong energy and wrong forces, so a downstream "optimized"
    geometry is wrong too (C11).

    `check_input` already runs this check (via check_smi_format /
    check_sdf_format, which call `_requires_aimnet` above) for main() and
    smiles2mols. calc_spe, opt_geometry and calc_thermo take an SDF path
    directly and never go through check_input, so they call this function
    themselves instead.

    AIMNET (and any aimnet registry name) and a path to a custom NNP are not
    restricted by this element set, so this is a no-op for them.

    Args:
        mols: A single RDKit Mol or an iterable of them, read from the
            caller's input SDF.
        optimizing_engine: The engine name exactly as passed to
            calc_spe/opt_geometry/calc_thermo (e.g. 'ANI2x', 'AIMNET', a
            registry name, or a custom NNP path).

    Raises:
        ConfigurationError: `optimizing_engine` is ANI2x/ANI2xt (matched
            case-insensitively, mirroring ModelFactory.create) and at least
            one molecule is charged or contains an element outside the ANI
            training set.
    """
    if optimizing_engine.upper() not in BUILTIN_ANI_MODELS:
        return
    mol_list = [mols] if isinstance(mols, Chem.Mol) else list(mols)
    incompatible = [
        mol.GetProp("_Name") if mol.HasProp("_Name") else "<unnamed>"
        for mol in mol_list
        if _requires_aimnet(mol)
    ]
    if incompatible:
        raise ConfigurationError(
            f"Only AIMNET can handle: {incompatible}, but {optimizing_engine} was parsed to Auto3D."
        )
