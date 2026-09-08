"""What a model requires of its host and of its molecules.

Two preconditions that belong to the model layer rather than to generic
helpers, and that every caller of a model has to satisfy: is the device the
caller asked for actually there, and can the engine they chose represent the
molecules they handed it. Both were in ``utils/validation.py`` alongside input
and configuration parsing, which made ``utils`` -- a leaf by intent -- depend on
the model package, and made a property calculator import a module named for
validating command-line input in order to ask whether a GPU exists.

Layering: ``torch`` and ``rdkit`` are NOT imported at module scope here, even
though every function below eventually needs one or the other. This module
sits on the CLI's ``--help`` path (``Auto3D.presentation.cli.app`` ->
``commands/properties.py`` -> here), and both third-party packages are the
expensive ones: a bare ``import Auto3D.presentation.cli.app`` cost ~1.9s and
pulled in both before this was fixed (issue #14), for a command that may do
nothing more than print usage. Each function defers its own import, with a
comment saying why -- see ``check_gpu_requested`` and ``_requires_aimnet`` /
``check_engine_supports_molecules`` below. The ``Chem.Mol`` annotations still
type-check without a runtime rdkit import because ``from __future__ import
annotations`` (PEP 563) makes every annotation a string that is never
evaluated at runtime; the ``TYPE_CHECKING`` import below exists only so a type
checker can resolve the name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Auto3D.engines.models.species import ANI2XT_INDEX
from Auto3D.foundation.constants import BUILTIN_ANI_MODELS
from Auto3D.foundation.exceptions import ConfigurationError, GPUError
from Auto3D.foundation.utils.logging_config import get_logger

if TYPE_CHECKING:
    from rdkit import Chem

logger = get_logger(__name__)

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
    # Deferred: this is the ONLY torch user in the module (verified -- every
    # other function below needs rdkit, not torch), and it is reached on
    # every CLI invocation via commands/properties.py, `--help` included.
    # Importing torch at module scope charged that cost unconditionally,
    # regardless of whether use_gpu was even True (issue #14).
    #
    # Genuinely conditional on `use_gpu`, not merely relocated: this function
    # is called FIRST inside `check_input` for every run, `use_gpu=False`
    # included, and `import torch` -- even for an already-cached module --
    # still executes a real `import` statement, which
    # `tests/test_validation.py::test_omega_without_openeye_raises_dependency_error`
    # observes by patching `builtins.__import__` itself for the whole
    # `check_input` call to simulate openeye missing. An unconditional
    # `import torch` here raised that test's `ImportError` before the omega
    # check downstream ever ran, turning a use_gpu=False call -- which never
    # needs torch at all -- into a hard failure it wasn't asking for. Nesting
    # the import inside the branch that actually needs `torch.cuda` makes
    # `use_gpu=False` a true no-op again, matching the pre-issue-#14 module-
    # level-import behavior for that path exactly.
    if use_gpu:
        import torch

        if not torch.cuda.is_available():
            raise GPUError(
                "No cuda device was detected, but use_gpu=True was requested. "
                "Pass --no-gpu on the CLI (or set use_gpu=False in the Python "
                "API) to run on CPU."
            )


def _requires_aimnet(mol: Chem.Mol) -> bool:
    """True if `mol` cannot be represented by ANI2x/ANI2xt.

    A molecule needs AIMNET when it carries an element outside ANI_ELEMENTS, a
    nonzero net formal charge, or any unpaired (radical) electron.
    ANI2x/ANI2xt have no notion of spin state -- a radical routed through
    either is silently scored as its closed-shell counterpart, wrong energy
    AND wrong geometry, with nothing in the output to notice by (issue #10;
    same failure shape as the charge/element cases C11 already covers, which
    at least raise). Single implementation of this test -- check_smi_format
    and check_sdf_format used to each inline the element/charge half as their
    own copy of the identical {1, 6, 7, 8, 9, 16, 17} literal, which is
    exactly how the two would silently drift apart (C11); the radical term
    lives here for the same reason rather than being added at each call site.

    Args:
        mol: An RDKit molecule.

    Returns:
        True if `optimizing_engine` must be AIMNET (or a custom NNP) for this
        molecule; False if ANI2x/ANI2xt can represent it.
    """
    # Deferred for the same reason as check_gpu_requested's `import torch`:
    # this module sits on the CLI's `--help` path, and rdkit is not needed
    # until a molecule actually has to be classified.
    from rdkit import Chem

    elements = {a.GetAtomicNum() for a in mol.GetAtoms()}
    charge = Chem.rdmolops.GetFormalCharge(mol)
    n_radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return (not elements.issubset(ANI_ELEMENTS)) or charge != 0 or n_radical_electrons != 0


def check_engine_supports_molecules(
    mols: Chem.Mol | list[Chem.Mol], optimizing_engine: str
) -> None:
    """Raise if `optimizing_engine` cannot represent every molecule in `mols`.

    ANI2x/ANI2xt can only represent uncharged, closed-shell molecules built
    from {H, C, N, O, F, S, Cl}. A charged, out-of-set, or open-shell species
    handed to either is silently evaluated as a different, neutral,
    closed-shell, in-set species -- tens of kcal/mol wrong energy and wrong
    forces, so a downstream "optimized" geometry is wrong too (C11, issue
    #10).

    `check_input` already runs the element/charge/radical half of this check
    (via check_smi_format / check_sdf_format, which call `_requires_aimnet`
    above) for main() and smiles2mols. calc_spe, opt_geometry and calc_thermo
    take an SDF path directly and never go through check_input, so they call
    this function themselves instead.

    AIMNET (and any aimnet registry name) and a path to a custom NNP are not
    restricted by the ANI element/charge set, so this raises nothing for
    them -- but they are still closed-shell-by-default models, so an
    open-shell molecule on that path gets a warning instead of the silent
    nothing it got before (issue #10): one warning per call, not one per
    molecule (calc_thermo separately warns per-molecule, in
    ``Auto3D.entry.ASE.thermo.properties``, when it derives the
    electronic-entropy term -- this is the signal the OTHER three entry
    points, which never reach that code, were missing entirely).

    Args:
        mols: A single RDKit Mol or an iterable of them, read from the
            caller's input SDF.
        optimizing_engine: The engine name exactly as passed to
            calc_spe/opt_geometry/calc_thermo (e.g. 'ANI2x', 'AIMNET', a
            registry name, or a custom NNP path).

    Raises:
        ConfigurationError: `optimizing_engine` is ANI2x/ANI2xt (matched
            case-insensitively, mirroring ModelFactory.create) and at least
            one molecule is charged, open-shell, or contains an element
            outside the ANI training set.
    """
    # Deferred for the same reason as _requires_aimnet's: this module sits on
    # the CLI's `--help` path, and only the isinstance check below (plus
    # whatever _requires_aimnet needs, imported separately there) actually
    # touches rdkit.
    from rdkit import Chem

    mol_list = [mols] if isinstance(mols, Chem.Mol) else list(mols)

    if optimizing_engine.upper() not in BUILTIN_ANI_MODELS:
        # AIMNET / custom-NNP path: no element/charge/radical restriction, but
        # still closed-shell-by-default, so warn rather than stay silent.
        # Batched into ONE warning for the whole call instead of one per
        # molecule -- a run over a large SDF must not turn into a wall of
        # near-identical log lines -- phrased like
        # ASE/thermo/properties.py:319-325's per-molecule warning so the two
        # read as the same policy stated at two different granularities.
        if any(a.GetNumRadicalElectrons() > 0 for mol in mol_list for a in mol.GetAtoms()):
            logger.warning(
                "Open-shell species detected among the input molecules; %s "
                "treats them at the model's closed-shell/default-multiplicity "
                "level.",
                optimizing_engine,
            )
        return

    incompatible = [
        mol.GetProp("_Name") if mol.HasProp("_Name") else "<unnamed>"
        for mol in mol_list
        if _requires_aimnet(mol)
    ]
    if incompatible:
        raise ConfigurationError(
            f"Only AIMNET can handle: {incompatible}, but {optimizing_engine} was "
            "parsed to Auto3D. A molecule requires AIMNET if it has an element "
            "outside the ANI training set, a nonzero net charge, or unpaired "
            "(radical) electrons."
        )
