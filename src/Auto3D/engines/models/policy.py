"""What a model requires of its host and of its molecules.

Two preconditions that belong to the model layer rather than to generic
helpers, and that every caller of a model has to satisfy: is the device the
caller asked for actually there, and can the engine they chose represent the
molecules they handed it. Both were in ``utils/validation.py`` alongside input
and configuration parsing, which made ``utils`` -- a leaf by intent -- depend on
the model package, and made a property calculator import a module named for
validating command-line input in order to ask whether a GPU exists.

Layering: ``torch`` and ``rdkit`` are NOT imported at module scope here, even
though every function below eventually needs one or the other. Both third-party
packages are the expensive ones, and importing either at module scope would
charge that cost to every future caller that imports this module from L5,
even one that only wants a name it exports and never calls a function that
needs torch or rdkit. ``commands/properties.py`` does not import this
module at all; the guards it needs run in the entry points' shared prologue
(``Auto3D.entry._run_setup.prepare_single_file_run``). The module-scope
importers today are ``_run_setup.py``, ``commands/models.py``,
``workflow_workers.py`` and ``pipeline/input_checks.py``, and the
deferrals are what keep those imports -- and any future L5 importer --
cheap. Each function defers its own import, with a comment saying why --
see ``check_gpu_requested`` and ``requires_aimnet`` /
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
    fallback**. A scripted user who sets ``use_gpu=True`` and silently gets
    CPU has no way to know their "GPU" results were actually computed on
    CPU -- possibly orders of magnitude slower than they assumed, with no
    signal anything was wrong. This function is called as the *first* check everywhere GPU
    use is decided -- ``check_input`` and ``check_valid_configuration`` for
    ``main()``/``smiles2mols``, the shared prologue
    ``Auto3D.entry._run_setup.prepare_single_file_run`` (its second step) for
    ``calc_spe``/``opt_geometry``/``calc_thermo``, which take an SDF path
    directly and never go through ``check_input``, and ``models test``
    itself -- so it fails fast, before any worker is forked and before any
    compute is spent, with the same exception type and the same "--no-gpu"
    hint regardless of entry point.

    Args:
        use_gpu: The ``use_gpu`` option requested by the caller.

    Raises:
        GPUError: `use_gpu` is True and `torch.cuda.is_available()` is False.
    """
    # Deferred: this is the ONLY torch user in the module (verified -- every
    # other function below needs rdkit, not torch), and it is reached on every
    # run that decides anything about a GPU. Importing torch at module scope
    # would charge that cost unconditionally, regardless of whether use_gpu
    # was even True (issue #14), to every future caller that imports this
    # module from L5, not just the ones that need it.
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
                "No cuda device was detected, but use_gpu=True was requested; "
                "set use_gpu=False to run on the CPU.",
                # The CLI's "--no-gpu" flag spelling lives in
                # cli.errors.get_error_hint's hint-key table, not here --
                # this message stays usable from the Python API, which has
                # no CLI flags to spell.
                hint_key="gpu_unavailable",
            )


def has_dummy_atoms(mol: Chem.Mol) -> bool:
    """True if `mol` carries a dummy atom (atomic number 0) -- see the definition
    in :func:`Auto3D.foundation.utils.molprops.has_dummy_atoms`.

    The predicate itself lives in ``foundation`` because ``utils/sdf_io.py``
    applies it per SDF record and ``foundation`` may not import ``engines``.
    It is surfaced here as well because it is a *model* precondition of the
    same kind as :func:`requires_aimnet`: a model-layer caller asking "can the
    engine represent this molecule" should find it next to the other answer
    rather than in a graph-properties helper.

    A delegating call, not ``from ... import has_dummy_atoms``: a module-scope
    import of ``molprops`` would pull rdkit in at *this* module's import time
    (``molprops`` imports ``rdkit`` at module scope), and this module sits on
    the CLI's ``--help`` path -- the exact cost issue
    #14 removed and ``tests/test_import_boundaries.py::
    test_cli_app_import_does_not_load_torch_or_rdkit`` now forbids. Deferring
    the import into the function body is the same pattern every other function
    in this module uses, for the same reason.

    Args:
        mol: An RDKit molecule.

    Returns:
        True if the molecule carries at least one dummy atom.
    """
    from Auto3D.foundation.utils.molprops import has_dummy_atoms as _has_dummy_atoms

    return _has_dummy_atoms(mol)


def requires_aimnet(mol: Chem.Mol) -> bool:
    """True if `mol` cannot be represented by ANI2x/ANI2xt.

    A molecule needs AIMNET when it carries an element outside ANI_ELEMENTS, a
    nonzero net formal charge, or any unpaired (radical) electron.
    ANI2x/ANI2xt have no notion of spin state -- a radical routed through
    either is silently scored as its closed-shell counterpart, wrong energy
    AND wrong geometry, with nothing in the output to notice by (issue #10) --
    the same failure shape as the charge/element cases already covered,
    which at least raise. Single implementation of this test, covering the
    element/charge half as well as the radical term, so ``check_smi_format``
    and ``check_sdf_format`` cannot each carry their own copy of the
    identical {1, 6, 7, 8, 9, 16, 17} literal and silently drift apart.

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


# Back-compat alias: every pre-existing in-module and test call site
# spells this name with the leading underscore; `requires_aimnet` above is
# the public name new callers (orchestration's input checks) should use.
_requires_aimnet = requires_aimnet


def _is_zwitterion(mol: Chem.Mol) -> bool:
    """True if `mol` is net-neutral but carries charge separated across its
    skeleton, rather than merely depicted with formal charges on one bond.

    A zwitterion (e.g. the amino-acid form ``[NH3+]CC(=O)[O-]``) passes
    ``requires_aimnet``'s charge test unchanged -- the net charge really is
    zero -- so it reaches ANI2x/ANI2xt silently today. Both networks take no
    total-charge input and were parameterized on net-neutral, closed-shell
    molecules, so a zwitterion's internal charge separation is an
    extrapolation for them (AIMNET takes the molecular charge as an input and
    supports charged and net-neutral zwitterionic species, so it needs no
    such warning).

    The required/forbidden distinction: a
    cation directly bonded to an anion is one functional group drawn with
    formal charges rather than two separated sites, so nitro
    (``[O-][N+](=O)...``), N-oxides, sulfoxides written with formal charges,
    azides and diazo compounds -- where the opposite charges sit on bonded
    neighbors -- must stay silent. Flagging requires BOTH a cation with no
    directly-bonded anionic neighbor AND an (not necessarily the same) anion
    with no directly-bonded cationic neighbor: each is a charge with nothing
    adjacent to balance it locally, the actual charge-separated-across-the-
    skeleton shape glycine, betaine and taurine have.

    Args:
        mol: An RDKit molecule.

    Returns:
        True if the molecule's net formal charge is zero and it carries at
        least one "free" cation (no directly-bonded anionic neighbor) and at
        least one "free" anion (no directly-bonded cationic neighbor).
    """
    # Deferred for the same reason as requires_aimnet's.
    from rdkit import Chem

    if Chem.rdmolops.GetFormalCharge(mol) != 0:
        return False
    cations = [a for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
    anions = [a for a in mol.GetAtoms() if a.GetFormalCharge() < 0]
    if not cations or not anions:
        return False
    free_cation = any(
        not any(nbr.GetFormalCharge() < 0 for nbr in cation.GetNeighbors()) for cation in cations
    )
    free_anion = any(
        not any(nbr.GetFormalCharge() > 0 for nbr in anion.GetNeighbors()) for anion in anions
    )
    return free_cation and free_anion


def check_engine_supports_molecules(
    mols: Chem.Mol | list[Chem.Mol], optimizing_engine: str
) -> None:
    """Raise if `optimizing_engine` cannot represent every molecule in `mols`.

    ANI2x/ANI2xt can only represent uncharged, closed-shell molecules built
    from {H, C, N, O, F, S, Cl}. A charged, out-of-set, or open-shell species
    handed to either is silently evaluated as a different, neutral,
    closed-shell, in-set species -- tens of kcal/mol wrong energy and wrong
    forces, so a downstream "optimized" geometry is wrong too (issue #10).

    `check_input` already runs the element/charge/radical half of this check
    (via check_smi_format / check_sdf_format, which call `requires_aimnet`
    above) for main() and smiles2mols. calc_spe, opt_geometry and calc_thermo
    take an SDF path directly and never go through check_input, so they call
    this function themselves instead.

    AIMNET (and any aimnet registry name) and a path to a custom NNP are not
    restricted by the ANI element/charge set, so this raises nothing for
    them -- but they are still closed-shell-by-default models, so an
    open-shell molecule on that path gets a warning instead of the silent
    nothing it got before (issue #10): one warning per call, not one per
    molecule (calc_thermo separately warns per-molecule, in
    ``Auto3D.domain.multiplicity.resolve_multiplicity``, when it derives the
    electronic-entropy term -- this is the signal the OTHER three entry
    points, which never reach that code, were missing entirely).

    A net-zero zwitterion requested on ANI2x/ANI2xt is not an error either --
    its net charge really is zero, so ``requires_aimnet`` does not flag it --
    but those networks take no total-charge input and were parameterized on
    net-neutral, closed-shell molecules, so a zwitterion's internal charge
    separation is an extrapolation for them; it gets the same
    one-warning-per-call treatment (see :func:`_is_zwitterion`).

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
    # Deferred for the same reason as requires_aimnet's: this module sits on
    # the CLI's `--help` path, and only the isinstance check below (plus
    # whatever requires_aimnet needs, imported separately there) actually
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

    # Every molecule here passed requires_aimnet (net charge zero, in-set
    # elements, no radical), so a zwitterion among them is not an error -- but
    # it is still an extrapolation for ANI2x/ANI2xt (see _is_zwitterion).
    # Batched into ONE warning per call, same reasoning as the AIMNET/
    # custom-NNP branch's radical warning above.
    if any(_is_zwitterion(mol) for mol in mol_list):
        logger.warning(
            "Net-zero zwitterion(s) detected among the input molecules; the "
            "ANI networks take no total-charge input and were parameterized "
            "on net-neutral closed-shell molecules, so a zwitterion's "
            "internal charge separation is an extrapolation; AIMNET takes "
            "the molecular charge as an input and supports charged and "
            "net-neutral zwitterionic species.",
        )
