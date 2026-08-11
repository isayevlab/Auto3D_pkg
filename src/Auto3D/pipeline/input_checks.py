"""Input and configuration checks that run before a pipeline starts.

Everything here inspects what the *caller* supplied -- the input file, its
format, the option object -- and runs in the process that parses the
configuration, before any worker is forked. It was in ``utils/validation.py``,
from where it reached up into ``Auto3D.models`` through two function-scope
imports that existed solely to avoid an import cycle. Those are module-scope
imports now: this package sits above the model layer, so it may simply say so.

Named ``input_checks`` rather than ``preflight`` deliberately. ``models/
preflight.py`` already exists and resolves a model *name* in the parent process
before forking; a second ``preflight`` module checking input files would be one
of the near-identical name pairs this codebase has been burned by.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rdkit import Chem

from Auto3D.exceptions import (
    ConfigurationError,
    DependencyError,
    FileFormatError,
    InputValidationError,
    ModelLoadError,
)
from Auto3D.models.loading import load_custom_nnp
from Auto3D.models.policy import (
    _requires_aimnet,
    check_gpu_requested,
)
from Auto3D.models.preflight import resolve_engine_name
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.smi_io import iter_smi_records
from Auto3D.utils.stereochemistry import count_unspecified_stereo

if TYPE_CHECKING:
    from Auto3D.config import Auto3DOptions

logger = get_logger(__name__)


def check_input(args: Any) -> None:
    """Check the input file and give recommendations.

    This function validates the input arguments and the input file format,
    checking for compatibility with the selected isomer and optimizing engines.

    Args:
        args: Arguments object containing Auto3D configuration options.
            Expected attributes:
            - use_gpu: Whether to use GPU acceleration
            - isomer_engine: Engine for isomer enumeration ('rdkit' or 'omega')
            - optimizing_engine: Engine for geometry optimization ('ANI2x', 'ANI2xt', 'AIMNET', or path)
            - input_format: Input file format ('smi' or 'sdf')
            - path: Path to input file
            - enumerate_isomer: Whether to enumerate stereoisomers

            ``opt_steps`` is no longer read here: its minimum lives in
            ``Auto3D.config.FIELD_BOUNDS`` and is enforced when the
            configuration is constructed, not when it is used.

    Returns:
        None. The function prints recommendations.

    Raises:
        GPUError: If GPU is requested but not available.
        DependencyError: If required dependency not available (OpenEye, TorchANI).
        ConfigurationError: If the optimizing engine cannot represent the input
            molecules (charged, or outside the ANI element set).
        ModelLoadError: If custom NNP cannot be loaded.
    """
    logger.info("Checking input file...")

    # Check --use_gpu. Delegates to check_gpu_requested so this and
    # check_valid_configuration can never drift onto different wording or a
    # different exception type again (M23).
    check_gpu_requested(args.use_gpu)

    isomer_engine = args.isomer_engine
    if ("OE_LICENSE" not in os.environ) and (isomer_engine == "omega"):
        raise DependencyError(
            "Omega is used as the isomer engine, but OE_LICENSE is not detected. Please use rdkit.",
            dependency_name="openeye",
        )

    # Check the installation for open toolkits, torchani
    if args.isomer_engine == "omega":
        try:
            from openeye import oechem  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Omega is used as isomer engine, but openeye toolkits are not installed.",
                dependency_name="openeye",
            )

    if args.optimizing_engine == "ANI2x":
        try:
            import torchani  # noqa: F401
        except ImportError:
            raise DependencyError(
                "ANI2x is used as optimizing engine, but TorchANI is not installed.",
                dependency_name="torchani",
            )

    if Path(args.optimizing_engine).exists():
        # Validate that a custom NNP path loads (TorchScript archive or eager
        # nn.Module); shared load contract -- see Auto3D.models.loading.
        #
        # Function-scope on purpose: `utils` is the bottom of the stack and must
        # not import the `models` domain package at module level. Reached only
        # when the engine really is a path on disk.

        try:
            load_custom_nnp(args.optimizing_engine, torch.device("cpu"))
        except ModelLoadError as e:
            raise ModelLoadError(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
                f"{e} See this link for information about saving and loading models: "
                "https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model"
            ) from e

    # No opt_steps check here any more. It hand-wrote `< 10` while
    # Auto3D.config.FIELD_BOUNDS declared ("ge", 1) -- two minimums for one
    # option (see that table's comment). FIELD_BOUNDS now declares 10 and is
    # enforced by Auto3DOptions.__post_init__ and CLIConfig's _check_bounds, so
    # every entry point rejects opt_steps=5 at construction, before this
    # function is reached and before any banner prints.

    # Check the input format
    if args.input_format == "smi":
        ANI, only_aimnet_smiles = check_smi_format(args)
    elif args.input_format == "sdf":
        ANI, only_aimnet_smiles = check_sdf_format(args)
    else:
        raise FileFormatError(
            f"Input file type is not supported. Only .smi and .sdf are supported, "
            f"but input_format is {args.input_format!r}."
        )

    logger.info("Suggestions for choosing isomer_engine and optimizing_engine: ")
    if ANI:
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info(
            "\tOptimizing engine options: AIMNET (or an aimnet registry name like "
            "aimnet2-2025), ANI2x, ANI2xt, or your own NNP."
        )
    else:
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info(
            "\tOptimizing engine options: AIMNET (or an aimnet registry name like "
            "aimnet2-2025), or your own NNP."
        )
        optimizing_engine = args.optimizing_engine
        if optimizing_engine in {"ANI2x", "ANI2xt"}:
            raise ConfigurationError(
                f"Only AIMNET can handle: {only_aimnet_smiles}, but {optimizing_engine} was parsed to Auto3D."
            )


def check_smi_format(args: Any) -> tuple[bool, list[str]]:
    """Check the SMILES input file format and validate molecules.

    Args:
        args: Arguments object containing Auto3D configuration options.
            Expected attributes:
            - path: Path to the input SMILES file
            - enumerate_isomer: Whether to enumerate stereoisomers

    Returns:
        A tuple containing:
            - ANI: Boolean indicating if all molecules are compatible with ANI models
            - only_aimnet_smiles: List of SMILES that require AIMNET (contain non-ANI elements or charged)

    Raises:
        InputValidationError: If a non-blank line lacks a SMILES and an ID.
    """
    ANI = True

    # iter_smi_records is the single parser for this format (M59): it already
    # skips blank lines and '#'-prefixed comments the same way
    # cli.commands.validate.validate_smiles_file does, and its on_malformed
    # ("raise") gives the same InputValidationError this loop used to raise by
    # hand for a line missing an ID, so `auto3d validate` and this check
    # cannot silently disagree about what a well-formed line looks like
    # (M25). The parser also tolerates ragged rows (extra whitespace columns
    # beyond SMILES+ID), matching the chunk loader's usecols=[0, 1].
    smiles_all = [
        smiles for _line_no, smiles, _id in iter_smi_records(args.path, on_malformed="raise")
    ]

    logger.info(f"\tThere are {len(smiles_all)} SMILES in the input file {args.path}.")
    logger.info("\tAll SMILES and IDs are valid.")

    # Warn about every stereo element the input leaves open -- tetrahedral
    # centers AND double-bond geometry. This used to call
    # CalcNumUnspecifiedAtomStereoCenters, which sees only ATOM centers, so an
    # unspecified C=C passed silently: with enumerate_isomer=False,
    # "OC(=O)C=CC(=O)O" embeds as fumaric AND maleic acid (~5 kcal/mol apart)
    # under one species id, and "CC=CC" embeds as cis-2-butene alone with the
    # trans isomer absent -- in both cases the user gets a molecule they did
    # not submit, or loses one they did. count_unspecified_stereo is the same
    # predicate RDKitSdfIsomer uses, so the SMILES and SDF paths agree.
    if not args.enumerate_isomer:
        for smiles in smiles_all:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                warnings.warn(f"Failed to parse SMILES: {smiles}", UserWarning)
                continue
            c = count_unspecified_stereo(mol)
            if c > 0:
                msg = (
                    f"{smiles} contains {c} unspecified stereo element(s) "
                    "(atomic stereo centers and/or double-bond geometry), but "
                    "enumerate_isomer=False, so its conformers will be a mixture "
                    "of configurations -- or one arbitrary configuration. "
                    "Please use enumerate_isomer=True so that Auto3D can enumerate "
                    "the unspecified stereo elements."
                )
                warnings.warn(msg, UserWarning)

    # Check the properties of molecules
    only_aimnet_smiles = []
    for smiles in smiles_all:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES: {smiles}")
            continue
        if _requires_aimnet(mol):
            ANI = False
            only_aimnet_smiles.append(smiles)
    return ANI, only_aimnet_smiles


def check_sdf_format(args: Any) -> tuple[bool, list[str]]:
    """Check the SDF input file format and validate molecules.

    This function validates the format of the input SDF file, checking the properties
    for each molecule in the input file.

    Args:
        args: Arguments object containing Auto3D configuration options.
            Expected attributes:
            - path: Path to the input SDF file
            - enumerate_isomer: Whether to enumerate stereoisomers

    Returns:
        A tuple containing:
            - ANI: Boolean indicating if all molecules are compatible with ANI models
            - only_aimnet_ids: List of molecule IDs that require AIMNET

    Raises:
        InputValidationError: If molecule ID is empty (_Name property is empty).
    """
    ANI = True

    supp = Chem.SDMolSupplier(args.path, removeHs=False)
    mols, only_aimnet_ids = [], []
    for i, mol in enumerate(supp):
        if mol is None:
            logger.warning(f"Skipping invalid molecule at index {i} in SDF")
            continue
        id = mol.GetProp("_Name")
        if len(id) == 0:
            # Same defect as check_smi_format's missing-ID check above --
            # both must raise the same Auto3DError subclass so the CLI shows
            # the same hint and exit code regardless of input format.
            raise InputValidationError("Empty molecule ID (empty _Name property)")
        mols.append(mol)

        if _requires_aimnet(mol):
            ANI = False
            only_aimnet_ids.append(id)

    logger.info(f"\tThere are {len(mols)} conformers in the input file {args.path}.")
    logger.info("\tAll conformers and IDs are valid.")

    if args.enumerate_isomer:
        msg = (
            "Enumerating stereocenters of an SDF file could change the conformers of the input file. "
            "Please use enumerate_isomer=False."
        )
        warnings.warn(msg, UserWarning)
    return ANI, only_aimnet_ids


def check_valid_configuration(options: Auto3DOptions) -> list[str]:
    """Validate an ``Auto3DOptions`` for things its own construction cannot check.

    Takes the configuration **object**, not a copy of its field names. The
    previous signature re-declared ten option names *and their own defaults*
    (``optimizing_engine="AIMNET"``, ``isomer_engine="rdkit"``,
    ``tauto_engine="rdkit"``, ``opt_steps=2000`` written as a literal rather
    than ``DEFAULT_OPT_STEPS``, ...), which made this function a third
    configuration schema alongside ``Auto3DOptions`` and ``CLIConfig`` -- one
    that could disagree with both about what an unspecified option means, and
    that silently never looked at the other eighteen fields. It also forced two
    byte-identical ten-keyword marshalling blocks at its only two call sites
    (``auto3D.py``'s ``smiles2mols`` and ``workflow.py``'s
    ``WorkflowOrchestrator._validate_input``), both of which read every value
    straight off an ``Auto3DOptions`` they already had.

    ``Auto3DOptions`` is the authoritative schema, so what belongs here is only
    what a dataclass cannot decide for itself: whether the input file exists,
    whether a selector was chosen, whether the requested GPU index exists on
    *this* machine, whether the engine name resolves in the model registry, and
    whether the OpenEye license the chosen engines need is present in the
    environment. Everything checkable from the values alone -- numeric bounds
    (``FIELD_BOUNDS``, including ``opt_steps >= 10``), the isomer/tautomer
    engine whitelists (``ENGINE_CHOICES``), and selector mutual exclusion --
    has already run in ``__post_init__``, so re-checking it here would be the
    duplicate this change removes.

    Args:
        options: The configuration to check. Any object exposing
            ``Auto3DOptions``'s attributes works, matching ``check_input``.

    Returns:
        List of error messages. Empty list if configuration is valid.

    Raises:
        GPUError: `use_gpu` is True and no CUDA device is visible. Raised
            immediately (not folded into the returned `errors` list) so every
            caller of this function -- main() via
            WorkflowOrchestrator._validate_input, and smiles2mols -- gets the
            same fatal GPUError, with the same "--no-gpu" hint, that
            check_input already raised for this condition (M23). See
            check_gpu_requested for the full rationale.
    """
    path = options.path
    use_gpu = options.use_gpu
    gpu_idx = options.gpu_idx
    isomer_engine = options.isomer_engine

    errors: list[str] = []

    # Check path
    if path is None:
        errors.append("Input path must be provided.")
    elif not Path(path).exists():
        errors.append(f"Input path does not exist: {path}")

    # Check k and window
    if not options.k and not options.window:
        errors.append("Either 'k' or 'window' must be specified for conformer selection.")

    # Check GPU configuration. Raises immediately rather than appending to
    # `errors`: every caller wraps a non-empty `errors` list into a single
    # ConfigurationError, which would show the CLI's "config init" hint --
    # unrelated to a GPU problem. Also means the gpu_idx range check below
    # only runs once CUDA is confirmed available, so an unavailable-CUDA box
    # never sees a confusing second "0 available GPUs" message.
    check_gpu_requested(use_gpu)

    if use_gpu:
        if isinstance(gpu_idx, int):
            if gpu_idx >= torch.cuda.device_count():
                errors.append(
                    f"GPU index {gpu_idx} is invalid. Available GPUs: {torch.cuda.device_count()}"
                )
        elif isinstance(gpu_idx, list):
            for idx in gpu_idx:
                if idx >= torch.cuda.device_count():
                    errors.append(
                        f"GPU index {idx} is invalid. Available GPUs: {torch.cuda.device_count()}"
                    )

    # Resolve rather than prefix-match: `aimnet2-2025x` starts with "aimnet2"
    # and so used to pass here, then failed inside a worker where
    # optim_rank_wrapper's per-chunk handler swallowed it. The registry lookup
    # is a pure offline dict read against a bundled YAML, so validating costs
    # nothing.
    #
    # Function-scope for the same reason as load_custom_nnp above: it keeps
    # `utils` from importing the `models` domain package at module level.

    try:
        resolve_engine_name(options.optimizing_engine)
    except ConfigurationError as exc:
        errors.append(str(exc))

    # No isomer_engine/tauto_engine whitelist here any more: both are in
    # Auto3D.config.ENGINE_CHOICES and enforced by Auto3DOptions.__post_init__,
    # so an unrecognized value cannot reach this function. The two local
    # `valid_*_engines` sets that used to stand here were the third and fourth
    # hand-written copies of those whitelists.
    #
    # The license checks below stay: they are about the *environment*, not about
    # the value, so no amount of construction-time validation can answer them.

    # Check OpenEye license for omega
    if isomer_engine.lower() == "omega" and "OE_LICENSE" not in os.environ:
        errors.append("OpenEye license (OE_LICENSE) not found but omega isomer_engine is selected.")

    if (
        options.enumerate_tautomer
        and options.tauto_engine.lower() == "oechem"
        and "OE_LICENSE" not in os.environ
    ):
        errors.append("OpenEye license (OE_LICENSE) not found but oechem tauto_engine is selected.")

    return errors
