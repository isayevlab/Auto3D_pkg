"""Validation functions for Auto3D.

This module provides input validation and filtering utilities for the Auto3D pipeline.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import (
    CalcNumUnspecifiedAtomStereoCenters,
)

from Auto3D.constants import BUILTIN_ANI_MODELS
from Auto3D.exceptions import (
    ConfigurationError,
    DependencyError,
    FileFormatError,
    GPUError,
    InputValidationError,
    ModelLoadError,
)
from Auto3D.models.loading import load_custom_nnp
from Auto3D.models.preflight import resolve_engine_name
from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

#: Elements ANI2x/ANI2xt were trained on. AIMNET (and any aimnet registry
#: model) and a custom NNP path are not restricted to this set.
ANI_ELEMENTS = frozenset({1, 6, 7, 8, 9, 16, 17})


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
            f"Only AIMNET can handle: {incompatible}, but {optimizing_engine} "
            "was parsed to Auto3D."
        )


def check_output_not_input(path: str, out_path: str | None) -> None:
    """Refuse to write the output over the input file.

    ``auto3d energy mols.sdf -o mols.sdf`` used to open ``mols.sdf`` for
    writing while the run was still reading from it, so the user's input was
    destroyed -- and, if the run then failed part-way, replaced by a truncated
    file with no surviving copy of either the input or the result (C14). The
    Phase 6 tmp+``os.replace`` staging fixes the *crash* half of C14 (a failed
    rewrite no longer leaves a partial file), but it cannot fix this half: a
    successful same-file run still deliberately overwrites the input, and no
    amount of atomicity brings the original back.

    Single source of truth for that policy, in the same spirit as
    ``check_gpu_requested`` and ``check_engine_supports_molecules``:
    ``calc_spe``, ``opt_geometry`` and ``calc_thermo`` each take an output path
    directly and never go through ``check_input``/``check_valid_configuration``,
    so all three call this function rather than carrying three copies of the
    test that would drift apart. The ``auto3d energy``/``optimize``/``thermo``
    CLI commands pass ``--output`` straight through to those functions, so they
    are covered by the same call.

    Compares resolved real paths, not the strings: ``mols.sdf``,
    ``./mols.sdf``, an absolute path to the same file, and a symlink pointing
    at it all name one file, and ``os.path.realpath`` collapses all four.
    ``realpath`` does not require either path to exist, so an output path that
    has not been created yet compares correctly.

    Args:
        path: The input file the caller will read.
        out_path: The requested output path, or None to use the default
            (which is derived from `path` and never equals it).

    Raises:
        ConfigurationError: `out_path` resolves to the same file as `path`.
    """
    if out_path is None:
        return
    if os.path.realpath(path) == os.path.realpath(out_path):
        raise ConfigurationError(
            f"Output path {out_path!r} is the same file as the input {path!r}. "
            "Auto3D would overwrite your input; pass a different output path."
        )


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
            - opt_steps: Number of optimization steps
            - input_format: Input file format ('smi' or 'sdf')
            - path: Path to input file
            - enumerate_isomer: Whether to enumerate stereoisomers

    Returns:
        None. The function prints recommendations.

    Raises:
        GPUError: If GPU is requested but not available.
        DependencyError: If required dependency not available (OpenEye, TorchANI).
        ConfigurationError: If configuration parameters are invalid (opt_steps, engine mismatch).
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
            "Omega is used as the isomer engine, but OE_LICENSE is not detected. "
            "Please use rdkit.",
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
        try:
            load_custom_nnp(args.optimizing_engine, torch.device("cpu"))
        except ModelLoadError as e:
            raise ModelLoadError(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
                f"{e} See this link for information about saving and loading models: "
                "https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model"
            ) from e

    if int(args.opt_steps) < 10:
        raise ConfigurationError(
            f"Number of optimization steps cannot be smaller than 10, but received {args.opt_steps}"
        )

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

    smiles_all = []
    with open(args.path) as f:
        data = f.readlines()
    for line in data:
        if line.isspace():
            continue
        # Skip '#'-prefixed comment lines, matching cli.commands.validate.
        # validate_smiles_file and file_ops.iter_smi_records -- all three must
        # agree on what counts as a comment vs. data, or `auto3d validate`
        # would approve a file this function then rejects (M25).
        if line.lstrip().startswith("#"):
            continue
        # Tolerate ragged rows the way the rest of the pipeline does: the chunk
        # loader reads only the first two whitespace columns (usecols=[0, 1]), so
        # trailing tokens (e.g. an inline comment column) must not be rejected
        # here. split() never yields empty tokens, so a present SMILES/ID is
        # guaranteed non-empty.
        parts = line.split()
        if len(parts) < 2:
            raise InputValidationError(
                "Each non-blank line must contain a SMILES and an ID separated by "
                f"whitespace, but got: {line.strip()!r}"
            )
        smiles = parts[0]  # parts[1] is the ID; its presence is enforced above
        smiles_all.append(smiles)

    logger.info(f"\tThere are {len(smiles_all)} SMILES in the input file {args.path}.")
    logger.info("\tAll SMILES and IDs are valid.")

    # Check number of unspecified atomic stereo center
    if not args.enumerate_isomer:
        for smiles in smiles_all:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                warnings.warn(f"Failed to parse SMILES: {smiles}", UserWarning)
                continue
            c = CalcNumUnspecifiedAtomStereoCenters(mol)
            if c > 0:
                msg = (
                    f"{smiles} contains unspecified atomic stereo centers, but enumerate_isomer=False. "
                    "Please use enumerate_isomer=True so that Auto3D can enumerate the "
                    "unspecified atomic stereo centers."
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


def check_valid_configuration(
    path: str | None = None,
    k: int | bool = False,
    window: float | bool = False,
    use_gpu: bool = True,
    gpu_idx: int | list[int] = 0,
    optimizing_engine: str = "AIMNET",
    isomer_engine: str = "rdkit",
    opt_steps: int = 2000,
    enumerate_tautomer: bool = False,
    tauto_engine: str = "rdkit",
) -> list[str]:
    """Validate Auto3D configuration parameters.

    This function checks if the provided configuration parameters are valid
    and compatible with each other.

    Args:
        path: Path to input file. Must be provided and exist.
        k: Number of top conformers to keep. Either k or window must be specified.
        window: Energy window in kcal/mol for conformer selection. Either k or window must be specified.
        use_gpu: Whether to use GPU acceleration.
        gpu_idx: GPU device index or list of indices.
        optimizing_engine: Engine for geometry optimization. Must be one of
            'ANI2x', 'ANI2xt', 'AIMNET' or a valid path to a custom model.
        isomer_engine: Engine for isomer enumeration. Must be 'rdkit' or 'omega'.
        opt_steps: Number of optimization steps. Must be >= 10.
        enumerate_tautomer: Whether to enumerate tautomers.
        tauto_engine: Engine for tautomer enumeration. Must be 'rdkit' or 'oechem'.

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
    errors: list[str] = []

    # Check path
    if path is None:
        errors.append("Input path must be provided.")
    elif not Path(path).exists():
        errors.append(f"Input path does not exist: {path}")

    # Check k and window
    if not k and not window:
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
                errors.append(f"GPU index {gpu_idx} is invalid. Available GPUs: {torch.cuda.device_count()}")
        elif isinstance(gpu_idx, list):
            for idx in gpu_idx:
                if idx >= torch.cuda.device_count():
                    errors.append(f"GPU index {idx} is invalid. Available GPUs: {torch.cuda.device_count()}")

    # Resolve rather than prefix-match: `aimnet2-2025x` starts with "aimnet2"
    # and so used to pass here, then failed inside a worker where
    # optim_rank_wrapper's per-chunk handler swallowed it. The registry lookup
    # is a pure offline dict read against a bundled YAML, so validating costs
    # nothing.
    try:
        resolve_engine_name(optimizing_engine)
    except ConfigurationError as exc:
        errors.append(str(exc))

    # Check isomer_engine
    valid_isomer_engines = {"rdkit", "omega"}
    if isomer_engine.lower() not in valid_isomer_engines:
        errors.append(f"isomer_engine must be one of {valid_isomer_engines}. Got: {isomer_engine}")

    # Check OpenEye license for omega
    if isomer_engine.lower() == "omega" and "OE_LICENSE" not in os.environ:
        errors.append("OpenEye license (OE_LICENSE) not found but omega isomer_engine is selected.")

    # Check opt_steps
    if opt_steps < 10:
        errors.append(f"opt_steps must be >= 10. Got: {opt_steps}")

    # Check tautomer configuration
    valid_tauto_engines = {"rdkit", "oechem"}
    if enumerate_tautomer and tauto_engine.lower() not in valid_tauto_engines:
        errors.append(f"tauto_engine must be one of {valid_tauto_engines}. Got: {tauto_engine}")

    if enumerate_tautomer and tauto_engine.lower() == "oechem" and "OE_LICENSE" not in os.environ:
        errors.append("OpenEye license (OE_LICENSE) not found but oechem tauto_engine is selected.")

    return errors
