"""Validation functions for Auto3D.

This module provides input validation and filtering utilities for the Auto3D pipeline.
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import (
    CalcNumUnspecifiedAtomStereoCenters,
)

from Auto3D.exceptions import (
    GPUError,
    DependencyError,
    ConfigurationError,
    ModelLoadError,
)
from Auto3D.utils.logging_config import get_logger

if TYPE_CHECKING:
    pass

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

    # Check --use_gpu
    gpu_flag = args.use_gpu
    if gpu_flag:
        if not torch.cuda.is_available():
            raise GPUError("No cuda device was detected. Please set use_gpu=False.")

    isomer_engine = args.isomer_engine
    if ("OE_LICENSE" not in os.environ) and (isomer_engine == "omega"):
        raise DependencyError(
            "Omega is used as the isomer engine, but OE_LICENSE is not detected. "
            "Please use rdkit."
        )

    # Check the installation for open toolkits, torchani
    if args.isomer_engine == "omega":
        try:
            from openeye import oechem  # noqa: F401
        except ImportError:
            raise DependencyError(
                "Omega is used as isomer engine, but openeye toolkits are not installed."
            )

    if args.optimizing_engine == "ANI2x":
        try:
            import torchani  # noqa: F401
        except ImportError:
            raise DependencyError(
                "ANI2x is used as optimizing engine, but TorchANI is not installed."
            )

    if Path(args.optimizing_engine).exists():
        try:
            model_ = torch.jit.load(args.optimizing_engine)  # noqa: F841
        except (RuntimeError, pickle.UnpicklingError, OSError) as e:
            raise ModelLoadError(
                "A path to a user NNP is used as optimizing engine, but it cannot be loaded. "
                f"Error: {type(e).__name__}: {e}. See this link for information about saving and loading models: "
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

    logger.info("Suggestions for choosing isomer_engine and optimizing_engine: ")
    if ANI:
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info("\tOptimizing engine options: ANI2x, ANI2xt, AIMNET or your own NNP.")
    else:
        logger.info("\tIsomer engine options: RDKit and Omega.")
        logger.info("\tOptimizing engine options: AIMNET or your own NNP.")
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
        ValueError: If SMILES or ID is empty in any line.
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    smiles_all = []
    with open(args.path) as f:
        data = f.readlines()
    for line in data:
        if line.isspace():
            continue
        smiles, id = tuple(line.strip().split())
        if len(smiles) == 0:
            raise ValueError(f"Empty SMILES string in line: {line.strip()!r}")
        if len(id) == 0:
            raise ValueError(f"Empty ID in line: {line.strip()!r}")
        smiles_all.append(smiles)

    logger.info(f"\tThere are {len(data)} SMILES in the input file {args.path}.")
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
        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if not elements.issubset(ANI_elements) or charge != 0:
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
        ValueError: If molecule ID is empty (_Name property is empty).
    """
    ANI_elements = {1, 6, 7, 8, 9, 16, 17}
    ANI = True

    supp = Chem.SDMolSupplier(args.path, removeHs=False)
    mols, only_aimnet_ids = [], []
    for i, mol in enumerate(supp):
        if mol is None:
            logger.warning(f"Skipping invalid molecule at index {i} in SDF")
            continue
        id = mol.GetProp("_Name")
        if len(id) == 0:
            raise ValueError("Empty molecule ID (empty _Name property)")
        mols.append(mol)

        charge = Chem.rdmolops.GetFormalCharge(mol)
        elements = set([a.GetAtomicNum() for a in mol.GetAtoms()])
        if not elements.issubset(ANI_elements) or charge != 0:
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

    # Check GPU configuration
    if use_gpu and not torch.cuda.is_available():
        errors.append("GPU requested but CUDA is not available. Set use_gpu=False.")

    if use_gpu:
        if isinstance(gpu_idx, int):
            if gpu_idx >= torch.cuda.device_count():
                errors.append(f"GPU index {gpu_idx} is invalid. Available GPUs: {torch.cuda.device_count()}")
        elif isinstance(gpu_idx, list):
            for idx in gpu_idx:
                if idx >= torch.cuda.device_count():
                    errors.append(f"GPU index {idx} is invalid. Available GPUs: {torch.cuda.device_count()}")

    # Check optimizing_engine
    valid_engines = {"ANI2x", "ANI2xt", "AIMNET"}
    if optimizing_engine not in valid_engines:
        if not Path(optimizing_engine).exists():
            errors.append(
                f"optimizing_engine must be one of {valid_engines} or a valid path to a custom model. "
                f"Got: {optimizing_engine}"
            )

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
