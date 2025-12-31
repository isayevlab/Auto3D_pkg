#!/usr/bin/env python
"""
Providing general utilities for working with different formats of molecular files

.. deprecated:: 1.0
    This module is deprecated. Use :mod:`Auto3D.utils.file_ops` instead.
    Functions will be removed in Auto3D v2.0.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


def _emit_deprecation_warning(old_name: str, new_location: str) -> None:
    """Emit deprecation warning for moved functions."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in Auto3D v2.0. "
        f"Use {new_location} instead.",
        DeprecationWarning,
        stacklevel=3
    )


def guess_file_type(filename: str) -> str:
    """Return the file extension for a given filename.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.guess_file_type` instead.

    Args:
        filename: Path or filename to analyze.

    Returns:
        The file extension without the leading dot.
    """
    _emit_deprecation_warning("guess_file_type", "Auto3D.utils.file_ops.guess_file_type")
    return Path(filename).suffix[1:]

# Functions related to smi files
def smiles2smi(smiles: list[str], path: str) -> str:
    """Convert a list of SMILES strings to a .smi file with InChIKey IDs.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.smiles2smi` instead.

    Args:
        smiles: List of SMILES strings to convert.
        path: Output file path for the .smi file.

    Returns:
        The output file path.
    """
    _emit_deprecation_warning("smiles2smi", "Auto3D.utils.file_ops.smiles2smi")
    lines = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        inchikey = inchi.MolToInchiKey(mol)
        lines.append(f"{smi}  {inchikey}\n")
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)
    return path

def combine_smi(smies: list[str], out: str) -> None:
    """Combine multiple SMILES files into a single file.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.combine_smi` instead.

    Args:
        smies: List of paths to input .smi files.
        out: Path for the combined output .smi file.

    Returns:
        None. Writes the combined result to the output file.
    """
    _emit_deprecation_warning("combine_smi", "Auto3D.utils.file_ops.combine_smi")
    data = []
    for smi in smies:
        with open(smi) as f:
            datai = f.readlines()
        data += datai
    data = list(set(data))
    with open(out, 'w+') as f2:
        for line in data:
            if not line.isspace():
                f2.write(line.strip() + '\n')

# Functions related to SDF files
def countSDF(sdf: str) -> int:
    """Count the number of molecules in an SDF file.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.count_sdf` instead.

    Args:
        sdf: Path to the SDF file.

    Returns:
        Number of molecules in the file.
    """
    _emit_deprecation_warning("countSDF", "Auto3D.utils.file_ops.count_sdf")
    mols = Chem.SDMolSupplier(sdf)
    mols2 = [mol for mol in mols]
    c = len(mols2)
    return c

def SDF2chunks(sdf: str) -> list[list[str]]:
    """Split an SDF file into chunks, one per molecule.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.SDF2chunks` instead.

    Args:
        sdf: Path to the input SDF file.

    Returns:
        List of chunks, where each chunk is a list of strings (lines)
        representing one molecule including the '$$$$' terminator.
    """
    _emit_deprecation_warning("SDF2chunks", "Auto3D.utils.file_ops.SDF2chunks")
    chunks = []
    with open(sdf) as f:
        data = f.readlines()
    chunk = []
    for line in data:
        if line.strip() == "$$$$":
            chunk.append(line)
            chunks.append(chunk)
            chunk = []
        else:
            chunk.append(line)
    return chunks
       
def find_smiles_not_in_sdf(smi: str, sdf: str) -> list[tuple[str, str]]:
    """Find SMILES that failed to generate 3D conformers.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.find_smiles_not_in_sdf` instead.

    Args:
        smi: Path to input SMILES file.
        sdf: Path to output SDF file.

    Returns:
        List of (id, smiles) tuples for molecules not in SDF.
    """
    _emit_deprecation_warning(
        "find_smiles_not_in_sdf",
        "Auto3D.utils.file_ops.find_smiles_not_in_sdf"
    )
    #find all SMILES ids
    smi_names = []
    with open(smi) as f:
        data = f.readlines()
    for line in data:
        smi, id = tuple(line.strip().split())
        smi_names.append((smi.strip(), id.strip()))
    
    sdf_data = []
    mols = Chem.SDMolSupplier(sdf)
    for mol in mols:
        sdf_data.append(mol.GetProp("_Name"))
    sdf_data = list(set(sdf_data))

    bad = []
    for smi, id in smi_names:
        has_3D_structure = False
        # for line in sdf_data:
        #     if id in line:
        #         has_3D_structure = True
        if id in sdf_data:
            has_3D_structure = True
        if not has_3D_structure:
            bad.append((id, smi))

    if len(bad) > 0:
        logger.warning("The following SMILES has no 3D structure in the SDF file.")
        logger.warning("ID, SMILES")
        for id, smi in bad:
            logger.warning(f"{id} {smi}")
    else:
        logger.info("Every SMILES has at least an 3D structure in the SDF file.")
    return bad

def encode_ids(path: str) -> tuple[str, dict]:
    """Encode molecule IDs to numeric indices.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.encode_ids` instead.

    Args:
        path: Path to the input .smi or .sdf file.

    Returns:
        Tuple containing:
        - Path to the new file with encoded IDs (adds '_encoded' suffix)
        - Dictionary mapping original IDs to their numeric indices

    Raises:
        ValueError: If the input file is neither .smi nor .sdf format.
    """
    _emit_deprecation_warning("encode_ids", "Auto3D.utils.file_ops.encode_ids")
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    new_path = path_obj.parent / f"{path_obj.stem}_encoded.{extension}"

    if extension == 'smi':
        new_data = []
        with open(path) as f:
            data = f.readlines()
        mapping = {}
        for i, line in enumerate(data):
            if line.isspace():
                continue
            smi, id = line.strip().split()
            mapping[id] = i
            new_data.append(f"{smi} {i}\n")
        with open(new_path, 'w') as f:
            for line in new_data:
                f.write(line)
        return str(new_path), mapping

    elif extension == 'sdf':
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mapping = {}
        with Chem.SDWriter(str(new_path)) as w:
            for i, mol in enumerate(suppl):
                id = mol.GetProp("_Name").strip()
                mapping[id] = i
                mol.SetProp("_Name", str(i))
                w.write(mol)
        return str(new_path), mapping

    else:
        raise ValueError("The input file should be either smi or sdf")

def decode_ids(path: str, mapping: dict) -> str:
    """Decode numeric IDs back to original molecule IDs.

    .. deprecated:: 1.0
        Use :func:`Auto3D.utils.file_ops.decode_ids` instead.

    Args:
        path: Path to the input SDF file with encoded (numeric) IDs.
        mapping: Dictionary mapping original IDs to their numeric indices
                 (as returned by encode_ids).

    Returns:
        Path to the new SDF file with decoded IDs (adds '_out' suffix).
    """
    _emit_deprecation_warning("decode_ids", "Auto3D.utils.file_ops.decode_ids")
    mapping = {v: k for k, v in mapping.items()}
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    # Reconstruct base name: remove last two underscore-separated parts
    stem_parts = path_obj.stem.split('_')[:-2]
    new_stem = '_'.join(stem_parts) + '_out'
    new_path = path_obj.parent / f"{new_stem}.{extension}"

    suppl = Chem.SDMolSupplier(path, removeHs=False)
    with Chem.SDWriter(str(new_path)) as w:
        for mol in suppl:
            name = mol.GetProp("_Name").strip()
            if '@taut' in name:
                components = name.split('@taut')
                new_name = mapping[int(components[0])] + '@taut' + ''.join(components[1:])
            else:
                new_name = mapping[int(name)]
            mol.SetProp("_Name", new_name)

            id = '_'.join(mol.GetProp("ID").strip().split('_')[1:])
            new_id = new_name + '_' + id
            mol.SetProp("ID", new_id)

            w.write(mol)
    return str(new_path)