#!/usr/bin/env python
"""File operation utilities for Auto3D.

This module provides functions for file I/O operations, including:
- SMILES file manipulation (hashing IDs, combining files)
- SDF file chunking
- ID encoding/decoding
- Temporary file housekeeping
- SDF reordering
- File type detection
- SMILES encoding for filenames
"""
from __future__ import annotations

import base64
import collections
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

from Auto3D.exceptions import InputValidationError
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


def smiles2smi(smiles: list[str], path: str) -> str:
    """Convert a list of SMILES strings to a .smi file with InChIKey IDs.

    Each SMILES string is converted to a molecule, and its InChIKey is computed
    to serve as a unique identifier. The output file contains one molecule per
    line in the format: "SMILES  InChIKey".

    Args:
        smiles: List of SMILES strings to convert.
        path: Output file path for the .smi file.

    Returns:
        The output file path.

    Example:
        >>> smiles2smi(["CCO", "CCC"], "molecules.smi")
        'molecules.smi'
        # File content:
        # CCO  LFQSCWFLJHTTHZ-UHFFFAOYSA-N
        # CCC  ATUOYWHBWRKTHZ-UHFFFAOYSA-N
    """
    lines = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise InputValidationError(
                f"Invalid SMILES at index {idx}: {smi!r} could not be parsed "
                "by RDKit."
            )
        inchikey = inchi.MolToInchiKey(mol)
        lines.append(f"{smi}  {inchikey}\n")

    with open(path, "w+") as f:
        for line in lines:
            f.write(line)

    return path


def guess_file_type(filename: str) -> str:
    """Return the file extension for a given filename.

    Determines the file type based on the extension of the provided filename.
    The extension is returned without the leading dot.

    Args:
        filename: Path or filename to analyze.

    Returns:
        The file extension without the leading dot (e.g., 'smi', 'sdf', 'xyz').

    Example:
        >>> guess_file_type("molecules.sdf")
        'sdf'
        >>> guess_file_type("/path/to/input.smi")
        'smi'
        >>> guess_file_type("file.mol2")
        'mol2'
    """
    return Path(filename).suffix[1:]


def encode_smiles(smiles: str, max_length: int = 50) -> str:
    """Encode a SMILES string for use in filenames.

    Transforms a SMILES string into a filesystem-safe string by replacing
    special characters with alphanumeric equivalents. For SMILES longer
    than max_length, uses a hash-based encoding.

    The encoding maps common SMILES characters to filename-safe alternatives:
    - '=' -> 'd' (double bond)
    - '#' -> 't' (triple bond)
    - '@' -> 'a' (stereochemistry)
    - '/' -> 's' (cis/trans)
    - '\\' -> 'b' (cis/trans)
    - '+' -> 'p' (positive charge)
    - '-' -> 'm' (negative charge)
    - '(' -> 'L' (left paren)
    - ')' -> 'R' (right paren)
    - '[' -> 'K' (left bracket)
    - ']' -> 'J' (right bracket)
    - '%' -> 'X' (ring number indicator)

    Args:
        smiles: The SMILES string to encode.
        max_length: Maximum length for encoded string before using hash.
            Defaults to 50 characters.

    Returns:
        A filesystem-safe encoded string representing the SMILES.

    Example:
        >>> encode_smiles("CCO")
        'CCO'
        >>> encode_smiles("C=C")
        'CdC'
        >>> encode_smiles("C#N")
        'CtN'
        >>> encode_smiles("[NH4+]")
        'KNH4pJ'
    """
    # Define character replacements for filesystem safety
    # Using single lowercase letters that are unlikely to cause collisions
    replacements = {
        '=': 'd',   # double bond
        '#': 't',   # triple bond
        '@': 'a',   # stereochemistry
        '/': 's',   # cis/trans
        '\\': 'b',  # cis/trans (backslash)
        '+': 'p',   # positive charge
        '-': 'm',   # negative charge (also single bond, but rare in SMILES)
        '(': 'L',   # left parenthesis
        ')': 'R',   # right parenthesis
        '[': 'K',   # left bracket
        ']': 'J',   # right bracket
        '%': 'X',   # ring number indicator (for rings > 9)
    }

    # Apply replacements
    encoded = smiles
    for char, replacement in replacements.items():
        encoded = encoded.replace(char, replacement)

    # If still too long, use a hash-based encoding
    if len(encoded) > max_length:
        # Use SHA256 hash truncated to produce a shorter, unique identifier
        hash_obj = hashlib.sha256(smiles.encode('utf-8'))
        # Take first 16 characters of base64-encoded hash (url-safe)
        hash_str = base64.urlsafe_b64encode(hash_obj.digest()[:12]).decode('utf-8')
        # Combine a prefix of the encoded SMILES with the hash
        prefix_len = max_length - len(hash_str) - 1  # -1 for separator
        if prefix_len > 0:
            encoded = f"{encoded[:prefix_len]}_{hash_str}"
        else:
            encoded = hash_str

    return encoded


def decode_smiles(encoded: str) -> str:
    """Decode an encoded SMILES string back to the original SMILES.

    Reverses the encoding performed by encode_smiles for short SMILES strings.
    Note: For hash-encoded (long) SMILES, the original cannot be recovered.

    Args:
        encoded: The encoded SMILES string.

    Returns:
        The decoded SMILES string. For hash-encoded strings, returns the
        input unchanged since the original cannot be recovered.

    Example:
        >>> decode_smiles("CdC")
        'C=C'
        >>> decode_smiles("CtN")
        'C#N'
        >>> decode_smiles("KNH4pJ")
        '[NH4+]'
    """
    # Define reverse replacements
    # Order matters: longer replacements should not interfere with shorter ones
    replacements = {
        'd': '=',   # double bond
        't': '#',   # triple bond
        'a': '@',   # stereochemistry
        's': '/',   # cis/trans
        'b': '\\',  # cis/trans (backslash)
        'p': '+',   # positive charge
        'm': '-',   # negative charge
        'L': '(',   # left parenthesis
        'R': ')',   # right parenthesis
        'K': '[',   # left bracket
        'J': ']',   # right bracket
        'X': '%',   # ring number indicator
    }

    # Check if this looks like a hash-encoded string (contains underscore near end
    # followed by base64-like characters)
    if '_' in encoded and len(encoded.split('_')[-1]) >= 12:
        # Likely hash-encoded, can't decode
        return encoded

    # Apply reverse replacements
    decoded = encoded
    for char, replacement in replacements.items():
        decoded = decoded.replace(char, replacement)

    return decoded


def hash_enumerated_smi_IDs(smi: str, out: str) -> None:
    """Write all SMILES with hashed IDs into a new file.

    Reads a SMILES file, sorts entries by ID, handles duplicate IDs by appending
    '_0' suffix, and writes the result to the output file.

    Args:
        smi: Path to the input .smi file.
        out: Path for the output .smi file with sorted/hashed IDs.

    Returns:
        None. Writes the result to the output file.

    Example:
        >>> hash_enumerated_smi_IDs("input.smi", "output.smi")
    """
    with open(smi) as f:
        data = f.readlines()

    dict0: dict[str, str] = {}
    for line in data:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Lenient parsing: ignore any extra whitespace-separated columns.
        smiles, id = parts[0], parts[1]
        while id in dict0:
            id += "_0"
        dict0[id] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    with open(out, "w+") as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + " " + id.strip() + "\n"
            f.write(molecule)


def hash_taut_smi(smi: str, out: str) -> None:
    """Write all SMILES with hashed IDs for tautomers.

    Reads a SMILES file and appends '@tautN' suffix to IDs where N is
    an incrementing counter, ensuring unique tautomer identifiers.

    Args:
        smi: Path to the input .smi file.
        out: Path for the output .smi file with tautomer IDs.

    Returns:
        None. Writes the result to the output file.

    Example:
        >>> hash_taut_smi("input.smi", "tautomers.smi")
    """
    with open(smi) as f:
        data = f.readlines()

    dict0: dict[str, str] = {}
    for line in data:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Lenient parsing: ignore any extra whitespace-separated columns.
        smiles, id = parts[0], parts[1]
        c = 1
        id_ = id
        while ("taut" not in id_) or (id_ in dict0):
            id_ = id + f"@taut{c}"
            c += 1
        dict0[id_] = smiles

    dict0 = collections.OrderedDict(sorted(dict0.items()))

    with open(out, "w+") as f:
        for id, smiles in dict0.items():
            molecule = smiles.strip() + " " + id.strip() + "\n"
            f.write(molecule)


def housekeeping_helper(folder: str, file: str) -> None:
    """Move a file into the specified folder.

    Args:
        folder: Destination folder path.
        file: Path to the file to move.

    Returns:
        None. Moves the file to the destination folder.

    Example:
        >>> housekeeping_helper("/tmp/output", "/tmp/results.sdf")
    """
    new_name = Path(folder) / Path(file).name
    shutil.move(file, str(new_name))


def housekeeping(job_name: str, folder: str, optimized_structures: str) -> None:
    """Move all metadata files into a folder.

    Moves all files from the job directory into the specified folder,
    except for the optimized structures file. Also moves any omega/flipper
    temporary files.

    Args:
        job_name: Path to the job directory containing files to move.
        folder: Destination folder for metadata files.
        optimized_structures: Path to the final output file (not moved).

    Returns:
        None. Moves files to the destination folder.

    Example:
        >>> housekeeping("/tmp/job1", "/tmp/job1/verbose", "/tmp/job1/output.sdf")
    """
    files = list(Path(job_name).glob("*"))
    for file in files:
        if str(file) != optimized_structures:
            shutil.move(str(file), folder)

    try:
        files1 = list(Path(".").glob("oeomega_*"))
        files2 = list(Path(".").glob("flipper_*"))
        files = files1 + files2
        for file in files:
            shutil.move(str(file), folder)
    except OSError:
        pass


def create_chunk_meta_names(path: str, dir: str) -> dict[str, str]:
    """Create output file names based on chunk input path and directory.

    Generates a dictionary of standardized file paths for all intermediate
    and output files used in the Auto3D workflow.

    Args:
        path: Chunk input .smi file path.
        dir: Chunk job folder path.

    Returns:
        Dictionary mapping meta names to file paths with the following keys:
        - output: Final 3D structure output file
        - optimized_og: Original optimized structures
        - output_taut: Tautomer SMILES output
        - smiles_enumerated: Enumerated SMILES file
        - smiles_reduced: Reduced enumerated SMILES file
        - smiles_hashed: Hashed enumerated SMILES file
        - enumerated_sdf: Enumerated SDF file
        - sorted_sdf: Sorted SDF file
        - housekeeping_folder: Verbose output folder
        - path: Original input path
        - dir: Job directory

    Example:
        >>> meta = create_chunk_meta_names("chunk1.smi", "/tmp/job")
        >>> meta["output"]
        '/tmp/job/chunk1_3d.sdf'
    """
    dct: dict[str, str] = {}
    dir_path = Path(dir)
    stem = Path(path).stem

    output = str(dir_path / f"{stem}_3d.sdf")
    optimized_og = str(dir_path / f"{stem}_3d0.sdf")
    output_taut = str(dir_path / "smi_taut.smi")
    smiles_enumerated = str(dir_path / "smiles_enumerated.smi")
    smiles_reduced = str(dir_path / "smiles_enumerated_reduced.smi")
    smiles_hashed = str(dir_path / "smiles_enumerated_hashed.smi")
    enumerated_sdf = str(dir_path / "smiles_enumerated.sdf")
    sorted_sdf = str(dir_path / "enumerated_sorted.sdf")
    housekeeping_folder = str(dir_path / "verbose")

    dct["output"] = output
    dct["optimized_og"] = optimized_og
    dct["output_taut"] = output_taut
    dct["smiles_enumerated"] = smiles_enumerated
    dct["smiles_reduced"] = smiles_reduced
    dct["smiles_hashed"] = smiles_hashed
    dct["enumerated_sdf"] = enumerated_sdf
    dct["sorted_sdf"] = sorted_sdf
    dct["housekeeping_folder"] = housekeeping_folder
    dct["path"] = path
    dct["dir"] = dir
    return dct


def combine_smi(smies: list[str], out: str) -> None:
    """Combine multiple SMILES files into a single file.

    Reads all input SMILES files, removes duplicates, and writes the
    combined unique entries to the output file.

    Args:
        smies: List of paths to input .smi files.
        out: Path for the combined output .smi file.

    Returns:
        None. Writes the combined result to the output file.

    Example:
        >>> combine_smi(["file1.smi", "file2.smi"], "combined.smi")
    """
    data: list[str] = []
    for smi in smies:
        with open(smi) as f:
            datai = f.readlines()
        data += datai
    data = list(set(data))
    with open(out, "w+") as f2:
        for line in data:
            if not line.isspace():
                f2.write(line.strip() + "\n")


def SDF2chunks(sdf: str) -> list[list[str]]:
    """Split an SDF file into chunks, one per molecule.

    Reads an SDF file and splits it into a list of chunks, where each chunk
    contains the lines of a single molecule as they appear in the original file.

    Args:
        sdf: Path to the input SDF file.

    Returns:
        List of chunks, where each chunk is a list of strings (lines)
        representing one molecule including the '$$$$' terminator.

    Example:
        >>> chunks = SDF2chunks("molecules.sdf")
        >>> len(chunks)  # Number of molecules
        10
        >>> chunks[0][-1].strip()  # Last line of first molecule
        '$$$$'
    """
    chunks: list[list[str]] = []
    with open(sdf) as f:
        data = f.readlines()
    chunk: list[str] = []
    for line in data:
        if line.strip() == "$$$$":
            chunk.append(line)
            chunks.append(chunk)
            chunk = []
        else:
            chunk.append(line)
    # A final record lacking the '$$$$' terminator leaves residual lines in
    # `chunk`. Preserve it as the last chunk rather than silently dropping it.
    if any(line.strip() for line in chunk):
        logger.warning(
            "SDF file %s ends without a '$$$$' terminator; "
            "keeping the trailing record as a final chunk.",
            sdf,
        )
        chunks.append(chunk)
    return chunks


def encode_ids(path: str) -> tuple[str, dict[str, int]]:
    """Encode molecule IDs to numeric indices.

    For a .smi or .sdf file, replaces all molecule IDs with sequential
    integer indices and returns a mapping from original IDs to indices.

    Args:
        path: Path to the input .smi or .sdf file.

    Returns:
        Tuple containing:
        - Path to the new file with encoded IDs (adds '_encoded' suffix)
        - Dictionary mapping original IDs to their numeric indices

    Raises:
        ValueError: If the input file is neither .smi nor .sdf format.
        InputValidationError: If a molecule has a missing/blank ID or a
            duplicate ID is encountered.

    Example:
        >>> new_path, mapping = encode_ids("molecules.smi")
        >>> mapping
        {'mol_A': 0, 'mol_B': 1, 'mol_C': 2}
    """
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    new_path = path_obj.parent / f"{path_obj.stem}_encoded.{extension}"

    if extension == "smi":
        new_data: list[str] = []
        with open(path) as f:
            data = f.readlines()
        mapping: dict[str, int] = {}
        for i, line in enumerate(data):
            if line.isspace():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                raise InputValidationError(
                    f"Line {i + 1} is missing a molecule ID "
                    f"(expected 'SMILES ID'): {line.strip()!r}"
                )
            # Lenient parsing: any extra whitespace-separated columns beyond the
            # ID are intentionally ignored (unlike a strict two-value unpack).
            smi, id = parts[0], parts[1]
            if id in mapping:
                raise InputValidationError(
                    f"Duplicate molecule ID {id!r} on line {i + 1}. "
                    "IDs must be unique."
                )
            mapping[id] = i
            new_data.append(f"{smi} {i}\n")
        with open(new_path, "w") as f:
            for line in new_data:
                f.write(line)
        return str(new_path), mapping

    elif extension == "sdf":
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mapping = {}
        with Chem.SDWriter(str(new_path)) as w:
            for i, mol in enumerate(suppl):
                if mol is None:
                    logger.warning(f"Skipping molecule at index {i}: failed to parse")
                    continue
                id = mol.GetProp("_Name").strip()
                if not id:
                    raise InputValidationError(
                        f"Molecule at index {i} has a missing or blank name."
                    )
                if id in mapping:
                    raise InputValidationError(
                        f"Duplicate molecule name {id!r} at index {i}. "
                        "Names must be unique."
                    )
                mapping[id] = i
                mol.SetProp("_Name", str(i))
                w.write(mol)
        return str(new_path), mapping

    else:
        raise ValueError("The input file should be either smi or sdf")


def decode_ids(path: str, mapping: dict[str, int]) -> str:
    """Decode numeric IDs back to original molecule IDs.

    For an SDF file with numeric IDs, restores the original IDs using
    the provided mapping dictionary.

    Args:
        path: Path to the input SDF file with encoded (numeric) IDs.
        mapping: Dictionary mapping original IDs to their numeric indices
                 (as returned by encode_ids).

    Returns:
        Path to the new SDF file with decoded IDs (adds '_out' suffix).

    Example:
        >>> mapping = {'mol_A': 0, 'mol_B': 1}
        >>> output_path = decode_ids("encoded_3d.sdf", mapping)
    """
    # Invert the mapping: index -> original_id
    inverse_mapping = {v: k for k, v in mapping.items()}
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    # Reconstruct base name: remove last two underscore-separated parts
    stem_parts = path_obj.stem.split("_")[:-2]
    new_stem = "_".join(stem_parts) + "_out"
    new_path = path_obj.parent / f"{new_stem}.{extension}"

    suppl = Chem.SDMolSupplier(path, removeHs=False)
    with Chem.SDWriter(str(new_path)) as w:
        for i, mol in enumerate(suppl):
            if mol is None:
                logger.warning(
                    "Skipping unparseable molecule at index %d in %s", i, path
                )
                continue
            name = mol.GetProp("_Name").strip()
            if "@taut" in name:
                components = name.split("@taut")
                new_name = (
                    inverse_mapping[int(components[0])] + "@taut" + "".join(components[1:])
                )
            else:
                new_name = inverse_mapping[int(name)]
            mol.SetProp("_Name", new_name)

            id = "_".join(mol.GetProp("ID").strip().split("_")[1:])
            new_id = new_name + "_" + id
            mol.SetProp("ID", new_id)

            w.write(mol)
    return str(new_path)


def reorder_sdf(sdf: str, source: str) -> list[Chem.Mol]:
    """Reorder conformers in an SDF file to match the input source file order.

    Reads the order of molecule IDs from the source file and rewrites the SDF
    file with conformers ordered to match. This ensures consistent output
    ordering regardless of processing order.

    Args:
        sdf: Path to the SDF file to reorder (will be overwritten).
        source: Path to the source .smi or .sdf file defining the desired order.

    Returns:
        List of RDKit Mol objects in the reordered sequence.

    Note:
        - For tautomer conformers (containing '@taut' in ID), the base ID
          is extracted for ordering purposes.
        - If the source format is unsupported, prints a message and returns None.

    Example:
        >>> ordered_mols = reorder_sdf("output_3d.sdf", "input.smi")
        >>> len(ordered_mols)
        10
    """
    # convert smi/sdf to a list of ids with correct order
    ids: list[str] = []
    format = guess_file_type(source)
    if format == "smi":
        with open(source) as f:
            data = f.readlines()
        for line in data:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ids.append(parts[1])
    elif format == "sdf":
        supp = Chem.SDMolSupplier(source, removeHs=False)
        for i, mol in enumerate(supp):
            if mol is None:
                logger.warning(
                    "Skipping unparseable molecule at index %d in source %s",
                    i,
                    source,
                )
                continue
            ids.append(mol.GetProp("_Name"))
    else:
        logger.warning("Unsupported file format: %s" % format)
        return None  # type: ignore

    # convert sdf to a Dict[id, List[mols]], preserving discovery order so any
    # molecule whose id is not in `source` can still be appended (no data loss).
    id_mols: dict[str, list[Chem.Mol]] = defaultdict(lambda: [])
    discovery_order: list[str] = []
    supp = Chem.SDMolSupplier(sdf, removeHs=False)
    for i, mol in enumerate(supp):
        if mol is None:
            logger.warning(
                "Skipping unparseable molecule at index %d in %s", i, sdf
            )
            continue
        id = mol.GetProp("_Name")
        if "@taut" in id:
            id = id.split("@taut")[0]
        if id not in id_mols:
            discovery_order.append(id)
        id_mols[id].append(mol)

    # Order: ids present in `source` first (in source order), then any
    # unmatched molecules appended in their original order so nothing is lost.
    source_id_set = set(ids)
    ordered_ids = list(ids)
    for id in discovery_order:
        if id not in source_id_set:
            logger.warning(
                "Molecule id %r in %s is not present in source %s; "
                "appending it at the end to avoid data loss.",
                id,
                sdf,
                source,
            )
            ordered_ids.append(id)

    # write the mols in the correct order to a temp file, then atomically
    # replace the original only on success (crash-safe in-place overwrite).
    sdf_path = Path(sdf)
    tmp_path = sdf_path.with_name(sdf_path.name + ".reorder.tmp")
    ordered_mols: list[Chem.Mol] = []
    written_ids: set[str] = set()
    try:
        with Chem.SDWriter(str(tmp_path)) as f:
            for id in ordered_ids:
                if id in written_ids:
                    continue
                written_ids.add(id)
                mols = id_mols[id]
                if len(mols) >= 1:
                    ordered_mols.extend(mols)
                    for mol in mols:
                        f.write(mol)
        os.replace(str(tmp_path), str(sdf))
    except BaseException:
        # Never leave a half-written temp file or destroy the original input.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return ordered_mols


def count_sdf(sdf: str) -> int:
    """Count the number of molecules in an SDF file.

    Args:
        sdf: Path to the SDF file.

    Returns:
        Number of molecules in the file.

    Example:
        >>> count_sdf("molecules.sdf")
        10
    """
    mols = Chem.SDMolSupplier(sdf)
    return len([mol for mol in mols if mol is not None])


def find_smiles_not_in_sdf(smi: str, sdf: str) -> list[tuple[str, str]]:
    """Find SMILES that failed to generate 3D conformers.

    Compares a SMILES input file against an SDF output file to identify
    molecules that did not successfully generate 3D structures.

    Args:
        smi: Path to input SMILES file.
        sdf: Path to output SDF file.

    Returns:
        List of (id, smiles) tuples for molecules not in SDF.

    Example:
        >>> bad = find_smiles_not_in_sdf("input.smi", "output.sdf")
        >>> for mol_id, smiles in bad:
        ...     print(f"Failed: {mol_id}")
    """
    # Find all SMILES ids
    smi_names: list[tuple[str, str]] = []
    with open(smi) as f:
        data = f.readlines()
    for line in data:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Lenient parsing: take the first two tokens, ignoring any extras.
        smiles_str, mol_id = parts[0], parts[1]
        smi_names.append((smiles_str.strip(), mol_id.strip()))

    # Get all molecule names from SDF
    sdf_data: list[str] = []
    mols = Chem.SDMolSupplier(sdf)
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning(
                "Skipping unparseable molecule at index %d in %s", i, sdf
            )
            continue
        sdf_data.append(mol.GetProp("_Name"))
    sdf_data = list(set(sdf_data))

    # Find molecules without 3D structures
    bad: list[tuple[str, str]] = []
    for smiles_str, mol_id in smi_names:
        if mol_id not in sdf_data:
            bad.append((mol_id, smiles_str))

    if len(bad) > 0:
        logger.warning("The following SMILES has no 3D structure in the SDF file.")
        logger.warning("ID, SMILES")
        for mol_id, smiles_str in bad:
            logger.warning(f"{mol_id} {smiles_str}")
    else:
        logger.info("Every SMILES has at least an 3D structure in the SDF file.")

    return bad
