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

from Auto3D.exceptions import ConfigurationError, InputValidationError
from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)


def iter_smi_records(path, *, on_malformed="skip"):
    """Yield (line_no, smiles, mol_id) for each non-blank, non-comment line of
    a .smi file.

    A line is 'SMILES ID [extra columns ignored]'. Blank/whitespace-only lines
    are skipped, as are lines whose first non-whitespace character is '#'
    (comments) -- matching cli.commands.validate.validate_smiles_file, so
    `auto3d validate` and every consumer of this function (encode_ids and so
    the whole run pipeline, plus the isomer/tautomer engines and the
    input/output reconciliation helpers) agree on what a comment line is
    (M25). A real SMILES token can never start with '#' (it is a bond symbol
    between two atoms, never a leading character), so this cannot misclassify
    a legitimate SMILES as a comment. on_malformed controls lines with fewer
    than 2 whitespace tokens:
      - "skip": log a warning and skip the line (lenient; default)
      - "raise": raise InputValidationError naming the line

    Args:
        path: Path to the input .smi file.
        on_malformed: How to handle lines with fewer than 2 tokens
            ("skip" or "raise").

    Yields:
        Tuples of (line_no, smiles, mol_id) where line_no is 1-based. Any extra
        whitespace-separated columns beyond the ID are intentionally ignored.

    Raises:
        InputValidationError: If on_malformed == "raise" and a non-blank,
            non-comment line has fewer than 2 whitespace tokens.
        ValueError: If on_malformed is not "skip" or "raise".
    """
    if on_malformed not in ("skip", "raise"):
        raise ValueError(
            f"on_malformed must be 'skip' or 'raise', got: {on_malformed!r}"
        )
    with open(path) as f:
        data = f.readlines()
    for line_no, line in enumerate(data, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            if on_malformed == "raise":
                raise InputValidationError(
                    f"Line {line_no} is missing a molecule ID "
                    f"(expected 'SMILES ID'): {line.strip()!r}"
                )
            logger.warning(
                f"Skipping molecule at line {line_no}: failed to parse "
                f"(need 'SMILES ID', got: {line.strip()!r})"
            )
            continue
        # Lenient parsing: ignore any extra whitespace-separated columns.
        yield line_no, parts[0], parts[1]


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
    seen_ids: dict[str, int] = {}
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise InputValidationError(
                f"Invalid SMILES at index {idx}: {smi!r} could not be parsed "
                "by RDKit."
            )
        inchikey = inchi.MolToInchiKey(mol)
        # Distinct inputs can share a standard InChIKey (e.g. tautomers the
        # standard InChIKey conflates, or the same molecule written two ways).
        # The InChIKey is used as the molecule's unique ID downstream, and
        # reorder_sdf collapses duplicate IDs -- so a colliding input would be
        # silently dropped. Disambiguate by suffixing repeats (_2, _3, ...) so
        # every input keeps its own conformers. The suffix stays a single
        # whitespace-delimited token and round-trips through enumeration.
        count = seen_ids.get(inchikey, 0) + 1
        seen_ids[inchikey] = count
        mol_id = inchikey if count == 1 else f"{inchikey}_{count}"
        if count > 1:
            logger.info(
                "Input SMILES %r shares InChIKey %s with an earlier input; "
                "assigning disambiguated id %s so it is not dropped.",
                smi,
                inchikey,
                mol_id,
            )
        lines.append(f"{smi}  {mol_id}\n")

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
    dict0: dict[str, str] = {}
    for _line_no, smiles, id in iter_smi_records(smi, on_malformed="skip"):
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
    dict0: dict[str, str] = {}
    for _line_no, smiles, id in iter_smi_records(smi, on_malformed="skip"):
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

    # Sweep OpenEye omega/flipper logfiles the binaries drop in the CWD. Guard
    # each move individually: with multi-GPU optimizers running concurrently a
    # peer may move/remove a file first, and a single bare try used to abandon
    # the rest of the sweep on the first such error. (Diagnostic logs only.)
    for file in list(Path(".").glob("oeomega_*")) + list(Path(".").glob("flipper_*")):
        try:
            if file.exists():
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
    # Order-preserving dedup: list(set(...)) randomizes line order across runs
    # (hash seed), making the combined output non-deterministic. dict.fromkeys
    # keeps first-seen order while removing exact duplicates.
    data = list(dict.fromkeys(data))
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


def encode_ids(
    path: str, out_dir: str | os.PathLike[str] | None = None
) -> tuple[str, dict[str, int]]:
    """Encode molecule IDs to numeric indices.

    For a .smi or .sdf file, replaces all molecule IDs with sequential
    integer indices and returns a mapping from original IDs to indices.

    The encoded file is named ``<stem>_encoded.<ext>``. That name is derived
    from the input, so it can collide with a file the user already owns:
    ``mols_encoded.smi`` sitting beside ``mols.smi`` is a perfectly ordinary
    thing for a user to have, and this function used to overwrite it without
    a word (``WorkflowOrchestrator`` then ``unlink()``ed it at the end of the
    run, so the file was destroyed twice over). Two things prevent that now:
    ``out_dir`` lets the caller redirect the encoded file somewhere it owns
    -- ``WorkflowOrchestrator`` passes its freshly created job directory --
    and the collision check below refuses to write over an existing file for
    every caller, including ones that take the default location.

    Args:
        path: Path to the input .smi or .sdf file.
        out_dir: Directory to write the encoded file into. Defaults to the
            input file's own directory.

    Returns:
        Tuple containing:
        - Path to the new file with encoded IDs (adds '_encoded' suffix)
        - Dictionary mapping original IDs to their numeric indices

    Raises:
        ValueError: If the input file is neither .smi nor .sdf format.
        ConfigurationError: If a file already exists at the encoded path.
        InputValidationError: If a molecule has a missing/blank ID or a
            duplicate ID is encountered.

    Example:
        >>> new_path, mapping = encode_ids("molecules.smi")
        >>> mapping
        {'mol_A': 0, 'mol_B': 1, 'mol_C': 2}
    """
    path_obj = Path(path).resolve()
    extension = path_obj.suffix[1:]
    # Checked up front rather than in a trailing `else`: the collision check
    # below must not be the thing that reports an unsupported extension.
    if extension not in ("smi", "sdf"):
        raise ValueError("The input file should be either smi or sdf")

    directory = Path(out_dir) if out_dir is not None else path_obj.parent
    new_path = directory / f"{path_obj.stem}_encoded.{extension}"
    if new_path.exists():
        raise ConfigurationError(
            f"encode_ids would overwrite the existing file {new_path}. "
            "Auto3D writes its encoded copy of the input there; move or "
            "rename that file, or pass out_dir to write the encoded copy "
            "somewhere else."
        )

    if extension == "smi":
        new_data: list[str] = []
        mapping: dict[str, int] = {}
        # iter_smi_records raises InputValidationError on a <2-token line
        # (on_malformed="raise"). Duplicate-id detection stays here because the
        # helper does not dedup. Index by a dense record counter, not the file
        # line number: blank/skipped lines would otherwise leave gaps in the
        # index space, which is inconsistent with the dense positions the chunk
        # manager assumes downstream. The original file line_no is still used in
        # the error message so it points at the real offending line.
        for i, (line_no, smi, id) in enumerate(
            iter_smi_records(path, on_malformed="raise")
        ):
            if id in mapping:
                raise InputValidationError(
                    f"Duplicate molecule ID {id!r} on line {line_no}. "
                    "IDs must be unique."
                )
            mapping[id] = i
            new_data.append(f"{smi} {i}\n")
        with open(new_path, "w") as f:
            for line in new_data:
                f.write(line)
        return str(new_path), mapping

    else:  # "sdf" -- the only remaining possibility, checked above
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
                logger.warning("Skipping molecule at index %d: failed to parse", i)
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
        - Molecules whose id is not present in ``source`` are appended at the
          end (not dropped), so no data is lost.
        - Duplicate source ids are de-duplicated: each id's molecules are
          written once, so the returned list may be shorter than the input if
          source ids repeat.

    Example:
        >>> ordered_mols = reorder_sdf("output_3d.sdf", "input.smi")
        >>> len(ordered_mols)
        10
    """
    # convert smi/sdf to a list of ids with correct order
    ids: list[str] = []
    format = guess_file_type(source)
    if format == "smi":
        for _line_no, _smiles, mol_id in iter_smi_records(source, on_malformed="skip"):
            ids.append(mol_id)
    elif format == "sdf":
        supp = Chem.SDMolSupplier(source, removeHs=False)
        for i, mol in enumerate(supp):
            if mol is None:
                logger.warning("Skipping molecule at index %d: failed to parse", i)
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
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        id = mol.GetProp("_Name")
        if "@taut" in id:
            id = id.split("@taut")[0]
        if id not in id_mols:
            discovery_order.append(id)
        id_mols[id].append(mol)

    # Release the RDKit supplier's file handle before overwriting `sdf`.
    # On Windows an open handle makes the later os.replace() fail with
    # "Access is denied" (WinError 5); on POSIX the replace would succeed.
    del supp

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
    for _line_no, smiles_str, mol_id in iter_smi_records(smi, on_malformed="skip"):
        smi_names.append((smiles_str, mol_id))

    # Get all molecule names from SDF
    sdf_data: list[str] = []
    mols = Chem.SDMolSupplier(sdf)
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        name = mol.GetProp("_Name")
        # decode_ids keeps a "@tautN" suffix on tautomer-enumerated conformers
        # (see decode_ids), but the .smi input has only the base id. Strip it
        # the same way reorder_sdf/count_output do, or every tautomer-derived
        # molecule would be misreported as missing.
        if "@taut" in name:
            name = name.split("@taut")[0]
        sdf_data.append(name)
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


def find_ids_not_in_sdf(source_sdf: str, sdf: str) -> list[str]:
    """Find molecule IDs from an SDF input that have no 3D structure in the output SDF.

    The SDF-input counterpart to :func:`find_smiles_not_in_sdf`. That function
    reads its expected-IDs list from a ``.smi`` file, which does not exist when
    the pipeline's input is itself an SDF file; this reads the same expected-IDs
    list from the source SDF's ``_Name`` property instead, so SDF-input runs get
    the same input/output reconciliation SMILES-input runs do.

    Args:
        source_sdf: Path to the original input SDF file (pre-encoding IDs).
        sdf: Path to the output SDF file (decoded IDs).

    Returns:
        List of input molecule IDs with no corresponding structure in ``sdf``.

    Example:
        >>> missing = find_ids_not_in_sdf("input.sdf", "output.sdf")
        >>> for mol_id in missing:
        ...     print(f"Failed: {mol_id}")
    """
    # Find all input molecule IDs
    source_ids: list[str] = []
    suppl = Chem.SDMolSupplier(source_sdf, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        source_ids.append(mol.GetProp("_Name").strip())

    # Get all molecule names from the output SDF
    sdf_ids: set[str] = set()
    mols = Chem.SDMolSupplier(sdf)
    for i, mol in enumerate(mols):
        if mol is None:
            logger.warning("Skipping molecule at index %d: failed to parse", i)
            continue
        name = mol.GetProp("_Name")
        if "@taut" in name:
            name = name.split("@taut")[0]
        sdf_ids.add(name)

    # Find molecules without 3D structures, preserving source order and
    # de-duplicating (an id can appear once per tautomer/isomer conformer
    # group in some callers, though not in the raw source SDF).
    bad: list[str] = []
    seen: set[str] = set()
    for mol_id in source_ids:
        if mol_id not in sdf_ids and mol_id not in seen:
            bad.append(mol_id)
            seen.add(mol_id)

    if bad:
        logger.warning("The following input IDs have no 3D structure in the SDF file.")
        for mol_id in bad:
            logger.warning(mol_id)
    else:
        logger.info("Every input molecule has at least one 3D structure in the SDF file.")

    return bad
