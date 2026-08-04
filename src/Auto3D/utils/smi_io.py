#!/usr/bin/env python
"""Reading and writing ``.smi`` files.

The one place that knows what a line of an Auto3D ``.smi`` file looks like:
``SMILES ID`` with any further whitespace-separated columns ignored, blank
lines skipped and ``#`` comment lines skipped. Everything that consumes or
produces that format goes through :func:`iter_smi_records` or one of the
writers here, so ``auto3d validate``, the run pipeline, the isomer/tautomer
engines and the input/output reconciliation cannot drift apart on it.

Deliberately free of ``torch``: these writers are reached from
``Auto3D.utils`` leaves and from the ID-encoding step, and pulling the model
tree in through a validation import would make an ordinary ``.smi`` write cost
the whole ``Auto3D.models`` package. The overwrite guards live in
``utils/output_guard.py`` for the same reason.
"""
from __future__ import annotations

import collections

from rdkit import Chem
from rdkit.Chem import inchi

from Auto3D.exceptions import InputValidationError
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
