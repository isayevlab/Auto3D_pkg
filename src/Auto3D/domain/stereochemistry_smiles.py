#!/usr/bin/env python
"""Stereoisomer enumeration by SMILES string surgery (P-m29).

Split out of ``Auto3D.domain.stereochemistry``, which kept both halves of
"stereochemistry utilities" even though they share no code and differ in
kind: the other module answers questions about a molecule's *configuration*
(is this an enantiomer of that, what's its canonical key) by asking RDKit.
This module instead edits the text of a SMILES string directly -- flipping
``@``/``@@`` tokens to synthesize an enantiomer's SMILES, and counting how
many of the expected ``2**n`` stereoisomer configurations a ``.smi`` file
actually enumerated -- which is why ``get_stereo_info`` deliberately works at
the character level instead of going through RDKit's own stereo perception.

Neither module re-exports the other's names: ``Auto3D.domain.stereochemistry``
was never documented public API (it is not in ``docs/source/api.rst``), so
the split carries no re-export obligation in either direction.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict, defaultdict

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters

from Auto3D.foundation.utils.atomic_io import atomic_write_path
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.smi_io import iter_smi_records

logger = get_logger(__name__)


def no_enantiomer_helper(info1: list[str], info2: list[str]) -> bool:
    """Check if two stereo info lists represent enantiomers.

    Compares lists of stereo symbols (@ or @@) to determine if they
    represent enantiomeric configurations.

    Args:
        info1: List of stereo symbols from first SMILES.
        info2: List of stereo symbols from second SMILES.

    Returns:
        True if info1 and info2 represent enantiomers (all symbols differ),
        False otherwise.

    Raises:
        ValueError: If info1 and info2 have different lengths.
    """
    if len(info1) != len(info2):
        raise ValueError(f"Stereo info lists must have same length: {len(info1)} vs {len(info2)}")
    for i in range(len(info1)):
        if info1[i].strip() == info2[i].strip():
            return False
    return True


def get_stereo_info(smi: str) -> OrderedDict[int, str]:
    """Extract stereochemistry symbols and their positions from a SMILES string.

    Parses a SMILES string to find all @ (single) and @@ (double) stereo
    symbols and their positions.

    Args:
        smi: SMILES string to parse.

    Returns:
        OrderedDict mapping character positions to tetrahedral stereo symbols
        (@ or @@), sorted by position. Multi-letter stereo-class descriptors
        (@SP, @TH, @OH, @TB, @AL for square-planar/tetrahedral-explicit/
        octahedral/trigonal-bipyramidal/allene centers) are deliberately
        excluded: create_enantiomer can only invert plain @/@@ by string
        surgery, and treating @SP1 as a bare @ used to splice it into the
        invalid token @@SP1, silently aborting the molecule's enumeration.

    Example:
        >>> get_stereo_info('C[C@H](O)[C@@H](F)Cl')
        OrderedDict([(2, '@'), (9, '@@')])
    """
    dct: dict[int, str] = {}
    # A bare tetrahedral '@' is never immediately followed by two uppercase
    # letters, whereas every multi-letter stereo class is (SP/TH/OH/TB/AL), so
    # the negative lookahead skips those without matching legitimate '@'/'@@'.
    regex1 = re.compile("[^@]@(?![A-Z][A-Z])[^@]")
    regex2 = re.compile("@@")

    # match @
    for m in regex1.finditer(smi):
        dct[m.start() + 1] = "@"

    # match @@
    for m in regex2.finditer(smi):
        dct[m.start()] = "@@"

    dct2: OrderedDict[int, str] = OrderedDict(sorted(dct.items()))
    return dct2


def no_enantiomer(smi: str, smiles: list[str]) -> bool:
    """Check if a SMILES has no enantiomer in a list of SMILES.

    Searches through a list of SMILES to determine if any represents
    an enantiomer of the given SMILES.

    Args:
        smi: SMILES string to check.
        smiles: List of SMILES strings to search.

    Returns:
        True if no enantiomer of smi exists in smiles, False otherwise.
    """
    stereo_infoi = list(get_stereo_info(smi).values())
    for i in range(len(smiles)):
        tar = smiles[i]
        if tar != smi:
            stereo_infoj = list(get_stereo_info(tar).values())
            # A different number of stereo markers means a different set of
            # stereocenters, so the two cannot be enantiomers. Skip rather than
            # calling no_enantiomer_helper, which raises on a length mismatch --
            # that ValueError used to bubble up and make amend_configuration
            # abandon the whole molecule's enumeration.
            if len(stereo_infoi) != len(stereo_infoj):
                continue
            if no_enantiomer_helper(stereo_infoi, stereo_infoj):
                return False
    return True


def create_enantiomer(smi: str) -> str:
    """Create an enantiomer SMILES by inverting all stereo centers.

    Inverts all @ to @@ and vice versa in a SMILES string to create
    the enantiomeric structure.

    Args:
        smi: SMILES string with stereochemistry.

    Returns:
        New SMILES string with all stereo centers inverted.

    Raises:
        ValueError: If invalid stereo symbols are encountered.

    Example:
        >>> create_enantiomer('C[C@H](O)F')
        'C[C@@H](O)F'
    """
    stereo_info = get_stereo_info(smi)
    keys = list(stereo_info.keys())
    if not keys:
        # No stereo centers to invert
        return smi

    # Single pass with a cursor, one key at a time: emit smi[cursor:key]
    # verbatim, then the inverted marker, then advance the cursor past the
    # marker just consumed. The tail after the last key is emitted once,
    # after the loop, from the final cursor position.
    #
    # This replaces two copies of the same logic that used to disagree in
    # shape: a `len(keys) == 1` branch handled one stereo center, and a
    # general loop handled two or more by reading `key2`/`val2` (set inside
    # the loop's `else` branch) *after* the `for` -- correct only because
    # Python does not scope loop variables to the loop body, so `key2` still
    # held the last iteration's value. That accident breaks the moment
    # `keys` can be empty going into the tail read, which is exactly what the
    # `len(keys) == 1` branch existed to avoid (M60).
    inverted = {"@": "@@", "@@": "@"}
    new_smi = ""
    cursor = 0
    for key in keys:
        val = stereo_info[key]
        if val not in inverted:
            raise ValueError("Invalid %s" % smi)
        new_smi += smi[cursor:key]
        new_smi += inverted[val]
        cursor = key + len(val)
    new_smi += smi[cursor:]
    return new_smi


def check_value(n: float) -> bool:
    """Check if a number is a power of 2.

    Used to verify that stereoisomer enumeration produced the expected
    number of configurations. Powers of 2 from 2^-2 to 2^n are considered
    valid because not all stereo centers can always be enumerated.

    Args:
        n: Number to check.

    Returns:
        True if n is a power of 2 (within tolerance), False otherwise.

    Example:
        >>> check_value(4)
        True
        >>> check_value(3)
        False
    """
    power = math.log(n, 2)
    decimal, integer = math.modf(power)
    i = abs(power - integer)
    if i < 0.0001:
        return True
    return False


def amend_configuration(smis: str) -> dict[str, list[str]]:
    """Add missing stereoisomer configurations to a SMILES file.

    Reads a SMILES file and checks if the expected number of stereoisomers
    is present for each molecule. If configurations are missing, attempts
    to generate them by creating enantiomers.

    Args:
        smis: Path to a .smi file with format "SMILES ID" per line.

    Returns:
        Dictionary mapping molecule IDs to lists of SMILES with complete
        stereoisomer enumeration.

    Note:
        If enumeration fails, prints a warning and keeps original SMILES.

    Example:
        For input "N=C1OC(CN2CC(C)OC(C)C2)CN1", if some stereo configurations
        are missing, this function will attempt to add them.
    """
    # iter_smi_records is the single parser for this format (M59). The
    # `tuple(line.strip().split())` this replaced required EXACTLY 2 tokens
    # (a third raised "too many values to unpack") and had no blank/comment
    # handling at all; "raise" preserves the fail-fast behavior for a
    # genuinely malformed row (now a named InputValidationError instead of a
    # bare ValueError) while tolerating a trailing extra column, matching
    # every other consumer of this format.
    dct: dict[str, list[str]] = defaultdict(lambda: [])
    for _line_no, smi, mol_id in iter_smi_records(smis, on_malformed="raise"):
        # See the matching note in remove_enantiomers (domain.stereochemistry):
        # strip only the trailing isomer-index component, not everything
        # after the first underscore, so a disambiguated id like "KEY_2" is
        # not merged back onto "KEY" here (M17).
        idx = mol_id.rsplit("_", 1)[0]
        dct[idx].append(smi)

    for key in dct.keys():
        value = dct[key]
        smi = value[0]
        mol = Chem.MolFromSmiles(smi)
        num_centers = CalcNumAtomStereoCenters(mol)
        num_configurations = 2**num_centers
        num = len(value) / num_configurations

        if not check_value(num) and "@" in smi:  # Missing configurations
            try:
                new_value = []
                for val in value:
                    if no_enantiomer(val, value):
                        new_val = create_enantiomer(val)
                        new_value.append(new_val)
                value += new_value

                # Validate enumeration completeness
                new_num = len(value) / num_configurations
                if not check_value(new_num):
                    raise ValueError(
                        f"Stereo enumeration incomplete for {key}: "
                        f"expected power of 2, got {new_num}"
                    )
                dct[key] = value
            except (ValueError, KeyError, IndexError) as e:
                # ValueError: from create_enantiomer or enumeration validation
                # KeyError: from stereo_info dictionary access
                # IndexError: from list indexing in create_enantiomer
                logger.debug(f"Stereo enumeration failed for {key}: {type(e).__name__}: {e}")
                logger.warning(f"Stereo centers for {key} are not fully enumerated.")
    return dct


def amend_configuration_w(smi: str) -> None:
    """Write amended stereoisomer configurations back to a file.

    Reads a SMILES file, amends missing configurations using
    amend_configuration(), and writes the results back to the same file.

    Args:
        smi: Path to a .smi file to amend in place.

    Note:
        The rewrite is staged through a sibling temp file and moved into place
        with ``os.replace`` (atomic on POSIX and Windows) by
        :func:`Auto3D.foundation.utils.atomic_io.atomic_write_path`, which owns
        that staging for this, Auto3D's other two in-place rewrites, and the
        ranked SDF write (not a rewrite). Opening ``smi`` for writing directly
        would truncate it, so a failure partway through the loop below would
        destroy the input this function just read and leave nothing to
        recover from (C14).
    """
    dct = amend_configuration(smi)

    with atomic_write_path(smi, suffix=".smi") as tmp_path, open(tmp_path, "w") as f:
        for key in dct.keys():
            val = dct[key]
            for i, smi_str in enumerate(val):
                idx = str(key).strip() + "_" + str(i + 1)
                line = smi_str + " " + idx + "\n"
                f.write(line)
