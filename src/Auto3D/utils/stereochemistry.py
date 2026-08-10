#!/usr/bin/env python
"""Stereochemistry-related utility functions for Auto3D.

This module provides functions for detecting and manipulating stereochemistry
in molecular structures, including:
- Enantiomer detection and filtering
- Stereo center enumeration and validation
- Stereochemistry information extraction from SMILES
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict, defaultdict

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters

from Auto3D.utils.atomic_io import atomic_write_path
from Auto3D.utils.logging_config import get_logger
from Auto3D.utils.smi_io import iter_smi_records

logger = get_logger(__name__)

#: Stereo-specification states RDKit uses for an element the input leaves open.
#: A tetrahedral center drawn with no wedge is ``Unspecified``; a double bond
#: drawn with no geometry is ``Unspecified`` when parsed from SMILES and
#: ``Unknown`` when read from a molblock that marks the bond "either" (the
#: crossed-bond flag). Both mean "the input did not say", so both count.
_UNSPECIFIED_STEREO_STATES = (
    Chem.StereoSpecified.Unspecified,
    Chem.StereoSpecified.Unknown,
)


def count_unspecified_stereo(mol: Chem.Mol) -> int:
    """Count stereo elements the input leaves unspecified.

    Single owner of that question for every Auto3D entry point: the SMILES
    path (``check_smi_format``) and the SDF path
    (``RDKitSdfIsomer.count_unspecified_stereo``) must agree on what counts,
    or the same molecule warns on one path and passes silently on the other.

    ``Chem.FindPotentialStereo`` is used rather than
    ``CalcNumUnspecifiedAtomStereoCenters`` because the latter sees only
    **atom** centers and never double-bond geometry. Counting only atom
    centers silently misses every unspecified C=C -- e.g. ``OC(=O)C=CC(=O)O``
    embeds as a mixture of fumaric *and* maleic acid (~5 kcal/mol apart)
    under one species id, and ``CC=CC`` embeds as cis-2-butene alone with the
    trans isomer absent, both with no warning.

    Args:
        mol: Molecule to inspect. Not modified.

    Returns:
        Number of potential stereo elements (tetrahedral centers, double-bond
        geometries, ...) whose configuration the input does not fix.
    """
    return sum(
        1
        for element in Chem.FindPotentialStereo(mol)
        if element.specified in _UNSPECIFIED_STEREO_STATES
    )


def enantiomer(l1: list[tuple[int, str]], l2: list[tuple[int, str]]) -> bool:
    """Check if two lists of stereo centers represent enantiomers.

    Two molecules are enantiomers if all their stereo centers have opposite
    configurations. This function compares lists of (atom_index, stereo_type)
    tuples describing each stereo center's atom index and configuration
    label, regardless of what produced them.

    Args:
        l1: List of (atom_index, stereo_type) tuples for first molecule.
        l2: List of (atom_index, stereo_type) tuples for second molecule.

    Returns:
        True if l1 and l2 represent enantiomers (both non-empty, same indices,
        every configuration inverted), False otherwise. Two empty lists are
        False: a molecule with no stereo centers is its own mirror image, so it
        has no enantiomer to pair with.

    Raises:
        ValueError: If l1 and l2 have different lengths or mismatched indices.

    Example:
        >>> # R and S configurations at same positions indicate enantiomers
        >>> l1 = [(1, 'R'), (5, 'S')]
        >>> l2 = [(1, 'S'), (5, 'R')]
        >>> enantiomer(l1, l2)
        True
    """
    if len(l1) != len(l2):
        raise ValueError(f"Stereo center lists must have same length: {len(l1)} vs {len(l2)}")
    # Two molecules with no stereo centers at all are not an enantiomeric pair:
    # they are either the same molecule or two unrelated achiral compounds. The
    # loop below cannot express that, because an empty loop leaves `indicator`
    # at its True initial value, so the caller must be told here.
    if not l1:
        return False
    for i in range(len(l1)):
        tp1 = l1[i]
        tp2 = l2[i]
        idx1, stereo1 = tp1
        idx2, stereo2 = tp2
        if idx1 != idx2:
            raise ValueError(f"Stereo center indices must match: {idx1} vs {idx2} at position {i}")
        if stereo1 == stereo2:
            return False
    return True


def _mirror_image(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with every tetrahedral center inverted.

    Reflection through a plane inverts tetrahedral configuration and leaves
    double-bond (E/Z) geometry untouched, which is why this function only
    swaps chiral tags: a cis alkene reflects to a cis alkene.
    """
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return work


def enantiomer_key(smi: str) -> tuple[str, ...]:
    """Return a key two SMILES share iff they are one molecule or mirror images.

    The key is the sorted set of the molecule's own canonical SMILES and its
    mirror image's. An achiral molecule is its own mirror image, so its key has
    one element and cannot collide with anything but itself; a chiral molecule
    and its enantiomer produce the same two-element key from either side.

    Keying is what makes the filter linear rather than a pairwise sweep, and it
    also collapses a meso form against the string-inverted twin that
    ``amend_configuration_w`` appends for it -- the same molecule written two
    ways, which a pairwise enantiomer test cannot catch because the two are not
    an enantiomeric pair.

    Public because the SDF isomer engine deduplicates enumerated stereoisomers
    with the same key, so both input paths remove enantiomers by one rule.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # Keep unparseable input unique to its own text so it is neither merged
        # with a real molecule nor dropped. The prefix cannot occur in a SMILES.
        return ("\x00unparseable", smi)
    canonical = Chem.MolToSmiles(mol)
    return tuple(sorted({canonical, Chem.MolToSmiles(_mirror_image(mol))}))


def are_enantiomers(smi1: str, smi2: str) -> bool:
    """Check whether two SMILES are a pair of enantiomers.

    Builds the mirror image of the first molecule and compares canonical
    SMILES. This needs no atom mapping between the two inputs, which matters
    because the two SMILES are independently canonicalized and their atom
    orderings are not guaranteed to agree.

    Being index-free also makes the test exact for double-bond stereo: a
    reflection cannot change E/Z, so two molecules that differ in a C=C
    configuration never compare equal, and geometric isomers -- which are
    distinct compounds, not an enantiomeric pair -- are both retained.

    Args:
        smi1: First SMILES string.
        smi2: Second SMILES string.

    Returns:
        True only if the two are distinct molecules related by reflection.
        False for identical molecules, for unparseable input, and for any
        molecule with no tetrahedral center (it is its own mirror image).

        Only tetrahedral configuration is inverted. A molecule that combines a
        tetrahedral center with a square-planar, trigonal-bipyramidal or
        octahedral center reflects incorrectly and two such diastereomers can
        be reported as an enantiomeric pair. No supported neural network
        potential covers the elements those stereo classes require.

    Example:
        >>> are_enantiomers('C[C@H](O)F', 'C[C@@H](O)F')
        True
        >>> are_enantiomers('C/C=C/C', 'C/C=C\\\\C')
        False
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    if Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2):
        # The same molecule is not its own enantiomeric partner.
        return False
    return enantiomer_key(smi1) == enantiomer_key(smi2)


def enantiomer_helper(smiles: list[str]) -> list[str]:
    """Get non-enantiomer SMILES from a list of SMILES strings.

    Filters a list of SMILES to remove enantiomeric pairs, keeping only
    one representative from each enantiomer pair.

    Args:
        smiles: List of SMILES strings to filter.

    Returns:
        List of SMILES strings with enantiomeric duplicates removed.

    Example:
        >>> smiles = ['C[C@H](O)F', 'C[C@@H](O)F']
        >>> result = enantiomer_helper(smiles)
        >>> len(result)  # Only one enantiomer kept
        1
        >>> enantiomer_helper(['C/C=C/C', 'C/C=C\\\\C'])  # E/Z are not enantiomers
        ['C/C=C/C', 'C/C=C\\\\C']
    """
    non_enantiomers: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for smi in smiles:
        key = enantiomer_key(smi)
        if key in seen:
            continue
        seen.add(key)
        non_enantiomers.append(smi)
    return non_enantiomers


def remove_enantiomers(inpath: str, out: str) -> dict[str, list[str]]:
    """Remove enantiomers from an input SMILES file.

    Reads a SMILES file, groups SMILES by molecule ID (the id with its
    trailing ``_<isomer_index>`` component -- appended by
    ``RDKitIsomer.write_enumerated_smi`` -- removed), removes enantiomeric
    duplicates from each group, and writes the filtered results to an output
    file.

    Args:
        inpath: Path to input .smi file with format "SMILES ID" per line.
        out: Path to output .smi file for filtered results.

    Returns:
        Dictionary mapping molecule IDs to lists of non-enantiomeric SMILES.

    Note:
        If enantiomer removal fails for a molecule, all original SMILES
        are kept and a message is printed.
    """
    # iter_smi_records is the single parser for this format (M59). The loop
    # this replaced built `vals = line.split()` and indexed vals[0]/vals[1]
    # with no guard, so a blank line in `inpath` raised a bare IndexError;
    # "raise" keeps that fail-fast behavior (as an InputValidationError naming
    # the line, not a crash) for any genuinely malformed row while blank lines
    # -- which should never appear in a file this module itself wrote, but
    # previously crashed the whole function if one did -- are now skipped like
    # every other consumer of this format.
    smiles: dict[str, list[str]] = defaultdict(lambda: [])
    for _line_no, smi, mol_id in iter_smi_records(inpath, on_malformed="raise"):
        # Strip only the trailing isomer-index component write_enumerated_smi
        # appends (rsplit, maxsplit=1), not everything after the first
        # underscore: an id like "KEY_2" -- smiles2smi's disambiguation of a
        # duplicate InChIKey (utils/smi_io.py), kept distinct from "KEY"
        # specifically so it is not dropped -- must survive this grouping
        # intact, or it silently merges back onto "KEY" here before ranking
        # ever sees it (M17).
        name = mol_id.rsplit("_", 1)[0]
        smiles[name].append(smi)

    for key, values in smiles.items():
        try:
            new_values = enantiomer_helper(values)
        except (ValueError, RuntimeError, AttributeError) as e:
            # Defensive against RDKit internals: if stereo perception raises for
            # this group, keep every original SMILES rather than lose one.
            new_values = values
            logger.debug(f"Enantiomer detection failed for {key}: {type(e).__name__}: {e}")
            logger.warning(f"Enantiomers not removed for {key}")

        smiles[key] = new_values

    with open(out, "w+") as f:
        for key, val in smiles.items():
            for i in range(len(val)):
                new_key = key + "_" + str(i)
                line = val[i].strip() + " " + new_key + "\n"
                f.write(line)
    return smiles


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
        # See the matching note in remove_enantiomers: strip only the
        # trailing isomer-index component, not everything after the first
        # underscore, so a disambiguated id like "KEY_2" is not merged back
        # onto "KEY" here (M17).
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
        :func:`Auto3D.utils.atomic_io.atomic_write_path`, which owns that
        staging for all three of Auto3D's in-place rewrites. Opening ``smi``
        for writing directly would truncate it, so a failure partway through
        the loop below would destroy the input this function just read and
        leave nothing to recover from (C14).
    """
    dct = amend_configuration(smi)

    with atomic_write_path(smi, suffix=".smi") as tmp_path, open(tmp_path, "w") as f:
        for key in dct.keys():
            val = dct[key]
            for i, smi_str in enumerate(val):
                idx = str(key).strip() + "_" + str(i + 1)
                line = smi_str + " " + idx + "\n"
                f.write(line)
