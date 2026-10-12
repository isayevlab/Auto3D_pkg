#!/usr/bin/env python
"""Stereochemistry-related utility functions for Auto3D.

This module answers questions about a molecule's *configuration* by asking
RDKit: enantiomer detection and filtering, and stereo center enumeration.
``Auto3D.domain.stereochemistry_smiles`` instead enumerates stereoisomers by
editing a SMILES string's ``@``/``@@`` tokens directly; the two modules
share no code.
"""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem

from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.smi_io import iter_smi_records

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
    # iter_smi_records is the single parser for this format. The loop
    # this replaced built `vals = line.split()` and indexed vals[0]/vals[1]
    # with no guard, so a blank line in `inpath` raised a bare IndexError;
    # "raise" keeps that fail-fast behavior (as an InputValidationError naming
    # the line, not a crash) for any genuinely malformed row while blank lines
    # -- which should never appear in a file this module itself wrote -- are
    # skipped like every other consumer of this format.
    smiles: dict[str, list[str]] = defaultdict(lambda: [])
    for _line_no, smi, mol_id in iter_smi_records(inpath, on_malformed="raise"):
        # Strip only the trailing isomer-index component write_enumerated_smi
        # appends (rsplit, maxsplit=1), not everything after the first
        # underscore: an id like "KEY_2" -- smiles2smi's disambiguation of a
        # duplicate InChIKey (utils/smi_io.py), kept distinct from "KEY"
        # specifically so it is not dropped -- must survive this grouping
        # intact, or it silently merges back onto "KEY" here before ranking
        # ever sees it.
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
