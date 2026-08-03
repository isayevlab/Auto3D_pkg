"""Post-optimization stereochemistry validation.

Geometry optimization can invert a stereocenter or rotate through a double
bond, producing a molecule of different chemical identity than the one its
title names. ``check_connectivity`` compares interatomic distances against UFF
radii and is stereo-blind, so nothing else catches it.

The comparison here never crosses molecules: descriptors are read from one
molecule object immediately before and immediately after its coordinates are
overwritten, so atom and bond indices match by construction and no atom
mapping or reference SMILES is required.
"""
from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem

#: SD property recording whether optimization changed a molecule's configuration.
STEREO_CHANGED_PROP = "Stereo_changed"

#: Sorted tetrahedral CIP codes by atom index, then double-bond stereo by bond index.
StereoDescriptors = tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]


def stereo_descriptors_from_3d(
    mol: Chem.Mol, conf_id: int = -1
) -> StereoDescriptors:
    """Perceive ``mol``'s stereochemistry from its 3D coordinates.

    Args:
        mol: Molecule with at least one conformer. Not modified.
        conf_id: Conformer to read. -1 (default) uses the molecule's default.

    Returns:
        A pair of sorted tuples: tetrahedral CIP codes keyed by atom index, and
        double-bond stereo labels keyed by bond index. Sorting makes two
        readings of the same molecule comparable with ``==``.

    Note:
        Indices are only meaningful within one molecule object. Compare two
        readings taken from the same ``mol``; never compare readings from two
        separately parsed molecules, whose atom orderings need not agree.

    Scope:
        Only atoms RDKit assigns a ``_CIPCode`` (genuine tetrahedral
        stereocenters) and bonds it assigns a non-``STEREONONE`` label
        (defined double bonds) are covered. Trivalent (sp3) nitrogen never
        receives a ``_CIPCode`` from ``AssignStereochemistryFrom3D``, so an
        inverting amine nitrogen -- a real stereocenter in principle, but one
        that freely interconverts at room temperature and is not treated as
        configurational by RDKit's perception -- is never flagged. That is
        the intended trade-off: it is exactly what keeps ordinary amine
        inversion from being reported as a false positive. Other stereo
        elements RDKit does not perceive this way (e.g. atropisomers) are
        likewise invisible to this function.

        A double bond explicitly marked ``Chem.BondStereo.STEREOANY``
        (drawn with no defined geometry) is also invisible to a change:
        ``AssignStereochemistryFrom3D`` leaves an existing ``STEREOANY``
        flag untouched rather than deriving E/Z from the coordinates, so
        the "before" and "after" readings both report ``STEREOANY`` for
        that bond even if optimization rotated it from cis to trans. This
        function cannot detect that rotation; only the unspecified-stereo
        warning at enumeration time (see
        ``RDKitSdfIsomer.count_unspecified_stereo``) flags the bond at all.
    """
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work, confId=conf_id)
    atoms = tuple(sorted(
        (atom.GetIdx(), atom.GetProp("_CIPCode"))
        for atom in work.GetAtoms()
        if atom.HasProp("_CIPCode")
    ))
    bonds = tuple(sorted(
        (bond.GetIdx(), str(bond.GetStereo()))
        for bond in work.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ))
    return atoms, bonds


def species_key(mol: Chem.Mol) -> str:
    """A canonical identifier for the *compound* ``mol``'s geometry represents.

    Two molecules share this key when they are conformers of the same
    stereoisomer, and differ when they are different compounds. Both duplicate
    filters use it to answer the question their RMSD comparison cannot: whether
    the pair in front of them is a repeated conformer or two distinct species.

    Why they need it: ``ranking.species_id`` strips ``<isomer>_<conformer>``, so
    every enumerated stereoisomer of one input arrives in the same group, and
    heavy-atom ``GetBestRMS`` between two diastereomers of a 1,4-disubstituted
    ring is small -- 0.300 A measured between cis- and trans-4-tert-
    butylcyclohexanol, 0.335 A for cyclohexane-1,4-diol, both at or below the
    0.3 A default threshold. Only the duplicate energy tolerance stood between
    them and a collapse, and two ring diastereomers within 0.23 kcal/mol are
    ordinary. When it fired, one of two distinct compounds left the output with
    nothing logged.

    Stereochemistry is perceived from the coordinates rather than read from the
    molecule's tags, because the question is about the geometry in front of us: a
    record from an SDF Auto3D did not write may carry no tags at all, and a stale
    tag would answer for a structure that no longer exists. Perception runs on a
    copy, so the caller's molecule is untouched, and on the H-explicit form,
    because a stereocenter whose fourth substituent is a hydrogen cannot be
    perceived once the hydrogens are gone.

    Contrast :func:`stereo_descriptors_from_3d`, which answers a different
    question: it keys descriptors by atom index and is therefore only comparable
    between two readings of *one* molecule object. This returns a canonical
    isomeric SMILES, which is comparable across separately parsed molecules --
    the case both filters actually have.

    Args:
        mol: Molecule with at least one conformer. Not modified.

    Returns:
        Canonical isomeric SMILES with explicit hydrogens retained. Hydrogens
        cannot change the comparison, since any pair reaching a duplicate check
        carries the same atoms, and keeping them avoids a second ``RemoveHs``
        per molecule on top of the one the RMSD comparison already needs.
    """
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe)
    return Chem.MolToSmiles(probe)


def apply_optimized_coords(
    mol: Chem.Mol, coords: Sequence[Sequence[float]]
) -> bool:
    """Write optimized coordinates into ``mol`` and record any stereo change.

    Reads the molecule's configuration from its current (pre-optimization)
    coordinates, overwrites the conformer with ``coords``, reads it again, and
    stores the comparison on the ``Stereo_changed`` property so the conformer
    filters can act on it after an SDF round trip.

    Args:
        mol: Molecule holding the pre-optimization conformer. Modified in place.
        coords: One (x, y, z) position per atom, in atom order.

    Returns:
        True if the configuration is unchanged, False if it changed.
    """
    before = stereo_descriptors_from_3d(mol)
    conformer = mol.GetConformer()
    for atom_idx in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, coords[atom_idx])
    preserved = stereo_descriptors_from_3d(mol) == before
    mol.SetProp(STEREO_CHANGED_PROP, str(not preserved))
    return preserved


def stereo_preserved(mol: Chem.Mol) -> bool:
    """True unless ``mol`` is marked as having changed configuration.

    Molecules from paths that never run the post-optimization check carry no
    marker and are treated as preserved, so this predicate can be added beside
    ``check_connectivity`` without dropping records from other entry points.
    """
    try:
        return mol.GetProp(STEREO_CHANGED_PROP).lower() != "true"
    except KeyError:
        return True
