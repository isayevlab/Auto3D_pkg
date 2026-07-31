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
