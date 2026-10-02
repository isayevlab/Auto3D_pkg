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

from collections import Counter
from collections.abc import Sequence

from rdkit import Chem

#: SD property recording whether optimization changed a molecule's configuration.
STEREO_CHANGED_PROP = "Stereo_changed"

#: Sorted tetrahedral chiral tags by atom index, then double-bond stereo by bond index.
StereoDescriptors = tuple[tuple[tuple[int, str], ...], tuple[tuple[int, str], ...]]


def _assign_phosphine_tags(work: Chem.Mol, conf_id: int = -1) -> None:
    """Tag every stereogenic trivalent phosphorus in ``work`` from its geometry.

    Modifies ``work`` in place; call it on a copy. Callers must have run
    ``AssignStereochemistryFrom3D`` first -- that call *clears* the tag on a
    degree-3 phosphorus, so running it afterwards would undo this one.

    A trivalent phosphine is a configurationally stable stereocenter: pyramidal
    inversion costs around 30 kcal/mol, against roughly 6 for the analogous
    amine, so its two epimers are separable compounds rather than conformers
    that interconvert at room temperature. RDKit perceives none of it -- neither
    ``AssignStereochemistryFrom3D`` nor ``AssignAtomChiralTagsFromStructure``
    tags a degree-3 phosphorus, and with no tag there is no label for
    ``rdCIPLabeler`` to compute either -- so the configuration has to be derived
    here (finding N-M10).

    The lone pair occupies the fourth vertex, which makes the sign of the
    signed volume of the three bond vectors a complete description of the
    configuration. Neighbors are read in **bond** order, because that is the
    reference RDKit's own ``CHI_TETRAHEDRAL_CW``/``CCW`` are defined against;
    reading them in neighbor-index order would still separate two epimers but
    could vary between conformers of one of them. Which sign means CW is not
    derivable from the enum names and was determined empirically against
    RDKit's reading of ``[P@]``/``[P@@]``; the mapping is pinned by
    ``test_phosphine_tag_matches_the_parsed_smiles_tag``.

    Scope is deliberately narrow, since a tag assigned where there is nothing
    to resolve would split one compound across two keys -- the mirror of the
    defect being fixed:

    * Phosphorus whose three neighbors do not have pairwise-distinct
      *symmetry-aware* canonical ranks (``breakTies=False``) is not stereogenic
      (``CP(C)CC``, with its two equivalent methyls) and is left alone.
    * Four-coordinate phosphorus -- P(V) such as a phosphine oxide, or a
      phosphonium -- is an ordinary tetrahedral center that
      ``AssignStereochemistryFrom3D`` already tags correctly, and a tag derived
      from only three of its four substituents would be wrong. The degree and
      hydrogen-count filters keep those out. On the H-explicit molecules
      Auto3D's own callers pass, that leaves a secondary phosphine
      ``P(H)(R)(R')`` in scope, where the bonded hydrogen is simply one of the
      three ranked neighbors; it is a genuine stereocenter of the same kind.

    Args:
        work: Molecule to tag, with a conformer. Modified in place.
        conf_id: Conformer to read. -1 (default) uses the molecule's default.
    """
    centers = [
        atom
        for atom in work.GetAtoms()
        if atom.GetAtomicNum() == 15 and atom.GetDegree() == 3 and atom.GetTotalNumHs() == 0
    ]
    if not centers:
        # The common path: one scan, no ranking and no conformer access.
        return

    # Ranked once, before any tag is written, so the outcome cannot depend on
    # the order two phosphorus centers in one molecule happen to be visited in.
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False))
    conformer = work.GetConformer(conf_id)

    for atom in centers:
        neighbors = [bond.GetOtherAtom(atom) for bond in atom.GetBonds()]
        if len({ranks[neighbor.GetIdx()] for neighbor in neighbors}) < len(neighbors):
            continue
        origin = conformer.GetAtomPosition(atom.GetIdx())
        first, second, third = (
            conformer.GetAtomPosition(neighbor.GetIdx()) - origin for neighbor in neighbors
        )
        volume = first.DotProduct(second.CrossProduct(third))
        atom.SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
            if volume < 0
            else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )


def stereo_descriptors_from_3d(mol: Chem.Mol, conf_id: int = -1) -> StereoDescriptors:
    """Perceive ``mol``'s stereochemistry from its 3D coordinates.

    Args:
        mol: Molecule with at least one conformer. Not modified.
        conf_id: Conformer to read. -1 (default) uses the molecule's default.

    Returns:
        A pair of sorted tuples: tetrahedral chiral tags keyed by atom index,
        and double-bond stereo labels keyed by bond index. Sorting makes two
        readings of the same molecule comparable with ``==``.

    Note:
        Indices are only meaningful within one molecule object. Compare two
        readings taken from the same ``mol``; never compare readings from two
        separately parsed molecules, whose atom orderings need not agree.

    Scope:
        Chiral tags, not ``_CIPCode``: the CIP property is populated only by
        RDKit's legacy stereo perception, a process-global setting (N-M9); tags
        are assigned by ``AssignStereochemistryFrom3D`` under both modes and
        flip on mirroring. Reading tags also widens coverage slightly, in the
        right direction: a ring diastereocenter (the 1,4-disubstituted
        cyclohexanes :func:`species_key` describes) carries a tag but no
        ``_CIPCode``, so its cis/trans configuration is now checked too.

        Covered, then, are atoms RDKit tags from the coordinates -- plus
        trivalent phosphorus, which it does not tag and
        :func:`_assign_phosphine_tags` supplies (N-M10) -- and bonds it
        assigns a non-``STEREONONE`` label (defined double bonds). Trivalent
        (sp3) nitrogen receives no chiral tag from
        ``AssignStereochemistryFrom3D`` under either perception mode, so an
        inverting amine nitrogen -- a real stereocenter in principle, but one
        that freely interconverts at room temperature and is not treated as
        configurational by RDKit's perception -- is never flagged. That is
        the intended trade-off: it is exactly what keeps ordinary amine
        inversion from being reported as a false positive, and it is why
        phosphorus is special-cased while nitrogen is not -- P(III) does not
        interconvert. Other stereo elements RDKit does not perceive this way
        (e.g. atropisomers) are likewise invisible to this function.

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
    _assign_phosphine_tags(work, conf_id)
    atoms = tuple(
        sorted(
            (atom.GetIdx(), str(atom.GetChiralTag()))
            for atom in work.GetAtoms()
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        )
    )
    bonds = tuple(
        sorted(
            (bond.GetIdx(), str(bond.GetStereo()))
            for bond in work.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        )
    )
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

    One stereocenter is tagged by hand, by :func:`_assign_phosphine_tags`:
    trivalent phosphorus, which RDKit's perception does not see at all. A
    phosphine's two epimers are separable compounds -- inversion costs around
    30 kcal/mol, five times the amine barrier -- but with no tag they
    canonicalize to the same SMILES, so a duplicate filter reads them as
    conformers of one species and drops the higher-energy one silently
    (finding N-M10). A tagged phosphorus writes as ``[P@]``/``[P@@]``, so the
    canonical SMILES separates them with no further work here.

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
    _assign_phosphine_tags(probe)
    return Chem.MolToSmiles(probe)


def formula_key(mol: Chem.Mol) -> str:
    """A canonical element-composition identifier for ``mol``, ignoring
    connectivity and stereochemistry.

    Two molecules share this key exactly when they have the same atoms in the
    same counts, however those atoms are bonded. That is the identity
    tautomer selection needs and :func:`species_key` cannot give it: a
    tautomer pair (e.g. keto/enol) is a constitutional isomer of its partner
    by definition, so a canonical-SMILES key never matches between them and
    would put every tautomer in its own singleton group -- which is
    indistinguishable from "nothing to rank" and defeats the point of
    tautomer selection entirely.

    Pair this with ``Chem.GetFormalCharge(mol)`` (as the tautomer-partition
    key in ``Auto3D.entry.tautomer.select_tautomers`` does) rather than
    folding charge in here: a genuine protonation-state pair (e.g. acetic
    acid vs. its conjugate base) already differs in formula -- the conjugate
    base has one fewer hydrogen -- so it splits on formula alone, while two
    true tautomers, which share both formula and charge, stay comparable.

    Args:
        mol: Any molecule. Not modified. Counts only atoms present on the
            ``Chem.Mol`` object (``GetAtoms()`` does not see implicit
            hydrogens) -- callers on molecules with implicit-only hydrogens
            would need ``Chem.AddHs(mol)`` first. Auto3D's own callers read
            explicit-H SDFs (``Chem.SDMolSupplier(sdf, removeHs=False)``), so
            this is never an issue on Auto3D's own path.

    Returns:
        A string built from a plain atom-symbol count, e.g. ``"C3H6O1"``.
        Not Hill-formula formatted and carries no charge suffix -- charge is
        a separate key component the caller adds, since a bare formula
        string cannot distinguish sign without one.
    """
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def apply_optimized_coords(mol: Chem.Mol, coords: Sequence[Sequence[float]]) -> bool:
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
