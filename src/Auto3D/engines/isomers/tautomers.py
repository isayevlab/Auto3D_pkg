"""Tautomer enumeration: the one engine that spans both backends.

``RDKitOrOEChemTautomerEngine`` dispatches on ``mode`` (``rdkit``/``oechem``)
inside a single class, which is why this module -- alone among the backend
modules beside it -- carries the OpenEye import guard as well as rdkit. Splitting
the class in two along that seam is a separate change with its own compatibility
question; this move does not make it.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

from Auto3D.domain.stereochemistry import count_unspecified_stereo, enantiomer_key
from Auto3D.foundation.utils.logging_config import get_logger
from Auto3D.foundation.utils.smi_io import (
    combine_smi,
    iter_smi_records,
)

try:
    from openeye import oechem, oequacpac
except ImportError:
    pass

logger = get_logger(__name__)


def _contradicts_reference_stereo(reference: Chem.Mol, tautomer: Chem.Mol) -> bool:
    """True if ``tautomer`` is the input's skeleton with a different configuration.

    A tautomer that comes back with the input's own constitution and a
    different configuration is the input written wrong, whatever route the
    enumerator took to it, so it is dropped outright rather than relaxed to an
    unspecified element.

    The test is deliberately narrow. A tautomer whose constitution differs from
    the input is a genuinely different species, and its stereo descriptors are
    not comparable to the input's -- a keto/enol shift can relabel an untouched
    center from R to S purely by changing a neighboring branch's CIP priority,
    which is a relabeling and not an inversion. Only when the constitution is
    identical does "different configuration" mean the molecule came back wrong.
    The same reasoning keeps :func:`_merge_same_constitution` from comparing
    records across constitutions.

    Comparing canonical SMILES rather than per-atom descriptors also means this
    depends on no assumption about the enumerator preserving atom ordering.
    """
    if Chem.MolToSmiles(tautomer, isomericSmiles=False) != Chem.MolToSmiles(
        reference, isomericSmiles=False
    ):
        return False
    return Chem.MolToSmiles(tautomer) != Chem.MolToSmiles(reference)


def _local_signatures(mol: Chem.Mol) -> list[tuple]:
    """Per atom: (symbol, total H count, sorted (neighbor index, bond order)).

    The signature differs exactly where a proton shift touched the atom: the
    hydrogen left or arrived, or the pi system moved onto it.
    """
    return [
        (
            atom.GetSymbol(),
            atom.GetTotalNumHs(),
            tuple(
                sorted(
                    (
                        nb.GetIdx(),
                        mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble(),
                    )
                    for nb in atom.GetNeighbors()
                )
            ),
        )
        for atom in mol.GetAtoms()
    ]


def _components(mol: Chem.Mol, atoms: set[int]) -> list[set[int]]:
    """Connected components of ``atoms`` in the subgraph of ``mol`` induced on them."""
    remaining, components = set(atoms), []
    while remaining:
        start = remaining.pop()
        component, frontier = {start}, [start]
        while frontier:
            current = frontier.pop()
            for nb in mol.GetAtomWithIdx(current).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining:
                    remaining.discard(j)
                    component.add(j)
                    frontier.append(j)
        components.append(component)
    return components


def _connected_without(
    mol: Chem.Mol, source: set[int], target: set[int], allowed: set[int]
) -> bool:
    """True if some route from ``source`` to ``target`` stays inside ``allowed``."""
    seen, frontier = set(source), list(source)
    while frontier:
        current = frontier.pop()
        if current in target:
            return True
        for nb in mol.GetAtomWithIdx(current).GetNeighbors():
            j = nb.GetIdx()
            if j in allowed and j not in seen:
                seen.add(j)
                frontier.append(j)
    return False


def _separates_shifted_regions(
    mol: Chem.Mol, pivot: set[int], delta: set[int], core: set[int]
) -> bool:
    """A core element not itself shifted was planarized on the way when the
    shifted regions lie on two or more sides of it: every route inside the
    tautomer core between two components of ``delta`` passes through ``pivot``."""
    components = _components(mol, delta)
    if len(components) < 2:
        return False
    allowed = (core | delta) - pivot
    for i, a in enumerate(components):
        for b in components[i + 1 :]:
            if not _connected_without(mol, a - pivot, b - pivot, allowed):
                return True
    return False


def _destroyed_stereo(
    reference: Chem.Mol, tautomer: Chem.Mol, core: set[int]
) -> tuple[set[int], set[int]]:
    """The tagged atoms and stereo bonds of ``tautomer`` whose configuration the
    route did not preserve.

    ``reference`` and ``tautomer`` share the enumerator's atom indexing, and
    ``core`` is the tautomer core the enumerator reports. An element outside
    the core is untouchable by any transform and keeps what the input gave it;
    an element inside the core loses it when its own bonding changed or when
    the shifted regions surround it.
    """
    ref_sig, taut_sig = _local_signatures(reference), _local_signatures(tautomer)
    delta = {i for i, (r, t) in enumerate(zip(ref_sig, taut_sig, strict=True)) if r != t}
    atoms, bonds = set(), set()
    for atom in tautomer.GetAtoms():
        i = atom.GetIdx()
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED or i not in core:
            continue
        if i in delta or _separates_shifted_regions(tautomer, {i}, delta, core):
            atoms.add(i)
    for bond in tautomer.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ref_bond = reference.GetBondBetweenAtoms(a, b)
        own_bonding_changed = (
            ref_bond is None
            or ref_bond.GetBondType() != bond.GetBondType()
            or a in delta
            or b in delta
        )
        surrounded = {a, b} <= core and _separates_shifted_regions(tautomer, {a, b}, delta, core)
        if own_bonding_changed or surrounded:
            bonds.add(bond.GetIdx())
    return atoms, bonds


def _clear_stereo(mol: Chem.Mol, atoms: set[int], bonds: set[int]) -> Chem.Mol:
    """Clear by setter and re-perceive.

    Never branch on the stereo spelling: ``STEREOE``/``STEREOZ`` and
    ``STEREOCIS``/``STEREOTRANS`` are the same geometry under RDKit's two
    perception modes.
    """
    work = Chem.Mol(mol)
    for i in atoms:
        work.GetAtomWithIdx(i).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in bonds:
        bond = work.GetBondWithIdx(b)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            for nb_bond in atom.GetBonds():
                nb_bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    return work


def _stereo_fingerprint(mol: Chem.Mol) -> Counter:
    """Multiset of (symmetry class on the stereo-stripped graph, CIP descriptor)
    over the specified elements.

    ``breakTies=False`` gives symmetry classes rather than a total order, so
    two records of one constitution that were parsed from different isomeric
    SMILES still key the same element the same way; a multiset rather than a
    set so that a record specifying two symmetry-equivalent centers is not
    flattened onto one that specifies a single center.
    """
    stripped = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stripped)
    ranks = list(Chem.CanonicalRankAtoms(stripped, breakTies=False))
    labeled = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(labeled)
    fingerprint: Counter = Counter()
    for atom in labeled.GetAtoms():
        if atom.HasProp("_CIPCode"):
            fingerprint[(ranks[atom.GetIdx()], atom.GetProp("_CIPCode"))] += 1
    for bond in labeled.GetBonds():
        if bond.HasProp("_CIPCode"):
            key = tuple(sorted((ranks[bond.GetBeginAtomIdx()], ranks[bond.GetEndAtomIdx()])))
            fingerprint[(key, bond.GetProp("_CIPCode"))] += 1
    return fingerprint


def _merge_same_constitution(records: list[tuple[Chem.Mol, str]]) -> list[tuple[Chem.Mol, str]]:
    """Within one input, records of one constitution that differ only in how much
    they specify collapse onto the most specified one; two records that specify
    one element with different descriptors emit that element unspecified.

    Fingerprints are multisets of (symmetry class, CIP descriptor), so a
    record is compared with another by what it says, not by atom indices,
    which differ between two records of one constitution read from opposite
    ends of a chain.
    """
    groups: dict[tuple[str, str], list[Chem.Mol]] = defaultdict(list)
    for mol, idx in records:
        groups[(idx, Chem.MolToSmiles(mol, isomericSmiles=False))].append(mol)
    merged: list[tuple[Chem.Mol, str]] = []
    for (idx, _constitution), mols in groups.items():
        fingerprints = [_stereo_fingerprint(m) for m in mols]
        # A record that says strictly less than another record of the same
        # constitution adds nothing.
        kept = [
            (m, fp)
            for i, (m, fp) in enumerate(zip(mols, fingerprints, strict=True))
            if not any(
                j != i and fp != other and fp <= other for j, other in enumerate(fingerprints)
            )
        ]
        # Among records that specify the same elements, a disagreement on one
        # element's descriptor means neither assignment is justified.
        by_elements: dict[tuple, list[tuple[Chem.Mol, Counter]]] = defaultdict(list)
        for m, fp in kept:
            elements = Counter(cls for cls, _ in fp.elements())
            by_elements[tuple(sorted(elements.items(), key=repr))].append((m, fp))
        for members in by_elements.values():
            first, first_fp = members[0]
            if all(fp == first_fp for _, fp in members):
                merged.append((first, idx))
                continue
            descriptors: dict = defaultdict(set)
            for _, fp in members:
                for cls, code in fp:
                    descriptors[cls].add(code)
            disagreeing = {cls for cls, codes in descriptors.items() if len(codes) > 1}
            stripped = Chem.Mol(first)
            Chem.RemoveStereochemistry(stripped)
            ranks = list(Chem.CanonicalRankAtoms(stripped, breakTies=False))
            atoms = {a.GetIdx() for a in first.GetAtoms() if ranks[a.GetIdx()] in disagreeing}
            bonds = {
                b.GetIdx()
                for b in first.GetBonds()
                if tuple(sorted((ranks[b.GetBeginAtomIdx()], ranks[b.GetEndAtomIdx()])))
                in disagreeing
            }
            logger.info(
                "%r: two tautomer records of one constitution disagree on the descriptor of %d "
                "stereo element(s) (atom(s) %s); the element is emitted unspecified.",
                idx,
                len(atoms) + len(bonds),
                sorted(atoms),
            )
            merged.append((_clear_stereo(first, atoms, bonds), idx))
    return merged


def _expand_cleared(mol: Chem.Mol) -> list[Chem.Mol]:
    """Give each configuration of a cleared element its own record, mirror pairs
    collapsed. Used when stereoisomer enumeration is switched off downstream."""
    options = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(mol, options=options))
    by_key: dict[tuple, Chem.Mol] = {}
    for isomer in isomers:
        by_key.setdefault(enantiomer_key(Chem.MolToSmiles(isomer)), isomer)
    return list(by_key.values())


class RDKitOrOEChemTautomerEngine:
    """Enumerate possible tautomers for input molecules, on either backend.

    Named for its backends, like ``RDKitIsomer``/``RDKitSdfIsomer`` beside it,
    and *not* ``TautomerEngine``: that name belongs to the role Protocol in
    :mod:`Auto3D.engines.isomers.base`, which this class structurally satisfies. The two
    shared the name until 3.0.0, which forced :mod:`Auto3D.engines.isomers.factory` to
    import this one under a local alias to keep them apart within a single file
    -- a per-file workaround for a package-wide collision, and one whose alias
    said "Omega" although tautomers come from ``oequacpac`` and Omega is the
    conformer generator. ``run()`` dispatches on ``mode``, so one class covers
    both backends.

    On the rdkit backend the output carries the input's stereo only where the
    proton shifts justify it: an element outside the tautomer core, or one
    whose own bonding is unchanged and that the shifted regions do not
    surround, keeps the configuration the input gave it, and every other
    element is emitted unspecified for the stereoisomer step to enumerate.
    Tetrahedral centers and double-bond geometry are covered; axial and planar
    chirality and the other non-tetrahedral classes are not perceived here and
    are emitted as the enumerator produces them.

    Args:
        mode: Tautomer engine to use: 'rdkit' or 'oechem'.
        input_f: Path to input SMI file.
        out: Path for output SMI file.
        pKaNorm: Normalize ionization state to pH ~7.4 (oechem only).
        expand_cleared_stereo: Write a tautomer whose stereo this step left
            unspecified once per configuration, mirror pairs collapsed, rather
            than once with the element unspecified. For the caller that has
            switched stereoisomer enumeration off downstream, where an
            unspecified element would otherwise reach embedding and be fixed
            per conformer. rdkit only.
    """

    def __init__(
        self,
        mode: str,
        input_f: str,
        out: str,
        pKaNorm: bool,
        *,
        expand_cleared_stereo: bool = False,
    ) -> None:
        self.mode = mode
        self.input_f = input_f
        self.output = out
        self.pKaNorm = pKaNorm
        self.expand_cleared_stereo = expand_cleared_stereo

    def oe_taut(self) -> None:
        """Enumerate tautomers using OEChem."""
        ifs = oechem.oemolistream()
        ifs.open(self.input_f)

        ofs = oechem.oemolostream()
        ofs.open(self.output)

        tautomerOptions = oequacpac.OETautomerOptions()

        for mol in ifs.GetOEGraphMols():
            for tautomer in oequacpac.OEGetReasonableTautomers(mol, tautomerOptions, self.pKaNorm):
                oechem.OEWriteMolecule(ofs, tautomer)

        # Appending input_f smiles into output
        combine_smi([self.input_f, self.output], self.output)

    def rd_taut(self) -> None:
        """Enumerate tautomers using RDKit, keeping only justified stereo.

        Each emitted tautomer keeps a stereo element when the element lies
        outside the tautomer core, or when its own bonding is unchanged and the
        shifted regions do not surround it; every other element is cleared and
        re-perceived, so the record states that the configuration is open
        rather than asserting the input's. A tautomer with the input's
        constitution and a different configuration is dropped. Records of one
        constitution that differ only in how much they specify are then merged
        onto the most specified one, and two that disagree on one element's
        descriptor emit that element unspecified.

        With ``expand_cleared_stereo`` the surviving unspecified elements are
        written out one record per configuration, mirror pairs collapsed,
        because the stereoisomer step that would otherwise own them is off.
        """
        enumerator = rdMolStandardize.TautomerEnumerator()
        # RDKit's default strips stereo from every atom in the tautomer core,
        # in every output tautomer, including tautomers formed at a site that
        # cannot reach a given center -- enolizing a ketone's other alpha
        # carbon, say. The stripped SMILES are then re-enumerated downstream by
        # EnumerateStereoisomers(onlyUnassigned=True) and one epimer is kept
        # arbitrarily, so a submitted (S) molecule comes back as (R) half the
        # time at identical energy. Disabling the default removal keeps what
        # the user specified; the enumerator then over-keeps in the other
        # direction, copying the input's local arrangement back onto a center
        # the route planarized, so _destroyed_stereo() below decides per
        # element which of the two the route justifies.
        enumerator.SetRemoveSp3Stereo(False)
        enumerator.SetRemoveBondStereo(False)
        smiles = []
        for _line_no, smi, idx in iter_smi_records(self.input_f, on_malformed="skip"):
            smiles.append((smi, idx))
        tautomers: list[tuple[Chem.Mol, str]] = []
        unspecified_in_input: dict[str, int] = {}
        for smi, idx in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Skipping molecule {idx!r}: failed to parse {smi!r}")
                continue
            unspecified_in_input[idx] = count_unspecified_stereo(mol)
            result = enumerator.Enumerate(mol)
            core = set(result.modifiedAtoms)
            cleared_total: set[int] = set()
            for taut in result:
                atoms, bonds = _destroyed_stereo(mol, taut, core)
                if atoms or bonds:
                    taut = _clear_stereo(taut, atoms, bonds)
                    cleared_total |= atoms
                if _contradicts_reference_stereo(mol, taut):
                    continue
                tautomers.append((taut, idx))
            if cleared_total:
                logger.info(
                    "%r: tautomer enumeration left %d stereo element(s) unspecified at atom(s) %s, "
                    "because every proton-shift route to those tautomers passes through them.",
                    idx,
                    len(cleared_total),
                    sorted(cleared_total),
                )
        tautomers = _merge_same_constitution(tautomers)
        if self.expand_cleared_stereo:
            # Stereoisomer enumeration is off downstream, so a record the
            # tautomer step left with more unspecified elements than the input
            # had (a cleared center, or one a shift created) is written once
            # per configuration here; a record with only the input's own
            # unspecified elements is left as the user wrote it.
            expanded: list[tuple[Chem.Mol, str]] = []
            warned: set[str] = set()
            for taut, idx in tautomers:
                n_unspecified = count_unspecified_stereo(taut)
                if n_unspecified <= unspecified_in_input[idx]:
                    expanded.append((taut, idx))
                    continue
                if idx not in warned:
                    warned.add(idx)
                    logger.warning(
                        "%r: stereoisomer enumeration is disabled, but tautomer enumeration left "
                        "stereo elements unspecified; those tautomers are written once per "
                        "configuration, mirror pairs collapsed, so each configuration is its own "
                        "species.",
                        idx,
                    )
                expanded.extend((iso, idx) for iso in _expand_cleared(taut))
            tautomers = expanded
        with open(self.output, "w+") as f:
            for taut, idx in tautomers:
                f.write(Chem.MolToSmiles(taut).strip() + " " + str(idx).strip() + "\n")

    def run(self) -> None:
        """Execute tautomer enumeration."""
        if self.mode == "oechem":
            self.oe_taut()
        elif self.mode == "rdkit":
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')
