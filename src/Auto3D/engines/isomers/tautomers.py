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

from Auto3D.domain.stereochemistry import enantiomer_key
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

    Applied after the unjustified elements of ``tautomer`` have been cleared,
    so a record that reaches here still carrying the input's constitution and a
    different configuration is asserting a configuration the clearing did not
    take away: the input written wrong. It is dropped outright rather than
    relaxed to an unspecified element.

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
    mol: Chem.Mol,
    pivot: set[int],
    delta: set[int],
    core: set[int],
    hydrogen_change: list[int],
) -> bool:
    """True when a hydrogen migrated from one side of ``pivot`` to the other.

    A pair of ``delta`` components is a candidate for one such migration when
    one of them gained hydrogens and the other lost them, and when no third
    shifted region lies between the two -- a pair with another shifted region
    in the way is two shifts in series, not one hydrogen crossing from the one
    region to the other. Each candidate pair is then asked the question the
    pivot is actually on trial for: whether every route between the two inside
    the tautomer core passes through it.

    All three conditions are needed. Without the signs, a 1,3-diketone that
    enolizes both of its outer methyls would lose the center between the
    carbonyls, although each shift moves a hydrogen only within its own region
    and the center keeps all four substituents. Without the neighbor test, a
    chain carrying two such independent 1,3-shifts, one at each end, would
    lose the center between them, because the outermost gaining and losing
    regions do flank it even though neither shift reaches it.

    ``hydrogen_change`` is the per-atom total hydrogen count in ``mol`` minus
    the count in the reference, over the shared atom indexing.
    """
    components = _components(mol, delta)
    if len(components) < 2:
        return False
    reachable = core | delta
    net = [sum(hydrogen_change[j] for j in component) for component in components]
    for i, a in enumerate(components):
        if net[i] == 0:
            continue
        for j in range(i + 1, len(components)):
            if net[j] == 0 or (net[i] > 0) == (net[j] > 0):
                continue
            # Nearest neighbors first, with the pivot itself allowed: a route
            # that has to cross a third shifted region to get from one of the
            # pair to the other means the hydrogen did not travel between
            # them, so the pivot is not on its way.
            between = {
                atom
                for k, component in enumerate(components)
                if k not in (i, j)
                for atom in component
            }
            if not _connected_without(mol, a, components[j], reachable - between):
                continue
            if not _connected_without(mol, a - pivot, components[j] - pivot, reachable - pivot):
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
    an element inside the core loses it when its own bonding changed or when a
    hydrogen migrated from one side of it to the other.

    Raises:
        ValueError: If ``reference`` and ``tautomer`` do not share one atom
            indexing, which every assumption here rests on.
    """
    ref_sig, taut_sig = _local_signatures(reference), _local_signatures(tautomer)
    delta = {i for i, (r, t) in enumerate(zip(ref_sig, taut_sig, strict=True)) if r != t}
    hydrogen_change = [
        t.GetTotalNumHs() - r.GetTotalNumHs()
        for r, t in zip(reference.GetAtoms(), tautomer.GetAtoms(), strict=True)
    ]
    atoms, bonds = set(), set()
    for atom in tautomer.GetAtoms():
        i = atom.GetIdx()
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED or i not in core:
            continue
        # `or` short-circuits, so the surround test runs only for an element
        # whose own bonding the shifts left alone.
        if i in delta or _separates_shifted_regions(tautomer, {i}, delta, core, hydrogen_change):
            atoms.add(i)
    for bond in tautomer.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ref_bond = reference.GetBondBetweenAtoms(a, b)
        destroyed = (
            ref_bond is None
            or ref_bond.GetBondType() != bond.GetBondType()
            or a in delta
            or b in delta
        ) or (
            {a, b} <= core
            and _separates_shifted_regions(tautomer, {a, b}, delta, core, hydrogen_change)
        )
        if destroyed:
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


def _stereo_classes(mol: Chem.Mol) -> list[int]:
    """Per-atom symmetry class of ``mol`` on its stereo-stripped graph.

    ``breakTies=False`` gives symmetry classes rather than a total order, so
    two records of one constitution that were parsed from different isomeric
    SMILES key the same element the same way. With ``breakTies=True`` the order
    within a class falls back to input atom order, which differs between those
    two records, and a comparison between them would silently find nothing in
    common.
    """
    stripped = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stripped)
    return list(Chem.CanonicalRankAtoms(stripped, breakTies=False))


def _stereo_fingerprint(mol: Chem.Mol) -> Counter | None:
    """Multiset of (symmetry class, CIP descriptor) over the specified elements.

    A multiset rather than a set, so that a record specifying two
    symmetry-equivalent centers is not flattened onto one that specifies a
    single center.

    Returns ``None`` when a tagged atom comes back without a CIP descriptor:
    the record then cannot be compared with the other records of its
    constitution, and the caller keeps it as it is rather than collapsing it
    on an incomplete comparison.
    """
    classes = _stereo_classes(mol)
    labeled = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(labeled)
    fingerprint: Counter = Counter()
    for atom in labeled.GetAtoms():
        if atom.HasProp("_CIPCode"):
            fingerprint[(classes[atom.GetIdx()], atom.GetProp("_CIPCode"))] += 1
        elif atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            logger.warning(
                "%s: the stereocenter at atom %d has no CIP descriptor, so this tautomer is "
                "not compared with the other records of its constitution and is written as it "
                "stands.",
                Chem.MolToSmiles(mol),
                atom.GetIdx(),
            )
            return None
    for bond in labeled.GetBonds():
        if bond.HasProp("_CIPCode"):
            key = tuple(sorted((classes[bond.GetBeginAtomIdx()], classes[bond.GetEndAtomIdx()])))
            fingerprint[(key, bond.GetProp("_CIPCode"))] += 1
    return fingerprint


def _descriptors_by_class(fingerprint: Counter) -> dict:
    """Regroup a fingerprint as symmetry class -> multiset of its descriptors."""
    by_class: dict = defaultdict(Counter)
    for (stereo_class, code), count in fingerprint.items():
        by_class[stereo_class][code] += count
    return by_class


def _disagreeing_classes(fingerprints: list[Counter]) -> set:
    """The symmetry classes the fingerprints do not all specify identically.

    A class one fingerprint does not mention at all counts as a disagreement:
    one record asserting an element that another leaves open is not agreement
    on that element.
    """
    by_class = [_descriptors_by_class(fp) for fp in fingerprints]
    classes = set().union(*(set(entry) for entry in by_class))
    return {
        stereo_class
        for stereo_class in classes
        if any(entry.get(stereo_class) != by_class[0].get(stereo_class) for entry in by_class)
    }


def _merge_same_constitution(records: list[tuple[Chem.Mol, str]]) -> list[tuple[Chem.Mol, str]]:
    """Reduce the records of one constitution, within one input, to one record.

    Fingerprints are multisets of (symmetry class, CIP descriptor), so a
    record is compared with another by what it says, not by atom indices,
    which differ between two records of one constitution read from opposite
    ends of a chain. A record that says strictly less than another says
    nothing the other does not and is dropped. If more than one record
    survives that, they came by routes that justify different assignments, and
    either

    * they are mirror images of one another differing at two or more elements,
      in which case one stands for the pair: they are exactly degenerate in
      energy and ``remove_enantiomers`` collapses them downstream, while
      clearing their elements would instead enumerate diastereomers that no
      route produced; or
    * one record is emitted keeping only the elements every survivor specifies
      with the same descriptor. An element that the survivors label differently,
      or that any survivor leaves open, is emitted unspecified.

    A record whose descriptors cannot be read is kept as it stands.
    """
    groups: dict[tuple[str, str], list[Chem.Mol]] = defaultdict(list)
    for mol, idx in records:
        groups[(idx, Chem.MolToSmiles(mol, isomericSmiles=False))].append(mol)
    merged: list[tuple[Chem.Mol, str]] = []
    for (idx, _constitution), mols in groups.items():
        fingerprints = [_stereo_fingerprint(m) for m in mols]
        comparable = [(m, fp) for m, fp in zip(mols, fingerprints, strict=True) if fp is not None]
        if comparable:
            merged.append((_merge_one_constitution(idx, comparable), idx))
        merged.extend((m, idx) for m, fp in zip(mols, fingerprints, strict=True) if fp is None)
    return merged


def _merge_one_constitution(idx: str, comparable: list[tuple[Chem.Mol, Counter]]) -> Chem.Mol:
    """The one record that stands for ``comparable``, all of one constitution."""
    survivors = [
        (mol, fingerprint)
        for i, (mol, fingerprint) in enumerate(comparable)
        if not any(
            j != i and fingerprint != other and fingerprint <= other
            for j, (_, other) in enumerate(comparable)
        )
    ]
    first = survivors[0][0]
    if len(survivors) == 1:
        return first
    disagreeing = _disagreeing_classes([fingerprint for _, fingerprint in survivors])
    if not disagreeing:
        # Survivors that say exactly the same thing: none is a strict subset of
        # another, so all of them survived the drop above, and any one of them
        # is the merged record as it stands.
        return first
    keys = {enantiomer_key(Chem.MolToSmiles(mol)) for mol, _ in survivors}
    # Elements, not symmetry classes: on a symmetric constitution two centers
    # of one class can both flip, which is one disagreeing class but two
    # differing elements and so a mirror pair rather than a single open center.
    differing = sum(
        count for (stereo_class, _), count in survivors[0][1].items() if stereo_class in disagreeing
    )
    if len(keys) == 1 and differing >= 2:
        logger.info(
            "%r: %d tautomer records of one constitution are mirror images of one another; "
            "one stands for them all, since they have the same energy and enantiomer removal "
            "keeps one of them.",
            idx,
            len(survivors),
        )
        return first
    # Every element of a disagreeing symmetry class, since a class is compared
    # as a multiset of descriptors and a difference there does not say which of
    # its members carries it. Elements that are not specified in `first` are
    # left out, so the clearing and the line below name only real assertions.
    classes = _stereo_classes(first)
    atoms = {
        a.GetIdx()
        for a in first.GetAtoms()
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        and classes[a.GetIdx()] in disagreeing
    }
    bonds = {
        b.GetIdx()
        for b in first.GetBonds()
        if b.GetStereo() != Chem.BondStereo.STEREONONE
        and tuple(sorted((classes[b.GetBeginAtomIdx()], classes[b.GetEndAtomIdx()]))) in disagreeing
    }
    logger.info(
        "%r: %d tautomer records of one constitution label %d stereo element(s) differently "
        "(atom(s) %s); those elements are emitted unspecified.",
        idx,
        len(survivors),
        len(atoms) + len(bonds),
        sorted(atoms),
    )
    return _clear_stereo(first, atoms, bonds)


#: ``Chem.FindPotentialStereo`` element types that live on a bond rather than
#: an atom, so an element can be named by the atom pair it spans.
_BOND_STEREO_TYPES = (
    Chem.StereoType.Bond_Double,
    Chem.StereoType.Bond_Atropisomer,
    Chem.StereoType.Bond_Cumulene_Even,
)


def _stereo_elements(mol: Chem.Mol) -> tuple[set[tuple], set[tuple]]:
    """The molecule's potential stereo elements, as (specified, unspecified).

    Each element is named by the molecule's own atom indexing -- ``("atom", i)``
    or ``("bond", (i, j))`` -- rather than by a stereo-element index, so the
    same element is recognizable between the input and a tautomer, which share
    that indexing, and comparing the two sets says which elements the
    enumeration opened up and which the input had already left open.
    """
    specified: set[tuple] = set()
    unspecified: set[tuple] = set()
    for element in Chem.FindPotentialStereo(mol):
        if element.type in _BOND_STEREO_TYPES:
            bond = mol.GetBondWithIdx(element.centeredOn)
            key = ("bond", tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
        else:
            key = ("atom", element.centeredOn)
        if element.specified == Chem.StereoSpecified.Specified:
            specified.add(key)
        else:
            unspecified.add(key)
    return specified, unspecified


def _name_elements(elements: set[tuple]) -> tuple[list[int], list[tuple[int, int]]]:
    """Split a stereo-element set into sorted atom indices and atom-index pairs."""
    atoms = sorted(index for kind, index in elements if kind == "atom")
    bonds = sorted(index for kind, index in elements if kind == "bond")
    return atoms, bonds


def _apply_stereo_rule(reference: Chem.Mol, tautomer: Chem.Mol, core: set[int]) -> Chem.Mol:
    """``tautomer`` with every stereo element the route did not justify cleared."""
    atoms, bonds = _destroyed_stereo(reference, tautomer, core)
    if not atoms and not bonds:
        return tautomer
    return _clear_stereo(tautomer, atoms, bonds)


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
    whose own bonding is unchanged and that no hydrogen migrated through,
    keeps the configuration the input gave it, and every other element is
    emitted unspecified for the stereoisomer step to enumerate. The records of
    one constitution are then reduced to one. Tetrahedral centers and
    double-bond geometry are covered; axial and planar chirality and the other
    non-tetrahedral classes are not perceived here and are emitted as the
    enumerator produces them.

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
        if expand_cleared_stereo and mode == "oechem":
            logger.warning(
                "The oechem tautomer backend applies no stereo policy, so it leaves no stereo "
                "element unspecified for this option to expand; its output is written as "
                "OEChem produces it."
            )

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
        outside the tautomer core, or when its own bonding is unchanged and no
        hydrogen migrated from one side of it to the other; every other element
        is cleared and re-perceived, so the record states that the
        configuration is open rather than asserting the input's. A tautomer
        with the input's constitution and a different configuration is then
        dropped, and the records of one constitution are reduced to one.

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
        inputs: dict[str, Chem.Mol] = {}
        for smi, idx in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Skipping molecule {idx!r}: failed to parse {smi!r}")
                continue
            result = enumerator.Enumerate(mol)
            core = set(result.modifiedAtoms)
            try:
                kept = [
                    taut
                    for taut in (_apply_stereo_rule(mol, taut, core) for taut in result)
                    if not _contradicts_reference_stereo(mol, taut)
                ]
            except ValueError as error:
                # _destroyed_stereo compares the input and the tautomer atom by
                # atom, so an enumerator output that does not share the input's
                # atom indexing cannot be judged. One molecule is skipped
                # rather than the whole chunk.
                logger.warning(
                    "Skipping molecule %r: its tautomers do not share the input's atom "
                    "indexing (%s)",
                    idx,
                    error,
                )
                continue
            inputs[idx] = mol
            tautomers.extend((taut, idx) for taut in kept)
        tautomers = _merge_same_constitution(tautomers)
        self._log_opened_elements(tautomers, inputs)
        if self.expand_cleared_stereo:
            tautomers = self._expand_opened_records(tautomers, inputs)
        with open(self.output, "w+") as f:
            for taut, idx in tautomers:
                f.write(Chem.MolToSmiles(taut).strip() + " " + str(idx).strip() + "\n")

    def _log_opened_elements(
        self, tautomers: list[tuple[Chem.Mol, str]], inputs: dict[str, Chem.Mol]
    ) -> None:
        """Report which of each input's own stereo elements the output leaves open.

        Read off the records that are actually written, not off the clearing
        decisions: an element cleared in one tautomer but kept in every record
        that survives the constitution merge is not something the user needs
        to hear about.
        """
        open_in_output: dict[str, set[tuple]] = defaultdict(set)
        for taut, idx in tautomers:
            open_in_output[idx] |= _stereo_elements(taut)[1]
        for idx, mol in inputs.items():
            opened = _stereo_elements(mol)[0] & open_in_output.get(idx, set())
            if not opened:
                continue
            atoms, bonds = _name_elements(opened)
            logger.info(
                "%r: %d stereo element(s) the input specified were left unspecified by tautomer "
                "enumeration -- atom(s) %s, double bond(s) %s, by atom index counted from zero "
                "in the order the input lists its atoms.",
                idx,
                len(opened),
                atoms,
                bonds,
            )

    def _expand_opened_records(
        self, tautomers: list[tuple[Chem.Mol, str]], inputs: dict[str, Chem.Mol]
    ) -> list[tuple[Chem.Mol, str]]:
        """One record per configuration for records carrying a newly open element.

        Stereoisomer enumeration is off downstream, so an element this step
        opened would otherwise reach embedding and be fixed per conformer,
        putting two epimers under one species id. A record whose open elements
        are all elements the input itself left open is written as the user
        wrote it.
        """
        open_in_input = {idx: _stereo_elements(mol)[1] for idx, mol in inputs.items()}
        expanded: list[tuple[Chem.Mol, str]] = []
        warned: set[str] = set()
        for taut, idx in tautomers:
            if not _stereo_elements(taut)[1] - open_in_input[idx]:
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
        return expanded

    def run(self) -> None:
        """Execute tautomer enumeration."""
        if self.mode == "oechem":
            self.oe_taut()
        elif self.mode == "rdkit":
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')
