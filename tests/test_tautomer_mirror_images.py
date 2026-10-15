"""Tautomer enumeration emits no mirror image of a legitimate tautomer.

A stereo element a hydrogen migrated through on every route to a tautomer is
emitted unspecified; one the shifts leave untouched keeps the input's
configuration. Each expected set below is derived from that rule by hand --
every test's docstring says which centers each route to each record flattens --
and the regression sets are the enumerator's own output before the rule, which
the rule must leave alone.
"""

from __future__ import annotations

import logging

import pytest
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from Auto3D.domain.stereochemistry import enantiomer_key
from Auto3D.engines.isomers.factory import create_tautomer_engine
from Auto3D.foundation.config import Auto3DOptions


def _run_rd_taut(job_dir, smiles: str) -> list[str]:
    in_smi = job_dir / "taut_in.smi"
    in_smi.write_text(f"{smiles} probe\n")
    out_smi = job_dir / "taut_out.smi"
    create_tautomer_engine("rdkit", str(in_smi), str(out_smi), pka_norm=False).run()
    return [line.split()[0] for line in out_smi.read_text().splitlines() if line.strip()]


def _canonical(smiles_list):
    return sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles_list)


def test_d_erythrose_emits_four_tautomers_and_no_mirror_image(job_dir):
    """Every proton-shift route to the ketose and the enediols flattens C2, so
    its configuration is not the input's to keep; C3 is untouched and stays.
    Before the rule the enumerator emitted six records including D- and
    L-erythrulose under two ids and a second mirror pair among the enediols."""
    out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@H](O)CO")
    assert _canonical(out) == _canonical(
        ["O=C[C@H](O)[C@H](O)CO", "O=C(CO)[C@H](O)CO", "OC=C(O)[C@H](O)CO", "OCC(O)=C(O)CO"]
    )


def test_a_center_outside_the_tautomer_core_keeps_its_configuration(job_dir):
    """The guard against over-clearing: the carbinol center is three bonds from
    the ketone and no route touches it."""
    out = _run_rd_taut(job_dir, "C[C@@H](O)CCC(C)=O")
    assert _canonical(out) == _canonical(
        ["C=C(O)CC[C@@H](C)O", "CC(=O)CC[C@@H](C)O", "CC(O)=CC[C@@H](C)O"]
    )


def test_d_ribose_keeps_untouched_centers_and_clears_the_flattened_one(job_dir):
    """Three centers: the 2-ketose keeps C3 and C4; the 3-ketose is reached only
    through both enediols, so C2 is cleared and C4 kept; the enediols keep what
    they do not touch. Eight records with three mirror pairs before the rule."""
    out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@H](O)[C@H](O)CO")
    assert _canonical(out) == _canonical(
        [
            "O=C[C@H](O)[C@H](O)[C@H](O)CO",
            "O=C(CO)[C@H](O)[C@H](O)CO",
            "O=C(C(O)CO)[C@H](O)CO",
            "OC=C(O)[C@H](O)[C@H](O)CO",
            "OCC(O)=C(O)[C@H](O)CO",
        ]
    )


def test_a_meso_input_with_no_mirror_pair_still_loses_the_surrounded_assertion(job_dir):
    """meso-2,3-dihydroxybutanedial emits eight records and no two are mirror
    images, so a mirror-pair search finds nothing; the rule still clears the
    one center that both shifted regions surround, and keeps the center the
    3,4-enediol route leaves untouched."""
    out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@@H](O)C=O")
    canon = _canonical(out)
    assert len(canon) == 8
    assert "O=CC(=O)C(O)CO" in canon
    assert "O=CC(=O)[C@@H](O)CO" not in canon and "O=CC(=O)[C@H](O)CO" not in canon
    assert "O=C[C@H](O)C(=O)CO" in canon
    keys = [enantiomer_key(s) for s in out]
    assert len(set(keys)) == len(keys)


REGRESSION = {
    # center on the tautomeric core: the identity tautomer keeps it, the enol
    # at that position is planar there
    "CC(=O)[C@@H](C)CC": ["C=C(O)[C@@H](C)CC", "CCC(C)=C(C)O", "CC[C@H](C)C(C)=O"],
    # CIP relabeling is not an inversion: three records, every one keeping the center
    "OC[C@H](CC(=O)C)C=C": ["C=C[C@@H](C=C(C)O)CO", "C=C[C@H](CO)CC(=C)O", "C=C[C@H](CO)CC(C)=O"],
    # a destroyed center carries no descriptor; the carbinol center the shift
    # creates is unspecified
    "C[C@H](C(=O)C)N": [
        "C=C(N)C(C)O",
        "C=C(O)[C@@H](C)N",
        "CC(=N)C(C)O",
        "CC(=O)[C@@H](C)N",
        "CC(N)=C(C)O",
    ],
    # specified enol geometry: the identity tautomer keeps it
    "C/C=C(\\O)C": ["C/C=C(/C)O", "C=C(O)CC", "CCC(C)=O"],
    # E-enone: the kinetic enol keeps the untouched C=C geometry; the dienol
    # moves the bond, so its geometry is unspecified
    "C/C=C/C(C)=O": ["C/C=C/C(C)=O", "C=C(O)/C=C/C", "C=CC=C(C)O", "C=CCC(=C)O", "C=CCC(C)=O"],
}


@pytest.mark.parametrize("smiles", sorted(REGRESSION))
def test_regression_molecules_are_emitted_exactly_as_before(job_dir, smiles):
    assert _canonical(_run_rd_taut(job_dir, smiles)) == _canonical(REGRESSION[smiles])


def _species(records: list[str]) -> list[str]:
    options = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    species: list[str] = []
    for smiles in records:
        seen: dict[tuple, str] = {}
        for isomer in EnumerateStereoisomers(Chem.MolFromSmiles(smiles), options=options):
            iso_smi = Chem.MolToSmiles(isomer)
            seen.setdefault(enantiomer_key(iso_smi), iso_smi)
        species.extend(seen.values())
    return species


def test_d_erythrose_yields_six_species_and_no_mirror_images(job_dir):
    """Identity 1, the 2-ketose 1, each enediol 2 (the two double-bond
    geometries, which are not mirror images): six, where the enumerator's old
    output yielded nine with two mirror pairs."""
    species = _species(_run_rd_taut(job_dir, "O=C[C@H](O)[C@H](O)CO"))
    assert len(species) == 6
    assert len({enantiomer_key(s) for s in species}) == 6


def test_d_ribose_yields_eight_species_with_the_3_ketopentose_as_two_diastereomers(job_dir):
    out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@H](O)[C@H](O)CO")
    species = _species(out)
    assert len(species) == 8
    assert len({enantiomer_key(s) for s in species}) == 8
    three_ketose = [
        s
        for s in out
        if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == "O=C(C(O)CO)C(O)CO"
    ]
    assert len(three_ketose) == 1
    # the meso and the chiral diastereomer, under one tautomer id
    assert len(_species(three_ketose)) == 2


def test_without_stereoisomer_enumeration_a_cleared_record_is_written_once_per_configuration(
    job_dir, caplog
):
    from Auto3D.engines.isomers.tautomers import RDKitOrOEChemTautomerEngine

    in_smi = job_dir / "taut_in.smi"
    in_smi.write_text("O=C[C@H](O)[C@H](O)CO probe\n")
    out_smi = job_dir / "taut_out.smi"
    with caplog.at_level(logging.WARNING):
        RDKitOrOEChemTautomerEngine(
            "rdkit", str(in_smi), str(out_smi), False, expand_cleared_stereo=True
        ).run()
    out = [line.split()[0] for line in out_smi.read_text().splitlines() if line.strip()]
    constitutions = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in out]
    assert constitutions.count("O=CC(O)C(O)CO") == 1  # the identity record, untouched
    assert constitutions.count("O=C(CO)C(O)CO") == 1  # the 2-ketose keeps C3, nothing was cleared
    assert constitutions.count("OC=C(O)C(O)CO") == 2  # the 1,2-enediol: both geometries
    assert constitutions.count("OCC(O)=C(O)CO") == 2  # the 2,3-enediol: both geometries
    assert len(out) == 6
    assert (
        sum("stereoisomer enumeration is disabled" in r.getMessage() for r in caplog.records) == 1
    )


def test_the_tautomer_processor_expands_only_when_stereoisomer_enumeration_is_off(monkeypatch):
    import Auto3D.orchestration.processors as proc

    seen = {}

    class _Engine:
        def run(self):
            pass

    def fake_factory(*args, **kwargs):
        seen.update(kwargs)
        return _Engine()

    monkeypatch.setattr(proc, "create_tautomer_engine", fake_factory)
    monkeypatch.setattr(proc, "hash_taut_smi", lambda a, b: None)
    for enumerate_isomers in (True, False):
        config = Auto3DOptions(
            path="x.smi",
            k=1,
            enumerate_tautomer=True,
            tauto_engine="rdkit",
            enumerate_isomers=enumerate_isomers,
        )
        proc.TautomerProcessor(config).process("in.smi", "out.smi")
        assert seen["expand_cleared_stereo"] is (not enumerate_isomers)


def test_two_justified_routes_that_disagree_emit_the_element_unspecified(job_dir):
    """The symmetric case the constitution merge exists for.

    The meso 3-ketopentose reaches one 2-ketose constitution from either end of
    its chain, and the two routes keep opposite configurations at the center
    each leaves untouched. Both are justified, so neither is chosen: the
    merged record carries the element unspecified. The identity tautomer, zero
    steps away, still keeps both of its centers.
    """
    out = _run_rd_taut(job_dir, "O=C([C@H](O)CO)[C@@H](O)CO")
    by_constitution = {
        Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False): Chem.MolToSmiles(
            Chem.MolFromSmiles(s)
        )
        for s in out
    }
    assert len(out) == len(by_constitution) == 5
    assert by_constitution["O=C(C(O)CO)C(O)CO"] == "O=C([C@H](O)CO)[C@@H](O)CO"
    assert by_constitution["O=C(CO)C(O)C(O)CO"] == "O=C(CO)C(O)C(O)CO"


def _unspecified_centers(smiles: str) -> int:
    """Potential tetrahedral centers the record leaves open.

    Counted instead of every stereo element, because a tautomer that creates a
    C=C legitimately leaves that bond's geometry open; only an open *center*
    can be an element the rule cleared.
    """
    mol = Chem.MolFromSmiles(smiles)
    return sum(
        1
        for element in Chem.FindPotentialStereo(mol)
        if element.type == Chem.StereoType.Atom_Tetrahedral
        and element.specified != Chem.StereoSpecified.Specified
    )


def test_two_independent_shifts_flanking_a_center_do_not_clear_it(job_dir):
    """3-methylhexane-2,4-dione: C3 sits between the two carbonyls, and the
    bis-enol enolizes both of them outward, toward C1 and C6.

    Each enolization is its own one-step shift: the left one moves a hydrogen
    from C1 to the C2 oxygen, the right one from C6 to the C4 oxygen, and
    neither hydrogen crosses C3, which keeps its four substituents throughout.
    Two shifted regions flank C3 but nothing migrated through it, so its
    configuration is still the input's. Only the new C=C of the enol is open.
    """
    out = _run_rd_taut(job_dir, "CC(=O)[C@H](C)C(=O)CC")
    bis_enol = "C=C(O)C(C)C(O)=CC"
    records = [
        s for s in out if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == bis_enol
    ]
    assert len(records) == 1
    assert Chem.MolToSmiles(Chem.MolFromSmiles(records[0])) == "C=C(O)[C@H](C)C(O)=CC"
    assert _unspecified_centers(records[0]) == 0


def test_a_ring_center_between_two_independent_enolizations_keeps_its_configuration(job_dir):
    """2,4-dimethylcyclohexane-1,3-dione: the same two-independent-shifts case
    inside a ring, where the route search cannot use the far arc.

    C2 lies between the C1 and C3 carbonyls. The bis-enol enolizes C1 toward
    the ring methine that carries its methyl and C3 toward C4, so each shift
    moves a hydrogen onto its own carbonyl oxygen and neither crosses C2. The
    far arc of the ring is outside the tautomer core, so the route search alone
    would call C2 a separator; the hydrogen bookkeeping is what keeps it. The
    canonical spelling writes the retained tag as ``[C@@H]`` -- a relabeling
    from the output's own atom order, not an inversion, since the rule only
    ever clears a tag and never rewrites one.
    """
    out = _run_rd_taut(job_dir, "C[C@H]1CCC(=O)[C@H](C)C1=O")
    bis_enol = "CC1=C(O)C(C)C(O)=CC1"
    records = [
        s for s in out if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == bis_enol
    ]
    assert len(records) == 1
    assert Chem.MolToSmiles(Chem.MolFromSmiles(records[0])) == "CC1=C(O)[C@@H](C)C(O)=CC1"
    assert _unspecified_centers(records[0]) == 0
    assert all(_unspecified_centers(s) == 0 for s in out), sorted(out)


def test_mirror_image_records_of_one_constitution_collapse_onto_one_that_keeps_its_centers(job_dir):
    """meso-3,4-dihydroxy... the meso alpha-hydroxy 1,3-diketone
    ``C[C@H](O)C(=O)CC(=O)[C@H](O)C``, whose two ends are related by the meso
    symmetry.

    Enolizing the left carbonyl toward the central CH2 and enolizing the right
    one give two records of one mono-enol constitution. Neither route touches
    either carbinol center -- the hydrogen moves from the central CH2 onto one
    carbonyl oxygen -- so both records justly keep both centers, and the two
    records are mirror images of each other. Clearing both centers would hand
    the stereoisomer step three diastereomers where only this mirror pair is
    reachable, so one record stands for the pair: they are exactly degenerate
    in energy and enantiomer removal keeps one of them anyway.
    """
    out = _run_rd_taut(job_dir, "C[C@H](O)C(=O)CC(=O)[C@H](O)C")
    mono_enol = "CC(O)C(=O)C=C(O)C(C)O"
    records = [
        s for s in out if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == mono_enol
    ]
    assert len(records) == 1
    assert _unspecified_centers(records[0]) == _unspecified_centers("C[C@H](O)C(=O)CC(=O)[C@H](O)C")
    assert Chem.MolToSmiles(Chem.MolFromSmiles(records[0])) == "C[C@H](O)C(=O)C=C(O)[C@@H](C)O"

    identity = Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](O)C(=O)CC(=O)[C@H](O)C"))
    assert identity in _canonical(out)
    species = _species(out)
    keys = [enantiomer_key(s) for s in species]
    assert len(set(keys)) == len(keys)


def test_open_chain_glucose_emits_the_3_ketohexose_once_with_no_specified_center(job_dir):
    """Open-chain D-glucose reaches the 3-ketohexose by two routes that keep
    different centers, and the merge cannot choose between them.

    Shifting the carbonyl from C1 to C3 flattens C2 on the way and leaves C4
    and C5 alone; shifting it from C1 to C4 instead -- the same constitution,
    read from the other end of the chain -- flattens C2 and C3 and leaves only
    the center next to the terminal CH2OH. Neither record says less than the
    other, they are not mirror images, and they agree on no element, so the
    one emitted record asserts nothing at any of the three centers.
    """
    out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO")
    three_ketose = "O=C(C(O)CO)C(O)C(O)CO"
    records = [
        s
        for s in out
        if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == three_ketose
    ]
    assert len(records) == 1
    assert Chem.MolToSmiles(Chem.MolFromSmiles(records[0])) == three_ketose
    assert _unspecified_centers(records[0]) == 3

    species = _species(out)
    assert len(set(species)) == len(species)
    keys = [enantiomer_key(s) for s in species]
    assert len(set(keys)) == len(keys)


def test_a_double_bond_whose_order_changed_loses_its_geometry():
    """The bond branch of the clearing rule, called directly.

    A specified C=C that the shift turns into a single bond and back has no
    geometry to inherit, whatever its atoms' own signatures say, so it must be
    among the bonds the rule returns.
    """
    from Auto3D.engines.isomers.tautomers import _destroyed_stereo

    reference = Chem.MolFromSmiles("C/C=C/C(C)=O")
    tautomer = Chem.MolFromSmiles("C/C=C/C(C)=O")
    moved = Chem.RWMol(tautomer)
    stereo_bond = next(b for b in moved.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)
    begin, end = stereo_bond.GetBeginAtomIdx(), stereo_bond.GetEndAtomIdx()
    moved.GetBondBetweenAtoms(begin, end).SetBondType(Chem.BondType.SINGLE)
    tautomer = moved.GetMol()

    core = set(range(reference.GetNumAtoms()))
    _atoms, bonds = _destroyed_stereo(reference, tautomer, core)
    assert tautomer.GetBondBetweenAtoms(begin, end).GetIdx() in bonds


HEPTOSE = "O=C[C@H](O)[C@H](O)[C@H](C)[C@H](O)[C@H](O)C(C)=O"


def test_two_independent_carbonyl_migrations_clear_only_the_centers_they_cross():
    """Two 1,3-shifts in series, one at each end of a chain, called directly.

    The reference is a 4-methyl heptose with a terminal methyl ketone; the
    tautomer moves the C1 aldehyde to C3 and the C7 ketone to C5. That leaves
    four shifted regions: C1 gained two hydrogens, C3 and C5 each lost two,
    C7 gained two. The left migration crosses C2 and the right one crosses C6,
    so those two centers lose their configuration. C4 keeps its own: it sits
    between the two *losing* regions, and the gaining region on either side of
    it is separated from the losing region on the other by a third shifted
    region, so no hydrogen travelled across C4.

    The signs alone do not say this -- the outermost gaining and losing
    regions do flank C4 with opposite signs -- which is why only nearest
    neighbors are put on trial.
    """
    from Auto3D.engines.isomers.tautomers import _destroyed_stereo

    reference = Chem.MolFromSmiles(HEPTOSE)
    tautomer = Chem.MolFromSmiles("OC[C@H](O)C(=O)[C@H](C)C(=O)[C@H](O)C(C)O")
    core = set(range(reference.GetNumAtoms()))
    atoms, bonds = _destroyed_stereo(reference, tautomer, core)
    assert atoms == {2, 10}
    assert bonds == set()


def test_the_heptose_3_5_diketone_keeps_the_center_between_the_two_migrations(job_dir):
    """The same molecule end to end: the record is emitted with the center
    between the two migrations intact and the two they crossed open, plus the
    center the right-hand shift created."""
    out = _run_rd_taut(job_dir, HEPTOSE)
    diketone = Chem.MolToSmiles(
        Chem.MolFromSmiles("CC(C(=O)C(O)CO)C(=O)C(O)C(C)O"), isomericSmiles=False
    )
    records = [
        s for s in out if Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) == diketone
    ]
    assert len(records) == 1
    assert _unspecified_centers(records[0]) == 3
    mol = Chem.MolFromSmiles(records[0])
    specified = [
        element.centeredOn
        for element in Chem.FindPotentialStereo(mol)
        if element.type == Chem.StereoType.Atom_Tetrahedral
        and element.specified == Chem.StereoSpecified.Specified
    ]
    assert len(specified) == 1
    # The one center that keeps its configuration is the methyl-bearing C4.
    kept = mol.GetAtomWithIdx(specified[0])
    assert any(nb.GetSymbol() == "C" and nb.GetTotalNumHs() == 3 for nb in kept.GetNeighbors()), (
        Chem.MolToSmiles(mol)
    )


def test_survivors_that_say_the_same_thing_are_merged_without_clearing_anything(job_dir, caplog):
    """Two records of one constitution can come out of the clearing identical.

    Neither says less than the other, so both survive the drop, and there is
    nothing for them to disagree about: one of them is the merged record as it
    stands. The 4-methyl heptose dialdehyde reaches that case twice.
    """
    with caplog.at_level(logging.INFO):
        out = _run_rd_taut(job_dir, "O=C[C@H](O)[C@H](O)[C@H](C)[C@H](O)[C@H](O)C=O")
    assert out
    messages = [r.getMessage() for r in caplog.records]
    assert any("stereo element" in m for m in messages), messages
    assert not [m for m in messages if "0 stereo element" in m], messages
