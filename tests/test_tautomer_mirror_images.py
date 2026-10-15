"""Tautomer enumeration emits no mirror image of a legitimate tautomer.

A stereo element the proton shifts must flatten on every route to a tautomer
is emitted unspecified; one the shifts leave untouched keeps the input's
configuration. The molecules and the expected sets are those the chemistry
brief verified by hand; the regression sets below are the enumerator's own
output before the rule, which the rule must leave alone.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from Auto3D.domain.stereochemistry import enantiomer_key
from Auto3D.engines.isomers.factory import create_tautomer_engine


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
        config = SimpleNamespace(
            enumerate_tautomer=True,
            tauto_engine="rdkit",
            pKaNorm=True,
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
