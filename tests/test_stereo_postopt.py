"""Post-optimization stereochemistry validation (C9).

An optimization that inverts a stereocenter or rotates through a double bond
produces a molecule of different chemical identity than its title. check_connectivity
compares interatomic distances against UFF radii and is stereo-blind, so nothing
caught it. These tests pin the detector and the three filters that act on it.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.domain.filtering import filter_unique_optimized
from Auto3D.domain.ranking import ConformerRanker
from Auto3D.domain.stereo_check import (
    STEREO_CHANGED_PROP,
    apply_optimized_coords,
    stereo_descriptors_from_3d,
    stereo_preserved,
)


def _embedded(smiles: str = "C/C=C/C[C@H](O)Cl", seed: int = 7) -> Chem.Mol:
    """A molecule carrying both a tetrahedral center and a defined C=C."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _reflected_coords(mol: Chem.Mol) -> list[list[float]]:
    """Coordinates reflected through the origin -- the mirror image."""
    conf = mol.GetConformer()
    return [
        [-conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y, -conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]


def _nudged_coords(mol: Chem.Mol) -> list[list[float]]:
    """Coordinates displaced far too little to change any configuration."""
    conf = mol.GetConformer()
    return [
        [conf.GetAtomPosition(i).x + 0.01, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]


class TestDescriptorReading:
    def test_reflection_inverts_the_center_and_spares_the_double_bond(self):
        """Reflection flips tetrahedral configuration; E/Z is reflection-invariant."""
        mol = _embedded()
        atoms_before, bonds_before = stereo_descriptors_from_3d(mol)
        assert atoms_before, "no tetrahedral descriptor was read"
        assert bonds_before, "no double-bond descriptor was read"

        conf = mol.GetConformer()
        for i, position in enumerate(_reflected_coords(mol)):
            conf.SetAtomPosition(i, position)
        atoms_after, bonds_after = stereo_descriptors_from_3d(mol)

        assert atoms_after != atoms_before, "reflection did not invert the center"
        assert bonds_after == bonds_before, "reflection changed double-bond stereo"

    def test_descriptors_are_stable_under_a_small_displacement(self):
        """A geometry that barely moves reads identically."""
        mol = _embedded()
        before = stereo_descriptors_from_3d(mol)
        conf = mol.GetConformer()
        for i, position in enumerate(_nudged_coords(mol)):
            conf.SetAtomPosition(i, position)
        assert stereo_descriptors_from_3d(mol) == before

    def test_rotating_a_double_bond_past_ninety_degrees_is_detected(self):
        """E/Z is the other half of the identity check and needs its own case.

        Reflection cannot exercise it: mirroring is E/Z-invariant by
        construction, so every other test here leaves the bond descriptor alone.
        """
        from rdkit.Chem import rdMolTransforms

        mol = _embedded("C/C=C/C")
        atoms_before, bonds_before = stereo_descriptors_from_3d(mol)
        assert bonds_before, "no double-bond descriptor was read"

        conf = mol.GetConformer()
        rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, 0.0)
        atoms_after, bonds_after = stereo_descriptors_from_3d(mol)

        assert bonds_after != bonds_before, "a rotated C=C was not detected"
        assert atoms_after == atoms_before

    def test_inverting_one_center_of_two_is_detected(self):
        """A single inverted center must register even when others hold.

        Reflection inverts every center at once, so on its own it cannot
        distinguish "the molecule was mirrored" from "one center moved".
        """
        mol = _embedded("C[C@H](O)[C@H](F)Cl")
        before = stereo_descriptors_from_3d(mol)
        assert len(before[0]) == 2, f"expected two centers: {before}"

        conf = mol.GetConformer()
        idx = sorted(i for i, _ in before[0])[0]
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        first, second = neighbors[0], neighbors[1]
        position_a = list(conf.GetAtomPosition(first))
        position_b = list(conf.GetAtomPosition(second))
        conf.SetAtomPosition(first, position_b)
        conf.SetAtomPosition(second, position_a)

        after = stereo_descriptors_from_3d(mol)
        assert after[0] != before[0], "inverting one center was not detected"
        assert len(after[0]) == len(before[0]), after

    def test_inversion_is_flagged_under_new_stereo_perception(self, new_stereo_perception):
        """The detector must not depend on a process-global RDKit setting.

        ``AssignStereochemistryFrom3D`` sets ``_CIPCode`` only under RDKit's
        legacy stereo perception; chiral tags it sets under both. A descriptor
        read from ``_CIPCode`` therefore returns an empty tetrahedral half --
        and reports every inversion as "preserved" -- as soon as anything in
        the process turns legacy perception off.
        """
        mol = _embedded("C[C@H](O)CC")
        before = stereo_descriptors_from_3d(mol)
        assert before[0], "no tetrahedral descriptor was read under new perception"

        conf = mol.GetConformer()
        for i, position in enumerate(_reflected_coords(mol)):
            conf.SetAtomPosition(i, position)

        assert stereo_descriptors_from_3d(mol) != before, (
            "reflection went undetected under new stereo perception"
        )

    def test_phosphine_inversion_is_detected(self, stereo_perception):
        """A trivalent phosphorus that inverts during optimization must register.

        A pyramidal P(III) is configurationally stable at room temperature
        (inversion costs tens of kcal/mol for an ordinary tertiary phosphine,
        against roughly 6 for an amine), so an optimizer that walks one through
        its planar transition state has changed the compound. The barrier is
        substituent-dependent and much lower for aromatic phosphorus, which is
        why that case is excluded instead -- see
        ``test_aromatic_phosphorus_is_not_flagged_after_relaxation``. RDKit perceives no stereochemistry there at all -- no
        ``_CIPCode`` and no chiral tag, under either perception mode -- so
        without the hand-assigned tag both readings are empty and the
        inversion is reported as preserved.
        """
        mol = _embedded("CC[P@@](C)CCC")
        before = stereo_descriptors_from_3d(mol)
        assert before[0], "no descriptor was read for the phosphine center"

        conf = mol.GetConformer()
        for i, position in enumerate(_reflected_coords(mol)):
            conf.SetAtomPosition(i, position)

        assert stereo_descriptors_from_3d(mol) != before, "a phosphine inversion went undetected"

    def test_aromatic_phosphorus_is_not_flagged_after_relaxation(self, stereo_perception):
        """A planar, aromatic phosphorus must not produce a false ``Stereo_changed``.

        A phosphole's P is sp2 and nearly coplanar with its three neighbors, so
        the signed volume that resolves a pyramidal phosphine is noise-level
        here and its sign is a property of the conformer. Tagging it means an
        ordinary relaxation can flip the tag -- measured 2 of 6 noise trials at
        this amplitude before aromatic P was excluded -- and each flip marks the
        record ``Stereo_changed=True``, which drops it in ``filter_conformers``
        and rejects the relaxation in ``clash_relief``. Excluding aromatic P
        makes the tetrahedral half empty for this molecule, so it cannot change.
        """
        mol = _embedded("Cc1cccp1C", seed=1)
        before = stereo_descriptors_from_3d(mol)
        assert not before[0], (
            f"an aromatic phosphorus was tagged, so a relaxation can flip it: {before[0]}"
        )

        noise = random.Random(0)
        for trial in range(6):
            perturbed = Chem.Mol(mol)
            conf = perturbed.GetConformer()
            for i in range(perturbed.GetNumAtoms()):
                position = conf.GetAtomPosition(i)
                conf.SetAtomPosition(
                    i,
                    (
                        position.x + noise.gauss(0, 0.15),
                        position.y + noise.gauss(0, 0.15),
                        position.z + noise.gauss(0, 0.15),
                    ),
                )
            AllChem.MMFFOptimizeMolecule(perturbed, maxIters=2000)
            assert stereo_descriptors_from_3d(perturbed) == before, (
                f"trial {trial}: relaxing a phosphole changed its descriptor, which "
                f"would be reported as a stereochemistry change"
            )


class TestApplyOptimizedCoords:
    def test_inversion_is_detected_and_marked(self):
        mol = _embedded()
        assert apply_optimized_coords(mol, _reflected_coords(mol)) is False
        assert mol.GetProp(STEREO_CHANGED_PROP) == "True"
        assert stereo_preserved(mol) is False

    def test_a_preserved_geometry_is_marked_preserved(self):
        mol = _embedded()
        assert apply_optimized_coords(mol, _nudged_coords(mol)) is True
        assert mol.GetProp(STEREO_CHANGED_PROP) == "False"
        assert stereo_preserved(mol) is True

    def test_the_coordinates_are_actually_written(self):
        """The function must still do the job it replaced, not only flag."""
        mol = _embedded()
        target = [[float(i), 0.0, 0.0] for i in range(mol.GetNumAtoms())]
        apply_optimized_coords(mol, target)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            assert conf.GetAtomPosition(i).x == pytest.approx(float(i))

    def test_a_molecule_without_stereo_is_never_flagged(self):
        """An achiral molecule cannot change configuration."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
        assert apply_optimized_coords(mol, _reflected_coords(mol)) is True
        assert mol.GetProp(STEREO_CHANGED_PROP) == "False"


class TestStereoPreservedPredicate:
    def test_absent_property_reads_as_preserved(self):
        """Molecules from paths that never run the check are not dropped."""
        assert stereo_preserved(_embedded()) is True

    def test_the_marker_is_read_case_insensitively(self):
        mol = _embedded()
        mol.SetProp(STEREO_CHANGED_PROP, "true")
        assert stereo_preserved(mol) is False


class TestClashReliefDoesNotRejectPlanarPnictogens:
    """``relieve_clash`` reads the descriptor, runs a force field, reads again.

    That makes it the one seam where the molecule's *graph* can change between
    the two readings, not just its coordinates: ``MMFFOptimizeMolecule``
    sanitizes under MMFF's own aromaticity model, which does not consider a
    phosphole aromatic, so it clears ``GetIsAromatic()`` on the phosphorus in
    place. An exclusion that consulted only that flag would therefore hold on
    the "before" read and lapse on the "after" read, inventing a configuration
    change out of nothing and discarding every phosphole conformer that needed
    clash relief. The companion to this is
    ``TestDescriptorReading.test_phosphine_inversion_is_detected``, which pins
    the other direction -- that a pyramidal phosphine is still checked.
    """

    def test_aromatic_phosphorus_survives_clash_relief(self, monkeypatch):
        """The forced clash must be a NON-bonded pair, or the bonded-pair
        exclusion short-circuits before MMFF ever runs, and this test stops
        exercising the aromaticity hazard it is named for.

        Atom 0 (the methyl carbon on ring C2) and atom 6 (the methyl carbon on
        P1) are a genuinely non-bonded 1,4 pair across the C2-P1 ring bond,
        naturally 3.814 A apart in this seed's embedding. Forcing them to
        0.5 A -- well below 0.8 -- still reaches the force-field branch, where
        MMFF clears the phosphole's aromatic flag and ``relieve_clash`` must
        not mistake that for a configuration change.
        """
        from rdkit.Chem import AllChem

        from Auto3D.domain.clash_relief import relieve_clash

        mol = _embedded("Cc1cccp1C", seed=1)
        assert mol.GetBondBetweenAtoms(0, 6) is None, "test premise: 0-6 must be non-bonded"
        conf = mol.GetConformer()
        origin = conf.GetAtomPosition(0)
        conf.SetAtomPosition(6, (origin.x + 0.5, origin.y, origin.z))

        calls = []
        real_mmff = AllChem.MMFFOptimizeMolecule

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_mmff(*args, **kwargs)

        monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", _spy)

        assert relieve_clash(mol, conf_id=0, min_distance=0.8) is True, (
            "a phosphole conformer was discarded by clash relief, because the "
            "force field cleared the aromatic flag between the two descriptor reads"
        )
        assert len(calls) == 1, (
            f"MMFF must run exactly once for this test to exercise the aromaticity "
            f"hazard it is named for; ran {len(calls)} times"
        )
        assert mol.GetAtomWithIdx(5).GetIsAromatic() is False, (
            "the phosphorus's aromatic flag must actually have been cleared by "
            "MMFF for this test to pin the hazard it is named for"
        )


def test_h2_survives_clash_relief():
    """A bonded pair shorter than ``MIN_ATOM_DISTANCE`` is not a clash.

    H2 (and its isotopologues HD/D2) is essentially the only ordinary
    covalent bond whose equilibrium length (0.741 A) sits below the 0.9 A
    non-bonded threshold -- H-F (0.917), O-H (0.958), N-H (1.012) and C-H
    (1.087) are all above it. The non-bonded clash test is therefore excused
    for bonded pairs; whether a bonded pair's own length is chemically sane
    is a separate question. For a BONDED pair specifically, the
    connectivity check (``domain.connectivity.check_connectivity``)
    downstream tests only whether it is too LONG (broken) -- its too-short
    test applies to non-bonded pairs only -- so the only check a
    too-short bond meets at all is ``MIN_BONDED_FRACTION`` here:
    H2's 0.741 A clears even that (0.6 x H-H's 0.708 A reference length is
    0.425 A).
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from Auto3D.domain.clash_relief import relieve_clash

    mol = Chem.RWMol()
    a = mol.AddAtom(Chem.Atom(1))
    b = mol.AddAtom(Chem.Atom(1))
    mol.AddBond(a, b, Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(0.74, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    assert relieve_clash(mol.GetMol(), 0) is True


def test_a_crushed_bonded_pair_still_triggers_relief(monkeypatch):
    """Excluding bonded pairs from the 0.9 A non-bonded test must not
    open a hole for a bonded pair at a physically impossible distance.

    A 0.35 A C-H distance -- far below any real bond, and far below even
    H2's 0.741 A -- used to be caught by the earlier unconditional clash
    test (min pairwise distance, bonded or not, < 0.9 A). Once bonded
    pairs were excluded outright, this geometry passed through untouched
    and unrelaxed. ``MIN_BONDED_FRACTION`` restores the floor for bonded
    pairs specifically, scaled to each bond's own UFF reference length so
    ordinary short bonds (H2 included) are still exempt.
    """
    from rdkit.Chem import AllChem

    from Auto3D.domain.clash_relief import relieve_clash

    mol = _embedded("CC", seed=3)  # ethane: atom 0-1 is C-C, atom 0-2 is a C-H bond
    assert mol.GetBondBetweenAtoms(0, 2) is not None, "test premise: 0-2 must be a bonded pair"
    conf = mol.GetConformer()
    origin = conf.GetAtomPosition(0)
    conf.SetAtomPosition(2, (origin.x + 0.35, origin.y, origin.z))

    calls = []
    real_mmff = AllChem.MMFFOptimizeMolecule

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_mmff(*args, **kwargs)

    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", _spy)

    assert relieve_clash(mol, conf_id=0) is True
    assert len(calls) == 1, f"expected MMFF to run exactly once, ran {len(calls)} times"


def test_the_bonded_floor_scales_with_the_bonds_own_reference_length(monkeypatch):
    """NEW-1/R51: a flat 0.5 A bonded floor undercovers heavier bonds.

    A C-H bond's UFF reference length is 0.757 + 0.354 = 1.111 A, so
    ``MIN_BONDED_FRACTION``'s 0.6x is 0.667 A -- well above the old flat
    0.5 A floor. A 0.60 A C-H distance is below 0.667 A (so it must trigger
    relief under the radius-scaled rule) but above the old flat 0.5 A floor
    (so it would NOT have triggered relief under R47). The 0.35 A case
    (further below both floors) must still trigger relief too.
    """
    from rdkit.Chem import AllChem

    from Auto3D.domain.clash_relief import relieve_clash

    def _mmff_call_count(ch_distance: float) -> int:
        mol = _embedded("CC", seed=3)
        assert mol.GetBondBetweenAtoms(0, 2) is not None, "test premise: 0-2 is a bonded pair"
        conf = mol.GetConformer()
        origin = conf.GetAtomPosition(0)
        conf.SetAtomPosition(2, (origin.x + ch_distance, origin.y, origin.z))

        calls = []
        real_mmff = AllChem.MMFFOptimizeMolecule

        def _spy(*args, **kwargs):
            calls.append(1)
            return real_mmff(*args, **kwargs)

        monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", _spy)
        assert relieve_clash(mol, conf_id=0) is True
        return len(calls)

    assert _mmff_call_count(0.60) == 1, "0.60 A is below the C-H radius-scaled floor (0.667 A)"
    assert _mmff_call_count(0.35) == 1, "0.35 A must still trigger relief"


def _optimized(energy: float, changed: bool | None) -> Chem.Mol:
    """A converged, connectivity-valid mol, optionally marked stereo-changed."""
    mol = _embedded()
    mol.SetProp("Converged", "True")
    mol.SetProp("E_tot", str(energy))
    if changed is not None:
        mol.SetProp(STEREO_CHANGED_PROP, str(changed))
    return mol


class TestFiltersExcludeStereoChangedRecords:
    def test_filter_unique_optimized_drops_the_changed_record(self):
        kept = _optimized(-1.0, changed=False)
        dropped = _optimized(-2.0, changed=True)
        result = filter_unique_optimized([dropped, kept], rmsd_threshold=0.3)
        assert len(result) == 1, f"expected only the preserved record: {len(result)}"
        assert result[0].GetProp("E_tot") == "-1.0"

    def test_the_filter_reports_stereochemistry_as_the_drop_reason(self):
        """Not just that it dropped, but that it says why.

        Until 3.0.0 the two conformer filters returned a bare list, so
        ``ranking`` reported a stereo-changed species as "No structure
        converged" -- pointing the reader at the optimizer settings for a
        problem in the input's stereo definitions.
        """
        from Auto3D.domain.filtering import filter_conformers

        result = filter_conformers(
            [_optimized(-2.0, changed=True), _optimized(-1.0, changed=False)],
            rmsd_threshold=0.3,
        )
        assert result.dropped == {"stereochemistry": 1}
        assert result.reasons == ("stereochemistry",)

    def test_top_k_one_skips_the_changed_lowest_energy_record(self):
        """k=1 takes a fast path that bypasses the RMSD filters entirely."""
        dropped = _optimized(-2.0, changed=True)
        kept = _optimized(-1.0, changed=False)
        for mol, name in ((dropped, "probe_0_0"), (kept, "probe_0_1")):
            mol.SetProp("_Name", name)
        group = pd.DataFrame(
            {
                "names": ["probe", "probe"],
                "energies": [-2.0, -1.0],
                "mols": [dropped, kept],
            }
        )
        ranker = ConformerRanker(
            input_path="unused.sdf", out_path="unused_out.sdf", threshold=0.3, k=1
        )
        result = ranker.top_k(group, k=1)
        assert len(result) == 1
        assert result[0].GetProp("E_tot") == "-1.0", (
            "top_k returned the stereo-changed lowest-energy conformer"
        )

    def test_unmarked_records_still_survive_every_filter(self):
        """No regression for molecules that never went through the check."""
        mols = [_optimized(-1.0, changed=None), _optimized(-2.0, changed=None)]
        assert len(filter_unique_optimized(mols, rmsd_threshold=0.3)) == 2
