"""Post-optimization stereochemistry validation (C9).

An optimization that inverts a stereocenter or rotates through a double bond
produces a molecule of different chemical identity than its title. check_connectivity
compares interatomic distances against UFF radii and is stereo-blind, so nothing
caught it. These tests pin the detector and the three filters that act on it.
"""
from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.filtering import filter_unique_optimized
from Auto3D.ranking import ConformerRanker
from Auto3D.utils.chemistry import filter_unique
from Auto3D.utils.stereo_check import (
    STEREO_CHANGED_PROP,
    apply_optimized_coords,
    stereo_descriptors_from_3d,
    stereo_preserved,
)


def _embedded(smiles: str = "C/C=C/C[C@H](O)Cl") -> Chem.Mol:
    """A molecule carrying both a tetrahedral center and a defined C=C."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
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
        [conf.GetAtomPosition(i).x + 0.01, conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
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

    def test_filter_unique_drops_the_changed_record(self):
        kept = _optimized(-1.0, changed=False)
        dropped = _optimized(-2.0, changed=True)
        result = filter_unique([dropped, kept], crit=0.3)
        assert len(result) == 1, f"expected only the preserved record: {len(result)}"
        assert result[0].GetProp("E_tot") == "-1.0"

    def test_top_k_one_skips_the_changed_lowest_energy_record(self):
        """k=1 takes a fast path that bypasses the RMSD filters entirely."""
        dropped = _optimized(-2.0, changed=True)
        kept = _optimized(-1.0, changed=False)
        for mol, name in ((dropped, "probe_0_0"), (kept, "probe_0_1")):
            mol.SetProp("_Name", name)
        group = pd.DataFrame({
            "names": ["probe", "probe"],
            "energies": [-2.0, -1.0],
            "mols": [dropped, kept],
        })
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
        assert len(filter_unique_optimized(mols, rmsd_threshold=0.3)) >= 1
        assert len(filter_unique(mols, crit=0.3)) >= 1
