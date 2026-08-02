"""Unit tests for the slow tier's shared output assertions.

The helpers in ``tests/helpers_pipeline_output.py`` are the only thing standing
between the slow NNP tier and its previous state of asserting nothing. Their
own callers cannot run on a machine without a GPU budget and a model cache, so
the helpers are exercised here instead -- hermetically, on RDKit-embedded
molecules and synthetic SDF files, with no potential loaded and no weights
downloaded.

Two things are pinned for every assertion helper: that it **passes** on
well-formed input, and that it **fails** on each specific defect it claims to
catch. A helper only verified in the passing direction is exactly the "names a
guarantee it does not provide" defect these tests exist to prevent.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from Auto3D.results import WorkflowResult
from Auto3D.utils.energy import E_TOT_HARTREE_PROP, E_TOT_PROP
from tests.helpers_pipeline_output import (
    ATOMIC_ENERGY_HARTREE,
    assert_energy_is_plausible_hartree,
    assert_geometry_is_physical,
    assert_opt_geometry_output,
    assert_pipeline_output,
    base_molecule_id,
    expanded_copy,
    formula_from_smiles,
    formulas_from_sdf_file,
    formulas_from_smi_file,
    max_atom_displacement,
    molecular_formula,
    read_sdf_records,
    self_energy_estimate_hartree,
    write_perturbed_sdf,
)

FILES = Path(__file__).parent / "files"


def _embedded(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    """A hydrogen-complete molecule with one ETKDG conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _plausible_energy(mol: Chem.Mol) -> float:
    """An energy of the size a real potential would report for ``mol``."""
    return self_energy_estimate_hartree(mol) * 1.008


def _annotate_ranked(
    mol: Chem.Mol,
    name: str,
    *,
    energy: float | None = None,
    e_rel: float = 0.0,
    conformer_id: str = "0_0_0",
) -> Chem.Mol:
    """Stamp a molecule with the properties ``ConformerRanker`` writes."""
    out = Chem.Mol(mol)
    out.SetProp("_Name", name)
    value = _plausible_energy(mol) if energy is None else energy
    out.SetProp(E_TOT_PROP, repr(value))
    out.SetProp(E_TOT_HARTREE_PROP, repr(value))
    out.SetProp("E_rel(kcal/mol)", repr(e_rel))
    out.SetProp("Converged", "True")
    out.SetProp("ID", conformer_id)
    return out


def _annotate_optimized(mol: Chem.Mol, name: str) -> Chem.Mol:
    """Stamp a molecule with the properties ``batch_opt.optimizing`` writes."""
    out = Chem.Mol(mol)
    out.SetProp("_Name", name)
    value = _plausible_energy(mol)
    out.SetProp(E_TOT_PROP, repr(value))
    out.SetProp(E_TOT_HARTREE_PROP, repr(value))
    out.SetProp("fmax", "0.0074")
    out.SetProp("Converged", "True")
    out.SetProp("Dropped_Oscillating", "False")
    out.SetProp("Stereo_changed", "False")
    out.SetProp("ID", name)
    return out


def _write_sdf(path: Path, mols: list[Chem.Mol]) -> str:
    with Chem.SDWriter(str(path)) as writer:
        for mol in mols:
            writer.write(mol)
    return str(path)


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

class TestFormulaHelpers:
    """Formula comparison is the pipeline's chemical-identity check."""

    @pytest.mark.parametrize(
        "smiles", ["CC(CC)=O", "COC(/C=C/C)=O", "CC(CCCl)=O", "CCO", "c1ccccc1"]
    )
    def test_implicit_and_explicit_hydrogens_give_one_formula(self, smiles):
        """The whole comparison rests on this: inputs are implicit-H SMILES and
        outputs are explicit-H 3D structures, so the two must agree."""
        implicit = Chem.MolFromSmiles(smiles)
        explicit = Chem.AddHs(implicit)
        assert molecular_formula(implicit) == molecular_formula(explicit)
        assert molecular_formula(implicit) == formula_from_smiles(smiles)

    def test_formula_survives_an_sdf_round_trip(self, tmp_path):
        mol = _embedded("CC(CCCl)=O")
        path = _write_sdf(tmp_path / "m.sdf", [mol])
        (reread,) = read_sdf_records(path)
        assert molecular_formula(reread) == "C4H7ClO"

    def test_different_molecules_have_different_formulas(self):
        """Non-vacuity: the check would be worthless if everything matched."""
        assert formula_from_smiles("CCO") != formula_from_smiles("CCC")

    def test_unparseable_smiles_is_refused(self):
        with pytest.raises(ValueError, match="could not parse"):
            formula_from_smiles("this is not a smiles")

    def test_none_is_refused(self):
        with pytest.raises(ValueError, match="received None"):
            molecular_formula(None)

    def test_formulas_from_smi_file_reads_the_real_fixture(self):
        assert formulas_from_smi_file(FILES / "smiles2.smi") == {
            "smi2": "C4H8O",
            "smi3": "C5H8O2",
            "smi4": "C4H7ClO",
        }

    def test_formulas_from_sdf_file_reads_the_real_fixture(self):
        assert formulas_from_sdf_file(FILES / "example.sdf") == {
            "mol1": "C6H12O2",
            "mol2": "C11H9ClN4O2",
        }

    def test_empty_smi_file_is_refused_rather_than_returning_nothing(self, tmp_path):
        """An empty expectation map would make every downstream loop vacuous."""
        empty = tmp_path / "empty.smi"
        empty.write_text("\n\n")
        with pytest.raises(ValueError, match="no '<smiles> <id>' records"):
            formulas_from_smi_file(empty)

    def test_sdf_file_without_parseable_records_is_refused(self, tmp_path):
        """A wholly empty file makes RDKit raise on open, which is loud enough;
        the guard exists for a file that parses to nothing usable."""
        junk = tmp_path / "junk.sdf"
        junk.write_text("\n")
        with pytest.raises(ValueError, match="no parseable records"):
            formulas_from_sdf_file(junk)


class TestBaseMoleculeId:
    """Must normalize names exactly as the pipeline's reconciliation does."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("smi2", "smi2"),
            ("  smi2  ", "smi2"),
            ("smi2@taut1", "smi2"),
            ("smi2@taut12", "smi2"),
            ("KEY_2", "KEY_2"),
            ("KEY_2@taut0", "KEY_2"),
        ],
    )
    def test_strips_only_the_tautomer_suffix(self, name, expected):
        assert base_molecule_id(name) == expected


# ---------------------------------------------------------------------------
# read_sdf_records
# ---------------------------------------------------------------------------

class TestReadSdfRecords:
    """The check that kills 'return a path you never wrote'."""

    def test_accepts_a_real_multi_record_file(self):
        records = read_sdf_records(FILES / "DA.sdf")
        assert len(records) == 3
        assert [m.GetProp("_Name") for m in records] == [
            "diene",
            "dieneophile",
            "product",
        ]

    def test_missing_file_fails(self, tmp_path):
        with pytest.raises(AssertionError, match="no file exists"):
            read_sdf_records(tmp_path / "never_written.sdf")

    def test_empty_file_fails(self, tmp_path):
        empty = tmp_path / "empty.sdf"
        empty.write_text("")
        with pytest.raises(AssertionError, match="is empty"):
            read_sdf_records(empty)

    def test_file_of_whitespace_fails(self, tmp_path):
        """RDKit yields one unparseable record rather than zero records for a
        non-empty file, so this lands on the parse guard; either way the helper
        must refuse it instead of returning an empty list to iterate over."""
        blank = tmp_path / "blank.sdf"
        blank.write_text("\n")
        with pytest.raises(AssertionError, match="do not parse as SDF"):
            read_sdf_records(blank)

    def test_unparseable_record_fails(self, tmp_path):
        good = _annotate_ranked(_embedded("CCO"), "ethanol")
        path = tmp_path / "mixed.sdf"
        _write_sdf(path, [good])
        # Append a record RDKit cannot parse (bad element symbol in the atom block).
        path.write_text(path.read_text() + "junk\n\n\n  bogus\n$$$$\n")
        with pytest.raises(AssertionError, match="do not parse as SDF"):
            read_sdf_records(path)


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------

class TestSelfEnergyEstimate:
    """The order-of-magnitude reference the energy bound is measured against."""

    def test_implicit_and_explicit_hydrogens_agree(self):
        implicit = Chem.MolFromSmiles("CC(CC)=O")
        assert self_energy_estimate_hartree(implicit) == pytest.approx(
            self_energy_estimate_hartree(Chem.AddHs(implicit))
        )

    def test_estimate_is_within_one_percent_of_real_reference_energies(self):
        """Pinned against the three DFT-level energies checked into DA.sdf.

        This is what justifies the 2x window in
        ``assert_energy_is_plausible_hartree``: the estimate is good to well
        under a percent, so the window has two orders of magnitude of headroom
        over its own error while still being 13x tighter than an eV/Hartree
        mix-up.
        """
        records = read_sdf_records(FILES / "DA.sdf")
        assert len(records) == 3
        for mol in records:
            reference = float(mol.GetProp(E_TOT_PROP))
            estimate = self_energy_estimate_hartree(mol)
            assert abs(estimate - reference) / abs(reference) < 0.01

    def test_is_negative_and_scales_with_size(self):
        small = self_energy_estimate_hartree(Chem.MolFromSmiles("CCO"))
        large = self_energy_estimate_hartree(Chem.MolFromSmiles("CCCCCCCCO"))
        assert small < 0
        assert large < small

    def test_unknown_element_raises_rather_than_skipping_the_check(self):
        """Silently returning 'no opinion' for an unsupported element is the
        vacuous-assertion trap; the helper must refuse instead."""
        assert 26 not in ATOMIC_ENERGY_HARTREE
        with pytest.raises(KeyError, match="no reference atomic energy"):
            self_energy_estimate_hartree(Chem.MolFromSmiles("[Fe]"))


class TestAssertEnergyIsPlausible:
    def test_accepts_real_reference_energies(self):
        for mol in read_sdf_records(FILES / "DA.sdf"):
            mol.SetProp(E_TOT_HARTREE_PROP, mol.GetProp(E_TOT_PROP))
            energy = assert_energy_is_plausible_hartree(mol)
            assert energy == pytest.approx(float(mol.GetProp(E_TOT_PROP)))

    def test_rejects_an_energy_written_in_ev(self):
        """The regression this project just fixed: eV under a Hartree name."""
        mol = _embedded("CCO")
        in_ev = _plausible_energy(mol) * 27.211386245988
        mol.SetProp(E_TOT_PROP, repr(in_ev))
        mol.SetProp(E_TOT_HARTREE_PROP, repr(in_ev))
        with pytest.raises(AssertionError, match="outside"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_a_missing_energy(self):
        mol = _embedded("CCO")
        with pytest.raises(AssertionError, match=f"no {E_TOT_PROP} property"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_a_positive_energy(self):
        mol = _embedded("CCO")
        mol.SetProp(E_TOT_PROP, "12.5")
        mol.SetProp(E_TOT_HARTREE_PROP, "12.5")
        with pytest.raises(AssertionError, match="must be negative"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_a_nan_energy(self):
        mol = _embedded("CCO")
        mol.SetProp(E_TOT_PROP, "nan")
        mol.SetProp(E_TOT_HARTREE_PROP, "nan")
        with pytest.raises(AssertionError, match="is nan"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_an_atomization_scale_energy(self):
        """Far too small to be a total energy for this molecule."""
        mol = _embedded("CCO")
        mol.SetProp(E_TOT_PROP, "-1.75")
        mol.SetProp(E_TOT_HARTREE_PROP, "-1.75")
        with pytest.raises(AssertionError, match="outside"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_a_missing_unit_labeled_sibling(self):
        mol = _embedded("CCO")
        mol.SetProp(E_TOT_PROP, repr(_plausible_energy(mol)))
        with pytest.raises(AssertionError, match="unit-labeled"):
            assert_energy_is_plausible_hartree(mol)

    def test_rejects_a_sibling_that_disagrees(self):
        """Catches a second unit conversion applied to only one of the pair."""
        mol = _embedded("CCO")
        value = _plausible_energy(mol)
        mol.SetProp(E_TOT_PROP, repr(value))
        mol.SetProp(E_TOT_HARTREE_PROP, repr(value / 27.211386245988))
        with pytest.raises(AssertionError, match="disagrees with"):
            assert_energy_is_plausible_hartree(mol)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestAssertGeometryIsPhysical:
    def test_accepts_real_structures(self):
        for mol in read_sdf_records(FILES / "DA.sdf"):
            assert_geometry_is_physical(mol)

    def test_rejects_a_collapsed_structure(self):
        mol = _embedded("CCO")
        conformer = mol.GetConformer()
        for idx in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(idx, [0.0, 0.0, 0.0])
        with pytest.raises(AssertionError, match="collapsed"):
            assert_geometry_is_physical(mol)

    def test_rejects_non_finite_coordinates(self):
        mol = _embedded("CCO")
        mol.GetConformer().SetAtomPosition(0, [float("nan"), 0.0, 0.0])
        with pytest.raises(AssertionError, match="non-finite"):
            assert_geometry_is_physical(mol)

    def test_rejects_a_record_carrying_more_than_one_conformer(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=7)
        with pytest.raises(AssertionError, match="exactly one conformer"):
            assert_geometry_is_physical(mol)

    def test_two_atoms_just_over_the_floor_are_accepted(self):
        """The floor is 0.5 A, well below the shortest real bond (H-H, 0.74 A),
        so it never fires on a genuine structure."""
        mol = Chem.MolFromMolBlock(
            "\n     RDKit          3D\n\n"
            "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 H   0  0\n"
            "    0.7400    0.0000    0.0000 H   0  0\n"
            "  1  2  1  0\n"
            "M  END\n",
            removeHs=False,
        )
        assert_geometry_is_physical(mol, min_separation=0.5)
        with pytest.raises(AssertionError, match="collapsed"):
            assert_geometry_is_physical(mol, min_separation=0.8)


class TestMaxAtomDisplacement:
    def test_identical_conformations_have_zero_displacement(self):
        mol = _embedded("CCO")
        assert max_atom_displacement(mol, Chem.Mol(mol)) == pytest.approx(0.0)

    def test_measures_a_known_rigid_shift(self):
        mol = _embedded("CCO")
        shifted = Chem.Mol(mol)
        conformer = shifted.GetConformer()
        positions = conformer.GetPositions()
        for idx in range(shifted.GetNumAtoms()):
            conformer.SetAtomPosition(idx, (positions[idx] + [0.3, 0.4, 0.0]).tolist())
        assert max_atom_displacement(mol, shifted) == pytest.approx(0.5, abs=1e-9)

    def test_reports_the_largest_single_atom_move(self):
        mol = _embedded("CCO")
        moved = Chem.Mol(mol)
        position = np.array(moved.GetConformer().GetPositions()[0])
        moved.GetConformer().SetAtomPosition(0, (position + [0.0, 0.0, 1.25]).tolist())
        assert max_atom_displacement(mol, moved) == pytest.approx(1.25)

    def test_refuses_a_different_molecule(self):
        with pytest.raises(ValueError, match="atom count differs"):
            max_atom_displacement(_embedded("CCO"), _embedded("CCCO"))

    def test_refuses_a_reordered_molecule(self):
        """Same atom count, different element order: the number would be a lie."""
        a = _embedded("CCO")
        b = _embedded("CCO")
        # Swapping the two carbons would be invisible to an element-order
        # check, so move a hydrogen into a heavy atom's slot instead.
        reordered = Chem.RenumberAtoms(
            b, [0, 1, b.GetNumAtoms() - 1, *range(2, b.GetNumAtoms() - 1)]
        )
        with pytest.raises(ValueError, match="atomic numbers differ"):
            max_atom_displacement(a, reordered)


class TestExpandedCopy:
    """The deterministic perturbation that makes 'the optimizer moved it'
    assertable on an input that is already at a minimum."""

    def test_scales_every_bond_by_the_factor(self):
        mol = read_sdf_records(FILES / "DA.sdf")[2]
        expanded = expanded_copy(mol, 1.05)
        original = mol.GetConformer().GetPositions()
        moved = expanded.GetConformer().GetPositions()
        assert mol.GetNumBonds() > 0
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            before = np.linalg.norm(original[i] - original[j])
            after = np.linalg.norm(moved[i] - moved[j])
            assert after == pytest.approx(before * 1.05, rel=1e-9)

    def test_preserves_the_centroid(self):
        """So the displacement measured afterwards is relaxation, not a shift."""
        mol = read_sdf_records(FILES / "DA.sdf")[0]
        expanded = expanded_copy(mol, 1.05)
        before = mol.GetConformer().GetPositions().mean(axis=0)
        after = expanded.GetConformer().GetPositions().mean(axis=0)
        assert after == pytest.approx(before, abs=1e-9)

    def test_leaves_the_original_untouched(self):
        mol = read_sdf_records(FILES / "DA.sdf")[0]
        before = mol.GetConformer().GetPositions().copy()
        expanded_copy(mol, 1.20)
        assert mol.GetConformer().GetPositions() == pytest.approx(before)

    def test_the_perturbation_clears_the_relaxation_floor_it_is_paired_with(self):
        """Every atom of every DA.sdf molecule moves further than the 0.01 A
        floor ``test_thermo`` asserts afterwards, so the perturbation is what
        makes that assertion meaningful rather than lucky."""
        for mol in read_sdf_records(FILES / "DA.sdf"):
            expanded = expanded_copy(mol, 1.05)
            positions = mol.GetConformer().GetPositions()
            shifted = expanded.GetConformer().GetPositions()
            per_atom = np.linalg.norm(shifted - positions, axis=1)
            assert per_atom.min() > 0.04
            assert max_atom_displacement(mol, expanded) > 0.1

    def test_a_factor_of_one_is_a_no_op(self):
        mol = read_sdf_records(FILES / "DA.sdf")[0]
        assert max_atom_displacement(mol, expanded_copy(mol, 1.0)) == pytest.approx(
            0.0, abs=1e-9
        )


class TestWritePerturbedSdf:
    """The staging step the opt_geometry tests depend on."""

    def test_writes_every_record_with_its_name(self, tmp_path):
        path, mols = write_perturbed_sdf(
            FILES / "DA.sdf", tmp_path / "DA.sdf", 1.05
        )
        assert Path(path) == tmp_path / "DA.sdf"
        assert [m.GetProp("_Name") for m in mols] == [
            "diene",
            "dieneophile",
            "product",
        ]

    def test_strips_stale_properties(self, tmp_path):
        """DA.sdf ships with E_tot/fmax/Converged from an earlier run. If those
        survived, an optimizer that wrote nothing would still emit output
        carrying a plausible energy and pass the energy check."""
        before = read_sdf_records(FILES / "DA.sdf")
        assert all(mol.HasProp(E_TOT_PROP) for mol in before)
        _, after = write_perturbed_sdf(FILES / "DA.sdf", tmp_path / "DA.sdf", 1.05)
        assert after
        for mol in after:
            assert list(mol.GetPropNames()) == []

    def test_perturbation_survives_the_sdf_round_trip(self, tmp_path):
        """SDF stores 4 decimal places; the displacement must still clear the
        0.01 A floor asserted after optimization, with room to spare."""
        originals = read_sdf_records(FILES / "DA.sdf")
        _, staged = write_perturbed_sdf(FILES / "DA.sdf", tmp_path / "DA.sdf", 1.05)
        assert len(staged) == len(originals) == 3
        for original, moved in zip(originals, staged, strict=True):
            assert max_atom_displacement(original, moved) > 0.1

    def test_returned_molecules_match_the_file_on_disk(self, tmp_path):
        """The returned list must be what the optimizer will read, to 4dp, not
        the full-precision in-memory copies."""
        path, returned = write_perturbed_sdf(
            FILES / "DA.sdf", tmp_path / "DA.sdf", 1.05
        )
        from_disk = read_sdf_records(path)
        for a, b in zip(returned, from_disk, strict=True):
            assert max_atom_displacement(a, b) == pytest.approx(0.0, abs=1e-12)

    def test_a_factor_of_one_stages_an_unmoved_copy(self, tmp_path):
        """Non-vacuity for the test above: the displacement comes from the
        factor, not from the round trip itself."""
        originals = read_sdf_records(FILES / "DA.sdf")
        _, staged = write_perturbed_sdf(FILES / "DA.sdf", tmp_path / "DA.sdf", 1.0)
        for original, moved in zip(originals, staged, strict=True):
            assert max_atom_displacement(original, moved) < 1e-3


# ---------------------------------------------------------------------------
# assert_pipeline_output
# ---------------------------------------------------------------------------

FORMULAS = {"smi2": "C4H8O", "smi3": "C5H8O2", "smi4": "C4H7ClO"}
SMILES_BY_ID = {"smi2": "CC(CC)=O", "smi3": "COC(/C=C/C)=O", "smi4": "CC(CCCl)=O"}


def _good_output(tmp_path, *, ids=("smi2", "smi3", "smi4"), name="out.sdf"):
    """A synthetic output SDF satisfying every contract the pipeline promises."""
    mols = [
        _annotate_ranked(_embedded(SMILES_BY_ID[i]), i, conformer_id=f"{n}_0_0")
        for n, i in enumerate(ids)
    ]
    path = _write_sdf(tmp_path / name, mols)
    return WorkflowResult(path), mols


class TestAssertPipelineOutput:
    def test_accepts_a_well_formed_run(self, tmp_path):
        result, _ = _good_output(tmp_path)
        records = assert_pipeline_output(result, formula_by_id=FORMULAS, k=1)
        assert len(records) == 3

    def test_accepts_a_partial_run_whose_losses_are_reported(self, tmp_path):
        """The reconciliation contract: absent is fine, absent *and silent* is not."""
        result, _ = _good_output(tmp_path, ids=("smi2", "smi3"))
        reported = WorkflowResult(str(result), failures=["smi4"])
        assert_pipeline_output(reported, formula_by_id=FORMULAS, k=1)

    def test_rejects_a_molecule_that_vanished_without_a_report(self, tmp_path):
        result, _ = _good_output(tmp_path, ids=("smi2", "smi3"))
        with pytest.raises(AssertionError, match="absent from the output"):
            assert_pipeline_output(result, formula_by_id=FORMULAS, k=1)

    def test_rejects_a_run_that_produced_nothing_but_claims_total_failure(
        self, tmp_path
    ):
        empty = tmp_path / "nothing.sdf"
        empty.write_text("")
        result = WorkflowResult(str(empty), failures=list(FORMULAS))
        with pytest.raises(AssertionError, match="is empty"):
            assert_pipeline_output(result, formula_by_id=FORMULAS, k=1)

    def test_rejects_an_id_that_was_never_in_the_input(self, tmp_path):
        result, _ = _good_output(tmp_path)
        with pytest.raises(AssertionError, match="never in the input"):
            assert_pipeline_output(
                result, formula_by_id={"smi2": "C4H8O", "smi3": "C5H8O2"}, k=1
            )

    def test_rejects_an_id_both_produced_and_reported_failed(self, tmp_path):
        result, _ = _good_output(tmp_path)
        contradictory = WorkflowResult(str(result), failures=["smi2"])
        with pytest.raises(AssertionError, match="reported as failures yet present"):
            assert_pipeline_output(contradictory, formula_by_id=FORMULAS, k=1)

    def test_rejects_a_changed_chemical_identity(self, tmp_path):
        """The mutation where the pipeline emits the wrong molecule."""
        wrong = _annotate_ranked(_embedded("CCCCCC"), "smi2")
        others = [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i) for i in ("smi3", "smi4")
        ]
        path = _write_sdf(tmp_path / "wrong.sdf", [wrong, *others])
        with pytest.raises(AssertionError, match="chemical identity changed"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_rejects_more_conformers_than_k_allows(self, tmp_path):
        mols = [
            _annotate_ranked(_embedded(SMILES_BY_ID["smi2"], seed=s), "smi2")
            for s in (1, 2)
        ]
        mols += [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i) for i in ("smi3", "smi4")
        ]
        path = _write_sdf(tmp_path / "extra.sdf", mols)
        with pytest.raises(AssertionError, match="conformers written for k=1"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_rejects_an_energy_window_violation(self, tmp_path):
        mols = [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i, e_rel=e)
            for i, e in zip(("smi2", "smi3", "smi4"), (0.0, 1.4, 3.7), strict=True)
        ]
        path = _write_sdf(tmp_path / "window.sdf", mols)
        result = WorkflowResult(path)
        assert_pipeline_output(result, formula_by_id=FORMULAS, window=4.0)
        with pytest.raises(AssertionError, match="exceeds the 2.0 kcal/mol window"):
            assert_pipeline_output(result, formula_by_id=FORMULAS, window=2.0)

    def test_rejects_a_negative_relative_energy(self, tmp_path):
        mols = [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i, e_rel=e)
            for i, e in zip(("smi2", "smi3", "smi4"), (0.0, 0.0, -0.9), strict=True)
        ]
        path = _write_sdf(tmp_path / "negative.sdf", mols)
        with pytest.raises(AssertionError, match="is negative"):
            assert_pipeline_output(
                WorkflowResult(path), formula_by_id=FORMULAS, window=5.0
            )

    def test_rejects_a_nonzero_relative_energy_when_k_is_one(self, tmp_path):
        mols = [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i, e_rel=e)
            for i, e in zip(("smi2", "smi3", "smi4"), (0.0, 0.0, 0.6), strict=True)
        ]
        path = _write_sdf(tmp_path / "krel.sdf", mols)
        with pytest.raises(AssertionError, match="E_rel must be 0"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_rejects_output_that_kept_the_working_ev_relative_energy(self, tmp_path):
        result, mols = _good_output(tmp_path)
        mols[0].SetProp("E_rel(eV)", "0.0")
        path = _write_sdf(tmp_path / "ev.sdf", mols)
        with pytest.raises(AssertionError, match=r"E_rel\(eV\) survived"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_rejects_an_unconverged_structure(self, tmp_path):
        result, mols = _good_output(tmp_path)
        mols[1].SetProp("Converged", "False")
        path = _write_sdf(tmp_path / "unconverged.sdf", mols)
        with pytest.raises(AssertionError, match="only emit converged structures"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_rejects_an_implausible_energy(self, tmp_path):
        """The 'arbitrary E_tot' half of the mutation this tier missed."""
        mols = [
            _annotate_ranked(_embedded(SMILES_BY_ID[i]), i) for i in ("smi3", "smi4")
        ]
        mols.insert(0, _annotate_ranked(_embedded(SMILES_BY_ID["smi2"]), "smi2",
                                        energy=-1.0))
        path = _write_sdf(tmp_path / "badenergy.sdf", mols)
        with pytest.raises(AssertionError, match="outside"):
            assert_pipeline_output(WorkflowResult(path), formula_by_id=FORMULAS, k=1)

    def test_refuses_an_empty_expectation_map(self, tmp_path):
        """The core non-vacuity guard: no expectations means no test."""
        result, _ = _good_output(tmp_path)
        with pytest.raises(AssertionError, match="no expected molecules"):
            assert_pipeline_output(result, formula_by_id={}, k=1)

    def test_refuses_both_selectors_at_once(self, tmp_path):
        result, _ = _good_output(tmp_path)
        with pytest.raises(AssertionError, match="alternative selectors"):
            assert_pipeline_output(result, formula_by_id=FORMULAS, k=1, window=2.0)

    def test_cross_checks_the_workflow_result_counters(self, tmp_path):
        """WorkflowResult computes its counts from the file, so agreement here
        also pins that lazy computation."""
        result, _ = _good_output(tmp_path)
        assert_pipeline_output(result, formula_by_id=FORMULAS, k=1)
        assert result.n_molecules == 3
        assert result.n_conformers == 3


# ---------------------------------------------------------------------------
# assert_opt_geometry_output
# ---------------------------------------------------------------------------

def _optimized_da(tmp_path, *, factor=1.05, name="DA_opt.sdf"):
    """Inputs perturbed off their minimum, plus a plausible 'optimized' result."""
    originals = read_sdf_records(FILES / "DA.sdf")
    inputs = [expanded_copy(mol, factor) for mol in originals]
    outputs = [_annotate_optimized(mol, mol.GetProp("_Name")) for mol in originals]
    return _write_sdf(tmp_path / name, outputs), inputs


class TestAssertOptGeometryOutput:
    def test_accepts_a_real_relaxation(self, tmp_path):
        path, inputs = _optimized_da(tmp_path)
        records = assert_opt_geometry_output(
            path, input_mols=inputs, moved_at_least=0.01
        )
        assert len(records) == 3

    def test_rejects_a_path_that_was_never_written(self, tmp_path):
        """The exact mutation the old tests swallowed: FileNotFoundError is an
        OSError, and their cleanup caught OSError."""
        _, inputs = _optimized_da(tmp_path)
        with pytest.raises(AssertionError, match="no file exists"):
            assert_opt_geometry_output(
                str(tmp_path / "never.sdf"), input_mols=inputs, moved_at_least=0.01
            )

    def test_rejects_an_unmoved_geometry(self, tmp_path):
        """Handing back the input unchanged must fail."""
        originals = read_sdf_records(FILES / "DA.sdf")
        outputs = [_annotate_optimized(mol, mol.GetProp("_Name")) for mol in originals]
        path = _write_sdf(tmp_path / "noop.sdf", outputs)
        with pytest.raises(AssertionError, match="moved no atom further"):
            assert_opt_geometry_output(
                path, input_mols=originals, moved_at_least=0.01
            )

    def test_rejects_a_dropped_record(self, tmp_path):
        originals = read_sdf_records(FILES / "DA.sdf")
        inputs = [expanded_copy(mol, 1.05) for mol in originals]
        outputs = [_annotate_optimized(m, m.GetProp("_Name")) for m in originals[:2]]
        path = _write_sdf(tmp_path / "short.sdf", outputs)
        with pytest.raises(AssertionError, match="returned 2 structures for 3"):
            assert_opt_geometry_output(path, input_mols=inputs, moved_at_least=0.01)

    def test_rejects_reordered_records(self, tmp_path):
        originals = read_sdf_records(FILES / "DA.sdf")
        inputs = [expanded_copy(mol, 1.05) for mol in originals]
        outputs = [_annotate_optimized(m, m.GetProp("_Name")) for m in originals]
        path = _write_sdf(tmp_path / "shuffled.sdf", list(reversed(outputs)))
        with pytest.raises(AssertionError, match="does not line up with input"):
            assert_opt_geometry_output(path, input_mols=inputs, moved_at_least=0.01)

    @pytest.mark.parametrize(
        ("prop", "match"),
        [
            ("Dropped_Oscillating", "lacks Dropped_Oscillating"),
            ("Stereo_changed", "lacks Stereo_changed"),
            # The energy check runs first and refuses this one on its own
            # terms, which is the same verdict by a more specific route.
            (E_TOT_HARTREE_PROP, "unit-labeled"),
        ],
    )
    def test_rejects_output_missing_a_property_only_the_optimizer_writes(
        self, tmp_path, prop, match
    ):
        """``DA.sdf`` already carries E_tot/fmax/Converged, so these three are
        what tell a genuine run apart from the input file handed back."""
        originals = read_sdf_records(FILES / "DA.sdf")
        inputs = [expanded_copy(mol, 1.05) for mol in originals]
        outputs = [_annotate_optimized(m, m.GetProp("_Name")) for m in originals]
        for mol in outputs:
            mol.ClearProp(prop)
        path = _write_sdf(tmp_path / "stripped.sdf", outputs)
        with pytest.raises(AssertionError, match=match):
            assert_opt_geometry_output(path, input_mols=inputs, moved_at_least=0.01)

    def test_rejects_a_changed_formula(self, tmp_path):
        originals = read_sdf_records(FILES / "DA.sdf")
        inputs = [expanded_copy(mol, 1.05) for mol in originals]
        outputs = [_annotate_optimized(m, m.GetProp("_Name")) for m in originals]
        outputs[0] = _annotate_optimized(_embedded("CCCCCCCC"), "diene")
        path = _write_sdf(tmp_path / "swapped.sdf", outputs)
        with pytest.raises(AssertionError, match="chemical identity changed"):
            assert_opt_geometry_output(path, input_mols=inputs, moved_at_least=0.01)

    def test_refuses_an_empty_input_list(self, tmp_path):
        path, _ = _optimized_da(tmp_path)
        with pytest.raises(AssertionError, match="no input molecules"):
            assert_opt_geometry_output(path, input_mols=[], moved_at_least=0.01)

    def test_rejects_a_non_finite_fmax(self, tmp_path):
        originals = read_sdf_records(FILES / "DA.sdf")
        inputs = [expanded_copy(mol, 1.05) for mol in originals]
        outputs = [_annotate_optimized(m, m.GetProp("_Name")) for m in originals]
        outputs[1].SetProp("fmax", "nan")
        path = _write_sdf(tmp_path / "badfmax.sdf", outputs)
        with pytest.raises(AssertionError, match="not a force magnitude"):
            assert_opt_geometry_output(path, input_mols=inputs, moved_at_least=0.01)


def test_formula_helper_agrees_with_rdkit_directly():
    """Guard against the helper quietly becoming a no-op wrapper."""
    mol = _embedded("CC(CCCl)=O")
    assert molecular_formula(mol) == CalcMolFormula(mol)
    assert math.isfinite(self_energy_estimate_hartree(mol))
