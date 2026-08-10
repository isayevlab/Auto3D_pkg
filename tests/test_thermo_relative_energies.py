"""``calc_thermo`` must publish a relative energy for the conformers it keeps.

``do_mol_thermo`` clears ``E_rel(kcal/mol)`` because the value it inherits was
computed by ``ranking.run`` from the *pre-relaxation* ``E_tot``, and the
relaxation replaces that number. Clearing is right per record -- a relative
energy must not outlive the absolute one it derives from -- but on its own it
left the output inverted: the stationary-point gate ``continue``s before
``do_mol_thermo`` runs, so the property survived on exactly the records a user
must discard and was absent on the good ones.

``E_rel(kcal/mol)`` is documented output (``README.md``,
``docs/source/usage.rst``: "Energy relative to the best conformer of that
molecule"), so it has to come back -- recomputed against the relaxed energies,
over the records that are actually comparable.

Nothing here loads a neural network potential.
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.utils.energy import (
    E_REL_KCAL_PROP,
    ev2kcalpermol,
    set_e_tot_from_ev,
    set_relative_energies,
)


def _conformer(name: str, energy_ev: float, smiles: str = "CCO") -> Chem.Mol:
    """A record shaped like one that survived ``do_mol_thermo``."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    set_e_tot_from_ev(mol, energy_ev)
    return mol


class TestGrouping:
    def test_each_group_is_referenced_to_its_own_minimum(self):
        mols = [
            _conformer("mol_a", -10.0),
            _conformer("mol_a", -9.5),
            _conformer("mol_b", -20.0),
            _conformer("mol_b", -19.0),
        ]

        set_relative_energies(mols)

        rel = [float(m.GetProp(E_REL_KCAL_PROP)) for m in mols]
        assert rel[0] == pytest.approx(0.0)
        assert rel[1] == pytest.approx(0.5 * ev2kcalpermol)
        assert rel[2] == pytest.approx(0.0), "group b used group a's reference"
        assert rel[3] == pytest.approx(1.0 * ev2kcalpermol)

    def test_the_minimum_is_chosen_not_the_first_record(self):
        """Order must not decide the reference; the relaxation can reorder."""
        mols = [_conformer("m", -9.0), _conformer("m", -11.0)]

        set_relative_energies(mols)

        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(
            2.0 * ev2kcalpermol
        )

    def test_relative_energies_are_never_negative(self):
        mols = [_conformer("m", e) for e in (-8.0, -12.0, -10.0)]

        set_relative_energies(mols)

        assert all(float(m.GetProp(E_REL_KCAL_PROP)) >= 0.0 for m in mols)

    def test_a_lone_conformer_is_its_own_reference(self):
        mols = [_conformer("only", -5.0)]

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)

    def test_names_are_not_stripped_the_way_ranking_strips_them(self):
        """``ranking.species_id`` is not idempotent and must not be reused here.

        In a ``main()`` output ``_Name`` is *already* the group key, so applying
        ``name.rsplit("_", 2)[0]`` a second time turns ``aspirin_analog_3`` into
        ``aspirin`` and merges compounds that merely share a prefix.
        """
        mols = [
            _conformer("aspirin_analog_3", -10.0),
            _conformer("aspirin", -30.0),
        ]

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0), (
            "two distinct compounds were merged into one relative-energy group"
        )
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)


class TestGuards:
    def test_a_group_of_different_compounds_gets_no_relative_energy(self):
        """``calc_thermo`` accepts any SDF, including one with reused titles.

        Subtracting the energies of two different compounds produces a number
        with no meaning, and it would look exactly like a conformational
        preference. Withholding is the only safe answer.
        """
        mols = [_conformer("shared", -10.0, "CCO"), _conformer("shared", -40.0, "c1ccccc1")]

        set_relative_energies(mols)

        assert not any(m.HasProp(E_REL_KCAL_PROP) for m in mols)

    def test_records_without_a_name_are_left_alone(self):
        """An untitled SDF must not become one giant group."""
        mols = [_conformer("", -10.0), _conformer("", -40.0, "c1ccccc1")]

        set_relative_energies(mols)

        assert not any(m.HasProp(E_REL_KCAL_PROP) for m in mols)

    def test_a_record_with_no_energy_is_skipped_without_taking_the_group_down(self):
        mols = [_conformer("m", -10.0), _conformer("m", -9.0)]
        mols[1].ClearProp("E_tot")
        mols[1].ClearProp("E_tot(Hartree)")

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert not mols[1].HasProp(E_REL_KCAL_PROP)

    def test_a_stale_value_is_replaced_not_kept(self):
        mols = [_conformer("m", -10.0), _conformer("m", -9.0)]
        for m in mols:
            m.SetProp(E_REL_KCAL_PROP, "999.0")

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(
            1.0 * ev2kcalpermol
        )

    def test_a_stale_value_is_cleared_when_the_group_is_refused(self):
        """Refusing to compute must not leave the old number in place."""
        mols = [_conformer("shared", -10.0, "CCO"), _conformer("shared", -40.0, "c1ccccc1")]
        for m in mols:
            m.SetProp(E_REL_KCAL_PROP, "999.0")

        set_relative_energies(mols)

        assert not any(m.HasProp(E_REL_KCAL_PROP) for m in mols), (
            "a refused group kept a relative energy derived from an older run"
        )


class TestCalcThermoWiring:
    """The helpers existing is not the fix; `calc_thermo` has to call both."""

    def test_failed_records_lose_their_inherited_relative_energy(self):
        """The inversion this change exists to close.

        The stationary-point gate ``continue``s before ``do_mol_thermo``, so a
        failed record never had its inherited ``E_rel(kcal/mol)`` cleared and
        never gets a recomputed one. Left alone, the documented property would
        survive only on the records the docs tell a reader to discard.
        """
        from Auto3D.utils.energy import clear_relative_energies

        failed = [_conformer("mol_a", -10.0), _conformer("mol_b", -3.0)]
        for m in failed:
            m.SetProp(E_REL_KCAL_PROP, "1.5")

        clear_relative_energies(failed)

        assert not any(m.HasProp(E_REL_KCAL_PROP) for m in failed)

    def test_clearing_a_record_that_has_none_is_a_no_op(self):
        from Auto3D.utils.energy import clear_relative_energies

        mols = [_conformer("m", -1.0)]
        clear_relative_energies(mols)
        assert not mols[0].HasProp(E_REL_KCAL_PROP)

    def test_calc_thermo_runs_both_halves_after_the_loop(self):
        """Wiring guard: the helpers must be reached from ``calc_thermo``.

        A source check rather than a behavioural one because the surrounding
        loop needs a neural network potential, which the fast tier does not
        load. It is narrow on purpose -- it pins that the calls exist and which
        collection each is handed, since swapping those two arguments would
        publish relative energies for the failures and strip them from the
        successes, restoring the exact defect being fixed.
        """
        import inspect

        import Auto3D.ASE.thermo as thermo_mod

        source = inspect.getsource(thermo_mod.calc_thermo)
        assert "set_relative_energies(out_mols)" in source, (
            "calc_thermo no longer recomputes relative energies after the loop, "
            "so its output carries none on the records that succeeded"
        )
        assert "clear_relative_energies(mols_failed)" in source, (
            "calc_thermo no longer strips the stale relative energy from failed "
            "records, so the property survives only where it is meaningless"
        )
