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

from Auto3D.domain.energy import (
    E_REL_KCAL_PROP,
    e_tot_hartree,
    set_e_tot_from_ev,
    set_relative_energies,
)
from Auto3D.foundation.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL


def _conformer(name: str, energy_ev: float, smiles: str = "CCO") -> Chem.Mol:
    """A record shaped like one that survived ``do_mol_thermo``."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    set_e_tot_from_ev(mol, energy_ev)
    return mol


def _record(name: str, e_tot_hartree_value: float) -> Chem.Mol:
    """A record with ``E_tot`` stored directly in Hartree, no unit conversion."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    mol.SetProp("E_tot", repr(e_tot_hartree_value))
    return mol


def test_relative_energies_are_formed_in_hartree_before_the_unit_conversion():
    ref = -2000.0
    low = _record("same", ref)
    high = _record("same", ref + 1.0 / HARTREE_TO_KCAL_PER_MOL)
    set_relative_energies([low, high])
    expected = (e_tot_hartree(high) - e_tot_hartree(low)) * HARTREE_TO_KCAL_PER_MOL
    assert float(high.GetProp("E_rel(kcal/mol)")) == expected
    assert float(high.GetProp("E_rel(kcal/mol)")) == pytest.approx(1.0, abs=1e-9)
    assert float(low.GetProp("E_rel(kcal/mol)")) == 0.0


def _finalize_thermo_records_source() -> str:
    """Confirm ``calc_thermo`` delegates to ``_finalize_thermo_records``, then return its source.

    Shared by the three wiring guards below so each checks both halves of the
    chain -- that ``calc_thermo`` still calls the helper, and that the helper
    still does the work -- without repeating the delegation assertion three
    times.
    """
    import inspect

    import Auto3D.entry.ASE.thermo.driver as thermo_mod

    assert "_finalize_thermo_records(" in inspect.getsource(thermo_mod.calc_thermo), (
        "calc_thermo no longer calls _finalize_thermo_records after the loop, "
        "so neither the counts log nor the relative-energy pass runs at all"
    )
    return inspect.getsource(thermo_mod._finalize_thermo_records)


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
        assert rel[1] == pytest.approx(0.5 * EV_TO_KCAL_PER_MOL)
        assert rel[2] == pytest.approx(0.0), "group b used group a's reference"
        assert rel[3] == pytest.approx(1.0 * EV_TO_KCAL_PER_MOL)

    def test_the_minimum_is_chosen_not_the_first_record(self):
        """Order must not decide the reference; the relaxation can reorder."""
        mols = [_conformer("m", -9.0), _conformer("m", -11.0)]

        set_relative_energies(mols)

        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(2.0 * EV_TO_KCAL_PER_MOL)

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
        ``species_id``'s strip (which only fires on a name ending in two
        trailing integer groups, N-m4) a second time turns e.g. ``foo_1_2``
        into ``foo`` and would merge compounds that merely share a prefix
        (NEW-6: ``aspirin_analog_3`` no longer demonstrates this -- it has
        only one trailing integer group, so ``species_id`` leaves it alone
        -- the same correction already applied in ``energy.py``'s own
        docstrings, task m2).
        """
        mols = [
            _conformer("foo_1_2", -10.0),
            _conformer("foo", -30.0),
        ]

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0), (
            "two distinct compounds were merged into one relative-energy group"
        )
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)


class TestGuards:
    def test_a_group_of_different_compounds_gets_relative_energies_only_within_each_compound(self):
        """I3 (R49): the withholding is per identity, not per title.

        ``calc_thermo`` accepts any SDF, including one with reused titles.
        Subtracting the energies of two different compounds would produce a
        number with no meaning, and it would look exactly like a
        conformational preference -- but refusing the WHOLE title group over
        that used to also blank out every conformer that *is* comparable to
        its own kind. Each identity now gets its own reference instead; a
        subgroup of one record (as here -- each compound is alone under this
        title) gets 0, not nothing.
        """
        mols = [_conformer("shared", -10.0, "CCO"), _conformer("shared", -40.0, "c1ccccc1")]

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)

    def test_stereoisomers_sharing_a_title_get_relative_energies_within_their_own_identity(self):
        """I3 (R49): a diastereomer pair sharing a title each gets its own reference.

        ``ranking.species_id`` strips ``<isomer>_<conformer>``, so two
        enumerated stereoisomers of one input sharing a species id/title
        verbatim is routine by the time a thermo output is written.
        ``species_key`` perceives stereochemistry from the 3D coordinates, so
        a diastereomer pair under one title is two distinct identities (Q3:
        no cross-diastereomer Boltzmann distribution) -- but each identity's
        OWN conformers are still comparable to each other, and withholding
        the whole title group used to blank those out too. A diastereomer
        pair, not an enantiomer pair (M5): mirror images are degenerate under
        every potential Auto3D uses and are removed during enumeration, so an
        enantiomer pair sharing a title is not a geometry Auto3D produces.
        """
        mols = [
            _conformer("stereo_0", -10.0, "C[C@H](O)[C@H](O)C"),
            _conformer("stereo_0", -9.5, "C[C@H](O)[C@H](O)C"),
            _conformer("stereo_0", -8.0, "C[C@H](O)[C@@H](O)C"),
            _conformer("stereo_0", -7.0, "C[C@H](O)[C@@H](O)C"),
        ]

        set_relative_energies(mols)

        # First diastereomer's pair: referenced to its own minimum (-10.0).
        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.5 * EV_TO_KCAL_PER_MOL)
        # Second diastereomer's pair: referenced to ITS OWN minimum (-8.0),
        # not to the first pair's -10.0.
        assert float(mols[2].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[3].GetProp(E_REL_KCAL_PROP)) == pytest.approx(1.0 * EV_TO_KCAL_PER_MOL)

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
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(1.0 * EV_TO_KCAL_PER_MOL)

    def test_a_stale_value_on_a_mixed_title_is_replaced_per_identity(self):
        """I3 (R49): a title spanning two compounds no longer keeps a stale
        value by refusing to touch it -- each identity is recomputed, same
        as any other subgroup."""
        mols = [_conformer("shared", -10.0, "CCO"), _conformer("shared", -40.0, "c1ccccc1")]
        for m in mols:
            m.SetProp(E_REL_KCAL_PROP, "999.0")

        set_relative_energies(mols)

        assert float(mols[0].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)


class TestCalcThermoWiring:
    """The helpers existing is not the fix; `calc_thermo` has to call both."""

    def test_failed_records_lose_their_inherited_relative_energy(self):
        """The inversion this change exists to close.

        The stationary-point gate ``continue``s before ``do_mol_thermo``, so a
        failed record never had its inherited ``E_rel(kcal/mol)`` cleared and
        never gets a recomputed one. Left alone, the documented property would
        survive only on the records the docs tell a reader to discard.
        """
        from Auto3D.domain.energy import clear_relative_energies

        failed = [_conformer("mol_a", -10.0), _conformer("mol_b", -3.0)]
        for m in failed:
            m.SetProp(E_REL_KCAL_PROP, "1.5")

        clear_relative_energies(failed)

        assert not any(m.HasProp(E_REL_KCAL_PROP) for m in failed)

    def test_clearing_a_record_that_has_none_is_a_no_op(self):
        from Auto3D.domain.energy import clear_relative_energies

        mols = [_conformer("m", -1.0)]
        clear_relative_energies(mols)
        assert not mols[0].HasProp(E_REL_KCAL_PROP)

    def test_calc_thermo_runs_both_halves_after_the_loop(self):
        """Wiring guard: the helpers must be reached from ``calc_thermo``.

        A source check rather than a behavioral one because the surrounding
        loop needs a neural network potential, which the fast tier does not
        load. It is narrow on purpose -- it pins that the calls exist and which
        collection each is handed, since swapping those two arguments would
        publish relative energies for the failures and strip them from the
        successes, restoring the exact defect being fixed.
        """
        source = _finalize_thermo_records_source()
        assert "set_relative_energies(out_mols)" in source, (
            "_finalize_thermo_records no longer recomputes relative energies after the "
            "loop, so its output carries none on the records that succeeded"
        )
        assert "clear_relative_energies(mols_failed)" in source, (
            "_finalize_thermo_records no longer strips the stale relative energy from "
            "failed records, so the property survives only where it is meaningless"
        )


class TestRelativeGibbsEnergies:
    """``G_rel(kcal/mol)`` -- the quantity conformer populations are built from.

    Populations go as ``exp(-dG/RT)``. At 298 K ``RT`` is 0.59 kcal/mol while
    conformer differences in ZPE and vibrational entropy run 0.3-1 kcal/mol, so
    a population built from the electronic energy is wrong by a factor of a few
    in exactly the regime it is used for.
    """

    @staticmethod
    def _thermo_record(name, g_hartree, t_k=298.15, e_ev=-10.0, smiles="CCO"):
        mol = _conformer(name, e_ev, smiles)
        mol.SetProp("G_hartree", str(g_hartree))
        mol.SetProp("T_K", str(t_k))
        return mol

    def test_referenced_to_the_lowest_gibbs_energy_in_the_group(self):
        from Auto3D.domain import energy as energy_mod

        mols = [
            self._thermo_record("m", -100.0),
            self._thermo_record("m", -99.5),
        ]

        energy_mod.set_relative_gibbs_energies(mols)

        assert float(mols[0].GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(
            0.5 * energy_mod.HARTREE_TO_KCAL_PER_MOL
        )

    def test_the_gibbs_reference_is_independent_of_the_electronic_one(self):
        """The G-minimum and the E-minimum need not be the same conformer.

        Once ZPE and vibrational entropy enter, the ordering can change. That is
        ordinary chemistry, not an inconsistency to be reconciled -- so the two
        properties pick their references separately.
        """
        from Auto3D.domain import energy as energy_mod

        # Record A is lower in electronic energy; record B is lower in G.
        low_e = self._thermo_record("m", g_hartree=-99.0, e_ev=-20.0)
        low_g = self._thermo_record("m", g_hartree=-100.0, e_ev=-10.0)
        mols = [low_e, low_g]

        energy_mod.set_relative_energies(mols)
        energy_mod.set_relative_gibbs_energies(mols)

        assert float(low_e.GetProp(E_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(low_g.GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(low_g.GetProp(E_REL_KCAL_PROP)) > 0.0
        assert float(low_e.GetProp(energy_mod.G_REL_KCAL_PROP)) > 0.0

    def test_a_group_at_mixed_temperatures_is_refused(self):
        """``G(T)`` contains ``-T*S``; a difference across two T is not a
        conformational preference. ``mol_info_func`` returns a per-record
        temperature, so one output file can legitimately hold both.
        """
        from Auto3D.domain import energy as energy_mod

        mols = [
            self._thermo_record("m", -100.0, t_k=298.15),
            self._thermo_record("m", -99.5, t_k=310.0),
        ]

        energy_mod.set_relative_gibbs_energies(mols)

        assert not any(m.HasProp(energy_mod.G_REL_KCAL_PROP) for m in mols)

    def test_a_stale_value_is_cleared_when_the_group_is_refused(self):
        from Auto3D.domain import energy as energy_mod

        mols = [
            self._thermo_record("m", -100.0, t_k=298.15),
            self._thermo_record("m", -99.5, t_k=310.0),
        ]
        for m in mols:
            m.SetProp(energy_mod.G_REL_KCAL_PROP, "7.0")

        energy_mod.set_relative_gibbs_energies(mols)

        assert not any(m.HasProp(energy_mod.G_REL_KCAL_PROP) for m in mols)

    def test_a_record_without_a_gibbs_energy_is_skipped(self):
        from Auto3D.domain import energy as energy_mod

        mols = [self._thermo_record("m", -100.0), _conformer("m", -10.0)]

        energy_mod.set_relative_gibbs_energies(mols)

        assert float(mols[0].GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert not mols[1].HasProp(energy_mod.G_REL_KCAL_PROP)

    def test_different_compounds_sharing_a_title_get_their_own_gibbs_reference(self):
        """I3 (R49): same per-identity split as ``set_relative_energies``."""
        from Auto3D.domain import energy as energy_mod

        mols = [
            self._thermo_record("shared", -100.0, smiles="CCO"),
            self._thermo_record("shared", -200.0, smiles="c1ccccc1"),
        ]

        energy_mod.set_relative_gibbs_energies(mols)

        assert float(mols[0].GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(0.0)
        assert float(mols[1].GetProp(energy_mod.G_REL_KCAL_PROP)) == pytest.approx(0.0)

    def test_calc_thermo_publishes_it(self):
        source = _finalize_thermo_records_source()
        assert "set_relative_gibbs_energies(out_mols)" in source, (
            "_finalize_thermo_records no longer computes the Gibbs relative energy"
        )


class TestGibbsIsOptIn:
    """Thermo is the slow path, so nothing about dG may be a default.

    ``G_rel`` itself is free once ``calc_thermo`` has run -- it is arithmetic
    over `G_hartree` values already in hand. What must stay opt-in is the *dG
    workflow*: obtaining a Gibbs energy at all means a Hessian per conformer,
    and a default that quietly depends on one would make the cheap path
    expensive. So the electronic quantity is what a run produces unless asked
    otherwise.
    """

    def test_calc_thermo_does_not_compute_it_by_default(self):
        import inspect

        import Auto3D.entry.ASE.thermo.driver as thermo_mod

        param = inspect.signature(thermo_mod.calc_thermo).parameters["relative_gibbs"]
        assert param.default is False, (
            "G_rel(kcal/mol) must be opt-in: the dG workflow costs a Hessian per "
            "conformer and must never be what a user gets without asking"
        )

    def test_the_electronic_relative_energy_has_no_such_switch(self):
        """``E_rel(kcal/mol)`` is documented output and stays unconditional."""
        source = _finalize_thermo_records_source()
        assert "set_relative_energies(out_mols)" in source, (
            "_finalize_thermo_records no longer recomputes relative energies unconditionally"
        )
        # The Gibbs call must sit behind the flag, not beside it.
        gibbs_line = next(
            line for line in source.splitlines() if "set_relative_gibbs_energies(out_mols)" in line
        )
        assert gibbs_line.startswith("        "), (
            f"set_relative_gibbs_energies is not nested under a conditional; got {gibbs_line!r}"
        )
