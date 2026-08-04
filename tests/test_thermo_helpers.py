"""Fast unit tests for thermochemistry helper functions.

These cover the pure-Python helpers in Auto3D.ASE.thermo and do not run any
neural-network potential or thermodynamic calculation, so they stay in the
fast test suite (the main tests/test_thermo.py module is marked slow).

The two AIMNET Hessian-model checks below are marked ``slow``: each requires a
real ~9s NNP load. They share a module-scoped ``aimnet_hessian_model`` fixture so
that, in the slow suite, the model still loads only once instead of twice.

Test doubles handed to ``Calculator`` mix in
:class:`tests.helpers_adapter.AdapterModuleMixin`. That is not incidental
boilerplate: ``Calculator``'s first argument is a
:class:`Auto3D.models.contract.ModelAdapter` and it asks that object for the
species convention, so a bare ``nn.Module`` with only a ``forward`` is exactly
the category error the constructor now rejects by name. The mixin supplies the
members these doubles do not care about (the two pads, ``to_species``,
``energy``, ``analytic_hessian``) with ``BaseModelAdapter``'s own defaults, so
the double satisfies the contract without any of them weakening it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402

from Auto3D.ASE.thermo import _detect_geometry, _is_collinear  # noqa: E402


def _bent_triatomic(symbols: str, bond_length: float, angle_deg: float):
    """A symmetric bent triatomic with the given apex angle, apex first."""
    half = np.radians(angle_deg) / 2.0
    return Atoms(symbols, [
        (0.0, 0.0, 0.0),
        (bond_length * np.sin(half), bond_length * np.cos(half), 0.0),
        (-bond_length * np.sin(half), bond_length * np.cos(half), 0.0),
    ])


class TestLinearity:
    """Linearity decides 3N-5 vs 3N-6 modes and 1 vs 3 rotational constants."""

    def test_exactly_linear_co2_is_linear(self):
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is True
        assert _detect_geometry(atoms) == "linear"

    def test_slightly_bent_co2_is_still_linear(self):
        """A 0.01 A transverse displacement is numerical, not a real bend.

        The absolute rank tolerance called this nonlinear, which invents a
        rotational degree of freedom and drops the real 667 cm-1 bend.
        """
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0.01, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is True

    def test_genuinely_bent_water_is_nonlinear(self):
        atoms = Atoms("OHH", [(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])
        assert _is_collinear(atoms) is False
        assert _detect_geometry(atoms) == "nonlinear"

    def test_diatomic_is_linear(self):
        assert _is_collinear(Atoms("HH", [(0, 0, 0), (0.74, 0, 0)])) is True

    def test_single_atom_is_monatomic(self):
        assert _detect_geometry(Atoms("He", [(0, 0, 0)])) == "monatomic"

    def test_a_large_bend_is_not_swallowed(self):
        """Guard the other direction: the test must not accept everything."""
        atoms = Atoms("OCO", [(-1.16, 0, 0), (0, 0.30, 0), (1.16, 0, 0)])
        assert _is_collinear(atoms) is False

    def test_a_thermally_bent_co2_is_still_linear(self):
        """CO2's bend is thermally populated to several degrees at 298 K.

        The threshold must sit well above that, or an imperfectly optimized
        linear molecule loses a real 667 cm-1 mode and gains a rotational
        degree of freedom it does not have.
        """
        assert _is_collinear(_bent_triatomic("COO", 1.16, 170.0)) is True

    def test_a_clearly_bent_triatomic_is_nonlinear(self):
        """30 degrees from linear is a bent molecule, not a floppy linear one."""
        assert _is_collinear(_bent_triatomic("COO", 1.16, 150.0)) is False

    def test_a_real_bent_species_is_nonlinear(self):
        """NO2 at 134 degrees is the most nearly-linear genuinely bent case."""
        assert _is_collinear(_bent_triatomic("NOO", 1.19, 134.1)) is False

    def test_octatriyne_off_axis_methyls_is_nonlinear(self):
        """The moment ratio alone passes 2,4,6-octatriyne as linear -- it is a
        regression this test pins down.

        The carbon backbone is straight, but the terminal methyl groups'
        hydrogens sit about 1 A off the backbone axis: a real bend. The
        ratio test misses it purely because the molecule is long enough that
        max(I) has grown large (it scales as N^2), which shrinks the ratio
        for the same absolute offset that made NO2 unambiguously nonlinear at
        a much smaller size -- the ratio is a size cutoff, not a shape test.
        The perpendicular-distance test (LINEARITY_MAX_PERP_ANGSTROM) is what
        actually catches this case; removing it regresses this assertion to
        True (verified by hand: reproduced and reverted, not left in the
        test).
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        mol = Chem.AddHs(Chem.MolFromSmiles("CC#CC#CC#CC"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        atoms = mol2atoms(mol)

        moments = atoms.get_moments_of_inertia()
        ratio = float(np.min(moments)) / float(np.max(moments))
        assert ratio < 1e-2, (
            "this test's premise is that the ratio alone calls octatriyne "
            "linear; if it no longer does, this case stopped reproducing "
            "the regression it is meant to guard"
        )
        assert _is_collinear(atoms) is False
        assert _detect_geometry(atoms) == "nonlinear"

    def test_long_polyyne_with_no_off_axis_atoms_stays_linear(self):
        """A genuinely linear long chain must not be penalized by the new
        absolute-distance test -- only atoms actually off axis should trip it.

        Butadiyne (HC#C-C#CH) has no substituents off the backbone, unlike
        octatriyne's terminal methyls, so every atom sits on (or extremely
        near) the principal axis.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        mol = Chem.AddHs(Chem.MolFromSmiles("C#CC#C"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        atoms = mol2atoms(mol)

        assert _is_collinear(atoms) is True
        assert _detect_geometry(atoms) == "linear"


class TestIsotopeMasses:
    """mol2atoms must carry isotope labels into ASE's per-atom masses.

    The moment-of-inertia linearity test and IdealGasThermo's rotational
    partition function both depend on mass, not just on element identity; an
    isotope label RDKit tracks (e.g. deuterium) is meaningless downstream if
    mol2atoms silently reduces every atom to its natural-abundance element mass.
    """

    def test_deuterated_species_is_heavier_than_protiated(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        def embedded(smiles):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=42)
            return mol2atoms(mol)

        deuterated = embedded("[2H]C#N")
        protiated = embedded("C#N")
        assert deuterated.get_masses().sum() > protiated.get_masses().sum()

    def test_an_unlabeled_atom_gets_its_most_abundant_isotope_mass(self):
        """The QM convention, not the natural-abundance average.

        Gaussian and ORCA build thermochemistry on the most abundant isotope;
        ASE's per-element default is the IUPAC standard atomic weight. Auto3D
        follows the QM programs, because that is what its numbers get compared
        against and because this module already claims to match them on standard
        state. Chlorine is the sharp case among common organic elements: 35.45
        averaged versus 34.96885 for 35-Cl, a 1.4% difference that propagates
        into every C-Cl frequency and into the moments of inertia.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        mol = Chem.AddHs(Chem.MolFromSmiles("ClC"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        atoms = mol2atoms(mol)
        assert atoms.get_chemical_symbols() == [a.GetSymbol() for a in mol.GetAtoms()]

        periodic_table = Chem.GetPeriodicTable()
        expected = [
            periodic_table.GetMostCommonIsotopeMass(a.GetAtomicNum())
            for a in mol.GetAtoms()
        ]
        assert list(atoms.get_masses()) == pytest.approx(expected, abs=1e-9)

        # And it is genuinely different from what ASE would have supplied, or
        # this test would pass against the behavior it replaced.
        ase_defaults = list(Atoms(atoms.get_chemical_symbols()).get_masses())
        assert list(atoms.get_masses()) != pytest.approx(ase_defaults, abs=1e-9)

    def test_a_labeled_atom_still_gets_the_isotope_it_names(self):
        """The QM convention applies to *unlabeled* atoms only.

        An explicit label is a statement about which isotope is present and must
        win over any per-element default, old or new.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        mol = Chem.AddHs(Chem.MolFromSmiles("[13C]#N"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        atoms = mol2atoms(mol)

        carbon = next(
            mass for atom, mass in zip(mol.GetAtoms(), atoms.get_masses())
            if atom.GetAtomicNum() == 6
        )
        assert carbon == pytest.approx(13.00335, abs=1e-4), (
            "an explicit 13C label was overwritten with the 12C mass"
        )


@pytest.fixture(scope="module")
def aimnet_hessian_model():
    """Load the AIMNET Hessian evaluator once for this module's NNP checks."""
    import torch
    from Auto3D.ASE.thermo import _load_hessian_model
    return _load_hessian_model("AIMNET", torch.device("cpu"))


def test_detect_geometry_linear_vs_nonlinear():
    from ase import Atoms
    from Auto3D.ASE.thermo import _detect_geometry
    co2 = Atoms("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    water = Atoms("OH2", [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59]])
    assert _detect_geometry(co2) == "linear"
    assert _detect_geometry(water) == "nonlinear"


def test_symmetry_number_defaults_to_one():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("CCO")
    assert _symmetry_number(m) == 1  # no property -> default 1


def test_symmetry_number_reads_property():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("c1ccccc1")
    m.SetProp("symmetry_number", "12")
    assert _symmetry_number(m) == 12


def test_symmetry_number_invalid_property_falls_back():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _symmetry_number
    m = Chem.MolFromSmiles("CCO")
    m.SetProp("symmetry_number", "not_a_number")
    assert _symmetry_number(m) == 1


def test_resolve_multiplicity_closed_shell_is_singlet():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("CCO")
    assert _resolve_multiplicity(m) == 1
    # Derived multiplicity is recorded on the mol.
    assert m.GetUnsignedProp("multiplicity") == 1


def test_resolve_multiplicity_radical_is_doublet():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("[CH3]")  # methyl radical, 1 unpaired electron
    assert _resolve_multiplicity(m) == 2
    assert m.GetUnsignedProp("multiplicity") == 2


def test_resolve_multiplicity_respects_explicit_property():
    from rdkit import Chem
    from Auto3D.ASE.thermo import _resolve_multiplicity
    m = Chem.MolFromSmiles("[CH3]")  # would derive 2 ...
    m.SetUnsignedProp("multiplicity", 4)  # ... but an explicit value wins
    assert _resolve_multiplicity(m) == 4


def test_do_mol_thermo_default_temperature_is_298_15():
    """Reference temperature must be the thermochemistry standard 298.15 K."""
    import inspect
    from Auto3D.ASE.thermo import do_mol_thermo
    assert inspect.signature(do_mol_thermo).parameters["T"].default == 298.15


@pytest.mark.slow
def test_load_hessian_model_aimnet(aimnet_hessian_model):
    """AIMNET yields an adapter that answers the analytic-Hessian capability.

    The value under test is that ``analytic_hessian`` returns a Hessian rather
    than ``None``: ``None`` would send AIMNet2 down the autograd path, which
    differentiates only the core module and silently drops the external D3 and
    Coulomb terms -- shifting C-H stretches by ~4% with nothing in the output
    saying so. "Something adapter-shaped" is not enough to establish that, which
    is why the capability itself is exercised.
    """
    import torch

    from Auto3D.models.adapter import AIMNet2Adapter

    m = aimnet_hessian_model
    assert isinstance(m, AIMNet2Adapter)

    hess = m.analytic_hessian(
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]]),
        torch.tensor([[1, 1]]),
        torch.tensor([0]),
    )
    assert hess is not None
    assert hess.numel() == 36  # (2 atoms * 3)**2


@pytest.mark.slow
def test_load_hessian_model_aimnet_is_fp32(aimnet_hessian_model):
    import torch
    # The underlying aimnet module stays fp32 (no whole-graph fp64 upcast, which
    # would be false precision here -- unlike the ANI/custom autograd branch).
    p = next(aimnet_hessian_model.model.parameters())
    assert p.dtype == torch.float32


from Auto3D.ASE.thermo import analyze_vibrations  # noqa: E402
from tests.helpers_vibrations import energies_ev as _ev  # noqa: E402
from tests.helpers_vibrations import wavenumbers  # noqa: E402

EV_PER_CM = 1.0 / 8065.54429  # eV per wavenumber


class TestVibrationAnalysis:
    """analyze_vibrations takes exactly the vibrations, and nothing else.

    Translation and rotation are removed upstream by ``projected_vibrations``
    (an Eckart/Sayvetz projection), so there is no selection left to perform
    here. The function used to mirror ``IdealGasThermo``'s own
    magnitude-sorted ``3N-6`` slice so Auto3D's reported diagnostics would
    match what ASE did internally; ASE changed that slice in 3.28.0, after
    which the mirror described a different mode set from the one that produced
    ``G_hartree``. Mirroring is gone, and the exact mode count is now a
    checked precondition instead.

    Cases that exercise the corrections rather than the classification pass
    ``low_freq_cutoff_cm=0.0`` so the quasi-harmonic floor does not obscure
    what is being asserted; the floor has its own tests in
    ``test_thermo_imaginary_mode_inversion.py``.
    """

    def test_a_clean_spectrum_is_untouched(self):
        result = analyze_vibrations(
            _ev(200, 800, 1600, 3000, 3100, 3200), n_atoms=4,
            geometry="nonlinear", low_freq_cutoff_cm=0.0,
        )
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert result.n_inverted == result.n_removed == result.n_raised == 0
        assert result.corrected_energies == result.energies

    def test_a_small_imaginary_mode_is_counted_but_tolerated(self):
        """A -15 cm-1 artifact is the reason the cutoff exists."""
        result = analyze_vibrations(
            _ev(-15, 800, 1600), n_atoms=3, geometry="nonlinear",
            low_freq_cutoff_cm=0.0,
        )
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(15.0, abs=0.5)
        assert result.is_transition_state is False
        assert result.n_inverted == 1
        assert len(result.corrected_energies) == 3, "the mode count must be conserved"

    def test_a_large_imaginary_mode_is_a_transition_state(self):
        """-400 cm-1 is a reaction coordinate, not numerical noise."""
        result = analyze_vibrations(
            _ev(-400, 800, 1600), n_atoms=3, geometry="nonlinear",
            low_freq_cutoff_cm=0.0,
        )
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(400.0, abs=1.0)
        assert result.is_transition_state is True
        assert result.n_removed == 1
        assert len(result.corrected_energies) == 2, (
            "a saddle point must pass 3N-7 modes"
        )

    def test_the_exact_mode_count_is_a_precondition(self):
        """The old contract -- pass 3N and let something downstream cut -- is gone.

        Feeding the full 3N spectrum must fail loudly rather than being
        silently re-sliced, because a silent re-slice is exactly the
        delegation that made G depend on the installed ASE version.
        """
        full_3n = _ev(-1, -2, -3, 4, 5, 6, 200, 800, 1600, 3000, 3100, 3200)
        with pytest.raises(ValueError, match="projected_vibrations"):
            analyze_vibrations(full_3n, n_atoms=4, geometry="nonlinear")
        # Non-vacuity: the same call with the right number of modes is fine.
        assert len(
            analyze_vibrations(full_3n[-6:], n_atoms=4, geometry="nonlinear").energies
        ) == 6

    def test_linear_geometry_expects_3n_minus_5(self):
        """A linear molecule has 5 external degrees of freedom, not 6, so its
        vibrational set is one mode wider than a nonlinear molecule of the same
        size. A collapse of the linear branch onto the nonlinear one would
        silently discard a genuine vibration -- CO2's degenerate bend."""
        result = analyze_vibrations(
            _ev(667, 667, 1333, 2349), n_atoms=3, geometry="linear",
            low_freq_cutoff_cm=0.0,
        )
        assert len(result.energies) == 3 * 3 - 5
        with pytest.raises(ValueError, match="4"):
            analyze_vibrations(
                _ev(667, 667, 1333, 2349), n_atoms=3, geometry="nonlinear",
            )

    def test_monatomic_has_no_vibrational_modes(self):
        result = analyze_vibrations([], n_atoms=1, geometry="monatomic")
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert result.energies == []
        assert result.corrected_energies == []

    def test_a_raised_cutoff_suppresses_the_transition_state_flag(self):
        """imag_cutoff_cm must actually be read by is_transition_state, not
        just accepted and ignored: at the 50 cm-1 default, a 100 cm-1
        imaginary mode is a saddle point; raising the cutoff to 500 makes it a
        tolerated artifact instead."""
        result = analyze_vibrations(
            _ev(-100, 800, 1600), n_atoms=3, geometry="nonlinear",
            imag_cutoff_cm=500.0, low_freq_cutoff_cm=0.0,
        )
        assert result.is_transition_state is False
        assert result.n_inverted == 1
        assert min(wavenumbers(result.corrected_energies)) == pytest.approx(
            100.0, abs=1e-6
        )

    def test_a_lowered_cutoff_triggers_the_transition_state_flag(self):
        """A 30 cm-1 imaginary mode is noise at the 50 cm-1 default, but must
        trigger the flag once the cutoff is lowered below it."""
        result = analyze_vibrations(
            _ev(-30, 800, 1600), n_atoms=3, geometry="nonlinear",
            imag_cutoff_cm=20.0, low_freq_cutoff_cm=0.0,
        )
        assert result.is_transition_state is True
        assert result.n_removed == 1

    def test_the_diagnostics_describe_the_input_not_the_corrected_list(self):
        """n_imag, max_imag_cm and is_transition_state are computed first.

        They are meaningless on a list that has already been inverted, trimmed
        and floored, so the ordering inside analyze_vibrations is load-bearing:
        here the corrected list contains no imaginary mode at all, yet the
        record must still report one.
        """
        result = analyze_vibrations(
            _ev(-30, 800, 1600), n_atoms=3, geometry="nonlinear",
        )
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(30.0, abs=0.5)
        assert all(v.imag == 0.0 for v in result.corrected_energies)


import logging  # noqa: E402

from rdkit import Chem  # noqa: E402

from Auto3D.ASE.thermo import (  # noqa: E402
    _electron_count,
    _resolve_multiplicity,
    _symmetry_number,
)


def _mol(smiles, **props):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for key, value in props.items():
        mol.SetProp(key, str(value))
    return mol


class TestSymmetryNumber:
    def test_defaulting_warns_prominently(self, caplog, monkeypatch):
        """sigma=1 biases G by RT*ln(sigma) and does not cancel between isomers."""
        from Auto3D.ASE import thermo as thermo_mod

        # Isolate from the module-level once-per-run de-dup flag: another
        # test (or test_symmetry_number_defaults_to_one, earlier in this
        # file) may already have tripped it, which would otherwise make this
        # assertion depend on test execution order.
        monkeypatch.setattr(thermo_mod, "_symmetry_default_warned", False)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _symmetry_number(_mol("c1ccccc1"))
        assert any("symmetry_number" in r.message for r in caplog.records), (
            f"defaulting to sigma=1 was not warned about: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_explicit_value_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _symmetry_number(_mol("c1ccccc1", symmetry_number=12))
        assert not any("symmetry_number" in r.message for r in caplog.records)

    def test_a_malformed_property_warns_as_well_as_falling_back(self, caplog):
        """The fallback value is already covered; the warning is new.

        Unaffected by the once-per-run de-dup flag: a malformed property
        takes the except branch, not the defaulting-from-absence branch that
        flag guards.
        """
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _symmetry_number(_mol("CCO", symmetry_number="not-a-number")) == 1
        assert any("symmetry_number" in r.message for r in caplog.records)

    def test_defaulting_warns_only_once_per_run(self, caplog, monkeypatch):
        """A 10,000-molecule batch must not emit 10,000 near-identical lines.

        _symmetry_number is called once per molecule from inside
        calc_thermo's loop, so this cannot rely on the "log once, outside the
        loop" placement that keeps calc_thermo's own symmetry-number INFO log
        to one line per run; it needs its own de-dup state, reset at the top
        of calc_thermo the same way.
        """
        from Auto3D.ASE import thermo as thermo_mod

        monkeypatch.setattr(thermo_mod, "_symmetry_default_warned", False)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            thermo_mod._symmetry_number(_mol("c1ccccc1"))
            thermo_mod._symmetry_number(_mol("CCO"))
        warnings = [r for r in caplog.records if "symmetry_number" in r.message]
        assert len(warnings) == 1, (
            f"expected exactly one defaulting warning across two calls in "
            f"the same run, got {len(warnings)}: {[r.message for r in warnings]}"
        )


class TestMultiplicity:
    def test_a_malformed_property_falls_back_to_the_radical_count(self):
        """The accessor was unguarded where _symmetry_number's is guarded."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "triplet")
        assert _resolve_multiplicity(mol) == 1

    def test_negative_multiplicity_falls_back_to_the_radical_count(self, caplog):
        """GetUnsignedProp("-1") wraps around to 4294967295 rather than
        raising, feeding spin = 2147483647.0 into IdealGasThermo's
        R*ln(multiplicity) term. The value must be validated, not just
        parsed."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "-1")
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_zero_multiplicity_falls_back_to_the_radical_count(self, caplog):
        """GetUnsignedProp("0") parses cleanly to 0, giving spin = -0.5."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "0")
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_dioxygen_is_flagged_as_ambiguous(self, caplog):
        """O=O draws closed-shell but is a ground-state triplet.

        The radical-electron count is 0 here, so nothing signals that the
        closed-shell assumption is wrong for this molecule.
        """
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            multiplicity = _resolve_multiplicity(_mol("O=O"))
        assert multiplicity == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records), (
            f"an ambiguous open-shell drawing was not flagged: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_ordinary_closed_shell_molecule_is_not_flagged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            _resolve_multiplicity(_mol("CCO"))
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_an_unsigned_wraparound_value_falls_back_to_the_radical_count(
        self, caplog
    ):
        """The other side of the guard: int("4294967295") parses cleanly (no
        wraparound -- that only afflicts GetUnsignedProp) to a value that is
        ">= 1" and so slipped past a lower-bound-only check, feeding
        spin = 2147483647.0 into R*ln(multiplicity) with no warning -- a 13.1
        kcal/mol shift in Gibbs energy at 298.15 K. It must be rejected by the
        upper bound (n_electrons + 1)."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "4294967295")
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_value_just_above_the_electron_bound_falls_back(self, caplog):
        """n_electrons + 1 is the physical ceiling (every electron unpaired);
        one above that is invalid regardless of parity."""
        mol = _mol("CCO")
        n_electrons = _electron_count(mol)
        mol.SetProp("multiplicity", str(n_electrons + 3))  # same parity, over the cap
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_wrong_parity_value_falls_back(self, caplog):
        """CCO is a 26-electron (even) closed-shell species, so a valid
        multiplicity must be odd; 2 (even) is unreachable by any spin state
        and must be rejected even though it is within the electron-count
        bound."""
        mol = _mol("CCO", multiplicity=2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 1
        assert any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_legitimate_high_multiplicity_on_an_even_electron_species_passes(
        self, caplog
    ):
        """A triplet (multiplicity 3) on an even-electron species is
        physically legitimate (e.g. a diradical or an excited state) and
        must pass through unchanged and unwarned."""
        mol = _mol("CCO", multiplicity=3)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 3
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)

    def test_a_legitimate_doublet_on_a_radical_passes(self, caplog):
        """The methyl radical is a 9-electron (odd) species, so multiplicity
        2 (even) is exactly the physically expected doublet and must pass
        through unchanged and unwarned."""
        mol = _mol("[CH3]", multiplicity=2)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _resolve_multiplicity(mol) == 2
        assert not any("multiplicity" in r.message.lower() for r in caplog.records)


class TestHessianGeometrySourcing:
    """vib_hessian must build the Hessian from the geometry it is handed."""

    def test_positions_argument_overrides_the_conformer(self, monkeypatch):
        """Passing positions must win over mol's (possibly stale) conformer --
        for BOTH the Atoms object and the coordinate array actually handed to
        the Hessian machinery.

        This is the sourcing half of the C5 fix. The slow tier exercises the
        whole path with a real potential; this pins the contract without one.

        Asserting only on the Atoms() construction is not enough: a mutation
        where Atoms receives the relaxed positions but the code that builds
        the Hessian's coordinate tensor separately re-reads
        mol.GetConformer() would still pass a check confined to Atoms(). So
        this also spies on torch.tensor (vib_hessian's first call to it is
        always the coordinate tensor -- numbers and charge are tensorized
        afterward) and asserts that array equals the relaxed positions too.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE import thermo as thermo_mod

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        stale = mol.GetConformer().GetPositions()
        relaxed = stale + 0.25

        seen = {}

        class _FakeAtoms:
            def __init__(self, *args, **kwargs):
                self._positions = np.asarray(args[1], dtype=float)
                seen["atoms_positions"] = self._positions

            def set_calculator(self, calc):
                pass

            def get_positions(self):
                return self._positions

            # mol2atoms now always applies the QM (most-abundant-isotope) mass
            # convention rather than leaving ASE's per-element default, so the
            # double has to accept masses even though this test is about
            # geometry sourcing.
            def set_masses(self, masses):
                self._masses = np.asarray(masses, dtype=float)

            def get_masses(self):
                return self._masses

        monkeypatch.setattr(thermo_mod, "Atoms", _FakeAtoms)
        monkeypatch.setattr(
            thermo_mod, "VibrationsData", lambda atoms, hess: ("vib", atoms)
        )

        real_tensor = torch.tensor

        def _spy_tensor(data, *args, **kwargs):
            if "hessian_coord" not in seen:
                seen["hessian_coord"] = np.array(data, dtype=float)
            return real_tensor(data, *args, **kwargs)

        monkeypatch.setattr(torch, "tensor", _spy_tensor)

        class _FakeCalculator:
            pass

        from tests.helpers_adapter import FakeAdapter

        # A conforming adapter with no native Hessian, so `analytic_hessian`
        # returns None and the autograd branch is taken. The call is expected to
        # blow up inside the (monkeypatched) tensor machinery; that happens well
        # after both the Atoms object and the Hessian's coordinate tensor exist,
        # which is all this test reads.
        try:
            thermo_mod.vib_hessian(
                mol, _FakeCalculator(), FakeAdapter(),
                positions=relaxed,
            )
        except Exception:
            # Reaching the model call is fine; we only need the geometry
            # that was handed to Atoms and to the Hessian tensor, both of
            # which happen first.
            pass

        assert "atoms_positions" in seen, "vib_hessian never constructed Atoms"
        np.testing.assert_allclose(seen["atoms_positions"], relaxed)
        assert not np.allclose(seen["atoms_positions"], stale), (
            "vib_hessian used the stale conformer instead of the positions given"
        )

        assert "hessian_coord" in seen, "vib_hessian never built a coordinate tensor"
        np.testing.assert_allclose(seen["hessian_coord"], relaxed)
        assert not np.allclose(seen["hessian_coord"], stale), (
            "the Hessian's coordinate tensor was built from the stale "
            "conformer, not the positions handed to Atoms -- the Atoms "
            "object alone is not evidence that the Hessian itself used the "
            "relaxed geometry"
        )


class TestStationaryPointGate:
    """A structure that never converged must not yield thermochemistry."""

    def test_a_converged_run_reports_true(self, monkeypatch):
        from Auto3D.ASE import thermo as thermo_mod

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                return True

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        assert thermo_mod.relax_to_stationary_point(
            object(), fmax=2e-4, steps=10, name="probe"
        ) is True

    def test_an_exhausted_run_reports_false_and_warns(self, monkeypatch, caplog):
        import logging

        from Auto3D.ASE import thermo as thermo_mod

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                return False

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            result = thermo_mod.relax_to_stationary_point(
                object(), fmax=2e-4, steps=10, name="probe"
            )
        assert result is False
        assert any("stationary point" in r.message for r in caplog.records), (
            f"a non-converged relaxation was not warned about: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_the_run_receives_the_thresholds_it_was_given(self, monkeypatch):
        """Guard against the hardcoded 3e-3 creeping back in."""
        from Auto3D.ASE import thermo as thermo_mod

        seen = {}

        class _FakeOptimizer:
            def __init__(self, atoms):
                pass

            def run(self, fmax, steps):
                seen["fmax"], seen["steps"] = fmax, steps
                return True

        monkeypatch.setattr(thermo_mod, "BFGS", _FakeOptimizer)
        thermo_mod.relax_to_stationary_point(
            object(), fmax=2e-4, steps=123, name="probe"
        )
        assert seen == {"fmax": 2e-4, "steps": 123}


class TestRecordFiltering:
    """One malformed record must not destroy a batch of Hessians."""

    def _mol_with_conformer(self, name):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        return mol

    def test_a_none_record_between_valid_ones_is_skipped(self):
        from Auto3D.ASE.thermo import iter_thermo_records

        good1 = self._mol_with_conformer("first")
        good2 = self._mol_with_conformer("second")
        kept = list(iter_thermo_records([good1, None, good2]))
        assert [m.GetProp("_Name") for m in kept] == ["first", "second"]

    def test_a_conformerless_record_is_skipped(self):
        from rdkit import Chem

        from Auto3D.ASE.thermo import iter_thermo_records

        flat = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        flat.SetProp("_Name", "no_conformer")
        good = self._mol_with_conformer("ok")
        kept = list(iter_thermo_records([flat, good]))
        assert [m.GetProp("_Name") for m in kept] == ["ok"]

    def test_skipping_is_reported(self, caplog):
        import logging

        from Auto3D.ASE.thermo import iter_thermo_records

        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            list(iter_thermo_records([None, self._mol_with_conformer("ok")]))
        assert any("Skipping record" in r.message for r in caplog.records), (
            f"a dropped record was not reported: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_an_all_valid_batch_is_untouched(self):
        from Auto3D.ASE.thermo import iter_thermo_records

        mols = [self._mol_with_conformer(f"m{i}") for i in range(3)]
        assert len(list(iter_thermo_records(mols))) == 3


class TestThermoFailedMarker:
    """Pins the success/failure marker CHANGELOG.md and the migration guide
    document as the filtering contract::

        if mol.GetProp("Thermo_failed") == "":
            g = mol.GetProp("G_hartree")

    None of this needs a real NNP or thermo calculation: `_write_thermo_output`
    is the exact code `calc_thermo` calls to write its output, so exercising it
    directly with plain RDKit mols pins the marking logic itself, not just the
    generic (and already-reliable) SDWriter/SDMolSupplier round-trip of an
    empty string property.
    """

    def _mol(self, name):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        return mol

    def test_a_success_record_gets_the_empty_string_marker(self, tmp_path):
        from rdkit import Chem

        from Auto3D.ASE.thermo import _write_thermo_output

        mol = self._mol("success")
        outpath = tmp_path / "out.sdf"
        _write_thermo_output(outpath, out_mols=[mol], mols_failed=[])

        results = list(Chem.SDMolSupplier(str(outpath)))
        assert len(results) == 1
        assert results[0].HasProp("Thermo_failed")
        assert results[0].GetProp("Thermo_failed") == ""

    def test_a_failed_record_keeps_its_own_marker_unmodified(self, tmp_path):
        from rdkit import Chem

        from Auto3D.ASE.thermo import _write_thermo_output

        mol = self._mol("failed")
        mol.SetProp("Thermo_failed", "not_converged")
        outpath = tmp_path / "out.sdf"
        _write_thermo_output(outpath, out_mols=[], mols_failed=[mol])

        results = list(Chem.SDMolSupplier(str(outpath)))
        assert len(results) == 1
        assert results[0].GetProp("Thermo_failed") == "not_converged"

    def test_the_documented_filter_selects_only_the_success(self, tmp_path):
        """Exercises the exact filter CHANGELOG.md/migration.rst document."""
        from rdkit import Chem

        from Auto3D.ASE.thermo import _write_thermo_output

        success = self._mol("success")
        failed = self._mol("failed")
        failed.SetProp("Thermo_failed", "RuntimeError")
        outpath = tmp_path / "out.sdf"
        _write_thermo_output(outpath, out_mols=[success], mols_failed=[failed])

        results = list(Chem.SDMolSupplier(str(outpath)))
        kept = [m for m in results if m.GetProp("Thermo_failed") == ""]
        assert [m.GetProp("_Name") for m in kept] == ["success"]


class TestFailedRecordKeepsInputGeometry:
    """A record that fails inside do_mol_thermo must keep its input geometry.

    The conformer sync used to happen at the top of do_mol_thermo, before the
    Hessian/vibrational-analysis/IdealGasThermo work that can actually raise.
    calc_thermo appends the very same `mol` object (not a copy) to
    mols_failed on an exception, so an early sync meant a failed record was
    written with a relaxed-but-unvalidated geometry and none of the
    properties that would justify it, while a converged record's conformer
    should still end up holding the relaxed geometry it was optimized to. The
    fix defers the sync to the very end of do_mol_thermo, after every thermo
    property has been set successfully. Both tests below run with no real
    NNP: vib_hessian is monkeypatched to a fake returning canned vibrational
    energies, so only do_mol_thermo's own control flow is exercised.
    """

    def _ethanol_with_conformer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "probe")
        return mol

    def _relaxed_atoms(self, mol, displacement):
        from ase import Atoms

        original = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        relaxed = original + displacement
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        atoms = Atoms(symbols, relaxed)
        atoms.get_calculator = lambda: None
        return atoms, original, relaxed

    def test_a_failure_after_the_hessian_leaves_the_conformer_at_its_input_geometry(
        self, monkeypatch
    ):
        from Auto3D.ASE import thermo as thermo_mod

        mol = self._ethanol_with_conformer()
        atoms, original, relaxed = self._relaxed_atoms(mol, displacement=0.5)
        atoms.get_potential_energy = lambda: 0.0

        from tests.helpers_vibrations import fake_vib_for

        n_vib = 3 * mol.GetNumAtoms() - 6
        fake = fake_vib_for(
            atoms, [200.0 + 50.0 * i for i in range(n_vib)],
            (1.6, -3.2, 3.4, -3.5, 3.7, -4.1), "nonlinear",
        )

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise ValueError("synthetic thermo failure")

        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: fake)
        monkeypatch.setattr(thermo_mod, "IdealGasThermo", _Boom)

        with pytest.raises(ValueError):
            thermo_mod.do_mol_thermo(mol, atoms, adapter=None)

        after = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        np.testing.assert_allclose(after, original)
        assert not np.allclose(after, relaxed), (
            "a failed record's conformer was overwritten with the relaxed "
            "geometry even though no thermochemistry was ever computed for it"
        )
        # Some per-mol props are set before the point of failure; G_hartree
        # is not among them, since it is only ever set after IdealGasThermo
        # succeeds.
        assert not mol.HasProp("G_hartree")

    def test_a_converged_record_ends_with_the_relaxed_conformer(self, monkeypatch):
        from Auto3D.ASE import thermo as thermo_mod

        mol = self._ethanol_with_conformer()
        atoms, original, relaxed = self._relaxed_atoms(mol, displacement=0.1)
        atoms.get_potential_energy = lambda: -1234.5

        from tests.helpers_vibrations import fake_vib_for

        n_vib = 3 * mol.GetNumAtoms() - 6
        fake = fake_vib_for(
            atoms, [400.0 + 80.0 * i for i in range(n_vib)],
            (1.6, -3.2, 3.4, -3.5, 3.7, -4.1), "nonlinear",
        )
        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: fake)

        result = thermo_mod.do_mol_thermo(mol, atoms, adapter=None)

        assert result is mol
        assert mol.HasProp("G_hartree")
        after = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        np.testing.assert_allclose(after, relaxed)
        assert not np.allclose(after, original), (
            "a converged record's conformer was not updated to the relaxed "
            "geometry"
        )


class TestTheNameKeyedHessianEvaluatorIsGone:
    """``aimnet_hessian_helper`` and its unknown-name ``ValueError``, retired.

    That function was a fifth way to call a model: five name-keyed branches,
    a per-engine argument order, its own Hartree->eV factor, and a final ``else``
    that existed only because the four branches could not cover every engine name
    -- every aimnet registry alias (``aimnet2-2025``, ``aimnet2-nse``, ...) and
    the lowercase ``aimnet`` landed there and raised. ``vib_hessian`` now asks the
    adapter (``analytic_hessian``, else autograd of ``energy``), so there is no
    name to fail to recognize and the alias case is *supported* rather than
    diagnosed. See ``tests/test_hessian_invocation_contract.py``.
    """

    def test_the_helper_does_not_exist(self):
        from Auto3D.ASE import thermo as thermo_mod

        assert not hasattr(thermo_mod, "aimnet_hessian_helper")

    def test_a_registry_alias_is_now_evaluable_rather_than_a_valueerror(self):
        """The old dispatch's own test asserted this raised; it must not now.

        Driven with a conforming adapter and no engine name at all, which is the
        point: an alias cannot fall through a dispatch that has no branches.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE import thermo as thermo_mod
        from tests.helpers_adapter import FakeAdapter

        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=42)

        vib = thermo_mod.vib_hessian(
            mol, object(), FakeAdapter(), torch.device("cpu")
        )
        assert vib.get_hessian_2d().shape == (3 * mol.GetNumAtoms(),) * 2


class TestLoadHessianModelRouting:
    """ANI2xt/ANI2x/custom-path branches of _load_hessian_model now route
    through ModelFactory instead of hand-rolling the dispatch (M40).

    These monkeypatch ``create_model`` itself, so no real NNP is loaded and
    torchani need not be installed -- only the wiring (which arguments reach
    ModelFactory, and that the adapter handed back owns a module upcast to fp64)
    is under test. The AIMNET/registry branch is deliberately NOT exercised here:
    constructing it for real would load an actual NNP (forbidden in this
    environment), and its contract is already pinned by the slow
    ``test_load_hessian_model_aimnet*`` tests above.

    ``_load_hessian_model`` returns the ADAPTER now, not its raw ``.model``:
    ``vib_hessian`` asks the adapter for the species convention and for either a
    native or an autograd Hessian, so handing back the bare module would put the
    fp64 upcast and the species remap in two different places again.
    """

    def _install_fake_factory(self, monkeypatch):
        import torch

        from Auto3D.ASE import thermo as thermo_mod
        from tests.helpers_adapter import AdapterModuleMixin

        calls = {}

        class _FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        class _FakeAdapter(AdapterModuleMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeModule()

            def forward(self, coords, species, charges, atom_mask=None):
                energy = coords.pow(2).sum(dim=(1, 2))
                return energy, torch.zeros_like(coords)

        def _fake_create_model(name, device, compile_model=None, use_cache=True):
            calls["args"] = (name, device, compile_model, use_cache)
            return _FakeAdapter()

        monkeypatch.setattr(thermo_mod, "create_model", _fake_create_model)
        return calls, thermo_mod, torch.device("cpu")

    def test_ani2xt_routes_through_model_factory_uncached_uncompiled(self, monkeypatch):
        import torch

        calls, thermo_mod, device = self._install_fake_factory(monkeypatch)

        result = thermo_mod._load_hessian_model("ANI2xt", device)

        # use_cache=False: this module is about to be mutated to fp64 in
        # place, and the factory cache is shared with the fp32 instance
        # model_name2model_calculator loads right after in calc_thermo.
        # compile_model=False: nothing here benefits from torch.compile.
        assert calls["args"] == ("ANI2xt", device, False, False)
        assert result.model.weight.dtype == torch.float64

    def test_custom_path_routes_through_model_factory(self, monkeypatch, tmp_path):
        import torch

        calls, thermo_mod, device = self._install_fake_factory(monkeypatch)
        fake_path = tmp_path / "custom_model.pt"
        fake_path.write_text("stands in for a model file; never actually loaded")

        result = thermo_mod._load_hessian_model(str(fake_path), device)

        name, device_arg, compile_model, _use_cache = calls["args"]
        # use_cache is deliberately not asserted: ModelFactory.create's
        # custom-path branch returns a fresh CustomModelAdapter before ever
        # consulting cls._cache (a custom path is never cached), so this
        # parameter has no observable effect for this branch -- pinning its
        # value here would be asserting a no-op. (It does matter for the
        # ANI2xt/ANI2x branch above, which the previous test covers.)
        assert (name, device_arg, compile_model) == (str(fake_path), device, False)
        assert result.model.weight.dtype == torch.float64


class TestCalculatorChargeInvalidatesCache:
    """A charge change must discard the energy AND forces cached at the old one.

    ASE decides whether a cached result is still valid with
    ``Calculator.check_state`` -> ``compare_atoms``, which looks at positions,
    atomic numbers, cell and pbc only. The molecular charge is invisible to it,
    so a calculator whose charge was reassigned without ``reset()`` handed the
    PREVIOUS molecule's energy and gradient to the next one whenever the two
    shared a geometry. That is exactly a vertical IP/EA input -- one geometry,
    two charges -- where the error is the whole ionization energy or electron
    affinity (20-90 kcal/mol) and nothing in the output says so.

    These tests assert the NUMBERS, not that a code path ran: the stub model's
    energy and forces are functions of the charge, so a reused cache shows up
    as an identical energy/force for two different charges.
    """

    @staticmethod
    def _calculator(charge=0):
        """A ``Calculator`` over a stub whose energy/forces depend on charge.

        The stub owns one real ``nn.Parameter`` on the CPU so
        ``Calculator.__init__`` reads device/dtype from it, pinning this test
        to CPU/float32 regardless of what is visible. (The param-less branch
        is CPU now too -- see
        ``TestCalculatorDeviceAndDtypeFollowTheCaller`` -- but reading a real
        parameter is what this class is about.) No NNP is loaded.
        """
        import torch
        from torch import nn

        from Auto3D.ASE.thermo import Calculator
        from tests.helpers_adapter import AdapterModuleMixin

        class _ChargeDependentModel(AdapterModuleMixin, nn.Module):
            def __init__(self):
                super().__init__()
                # Anchors device=cpu / dtype=float32 for the ASE-facing tensors.
                self.anchor = nn.Parameter(torch.zeros(1))
                self.calls = 0

            def forward(self, coords, species, charges, atom_mask=None):
                self.calls += 1
                q = float(charges.reshape(-1)[0].item())
                # E = -1 eV per unit charge; F = +0.5 q eV/A along x on atom 0.
                energy = torch.tensor([-1.0 * q], dtype=torch.double)
                forces = torch.zeros_like(coords)
                forces[0, 0, 0] = 0.5 * q
                return energy, forces

        model = _ChargeDependentModel()
        return Calculator(model, charge), model

    @staticmethod
    def _atoms():
        from ase import Atoms

        # One fixed geometry reused at both charges -- the vertical IP/EA case.
        return Atoms("H2", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])

    def test_energy_recomputed_after_set_charge(self):
        calc, model = self._calculator(charge=0)
        atoms = self._atoms()
        atoms.calc = calc

        e_neutral = atoms.get_potential_energy()
        calc.set_charge(1)
        e_cation = atoms.get_potential_energy()

        assert e_neutral == pytest.approx(0.0)
        # The number, not the code path: a reused cache returns 0.0 here.
        assert e_cation == pytest.approx(-1.0)
        assert e_cation != e_neutral
        assert model.calls == 2

    def test_forces_recomputed_after_set_charge(self):
        calc, _ = self._calculator(charge=0)
        atoms = self._atoms()
        atoms.calc = calc

        f_neutral = atoms.get_forces()
        calc.set_charge(2)
        f_cation = atoms.get_forces()

        # BFGS reads forces, not energy: a stale gradient makes it "converge"
        # in zero steps on the previous molecule's geometry.
        assert f_neutral[0, 0] == pytest.approx(0.0)
        assert f_cation[0, 0] == pytest.approx(1.0)
        assert not np.allclose(f_neutral, f_cation)

    def test_direct_charge_assignment_also_invalidates(self):
        """``calc.charge = q`` is the same path as ``set_charge(q)``."""
        calc, _ = self._calculator(charge=0)
        atoms = self._atoms()
        atoms.calc = calc

        e_neutral = atoms.get_potential_energy()
        calc.charge = -1
        assert atoms.get_potential_energy() == pytest.approx(1.0)
        assert atoms.get_potential_energy() != e_neutral

    def test_charge_is_calculator_state_not_a_bare_attribute(self):
        """The charge lives in ASE's own ``parameters`` dict."""
        calc, _ = self._calculator(charge=0)
        assert calc.parameters["charge"] == 0
        calc.set_charge(-2)
        assert calc.parameters["charge"] == -2
        assert int(calc.charge.reshape(-1)[0].item()) == -2

    def test_unchanged_charge_keeps_the_cache(self):
        """Re-setting the same charge must not force a needless recompute."""
        calc, model = self._calculator(charge=0)
        atoms = self._atoms()
        atoms.calc = calc

        atoms.get_potential_energy()
        calc.set_charge(0)
        atoms.get_potential_energy()
        assert model.calls == 1


class TestCalculatorDeviceAndDtypeFollowTheCaller:
    """A ``calc_thermo`` call must run on the one device the user asked for.

    For a **param-less** custom NNP (one that builds its backend lazily, so
    ``Calculator.__init__`` has no ``nn.Parameter`` to read a device off),
    the calculator used to choose
    ``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` and
    ``torch.double`` on its own. ``use_gpu`` and ``gpu_idx`` never reached it,
    so ``calc_thermo(..., use_gpu=False)`` relaxed the geometry on **cuda:0 in
    float64** while the fmax pre-check and the Hessian ran on **cpu in
    float32** -- one call, two devices, two precisions, nothing logged, and
    ``gpu_idx`` ignored entirely (always device 0).

    These tests assert the device and dtype of the tensors the model actually
    receives, not that a branch was taken, and they fake CUDA availability so
    they mean the same thing on a CI runner with no GPU as on an 8-GPU box.
    """

    @staticmethod
    def _paramless_model():
        """A custom NNP holding no ``nn.Parameter`` -- the H1 input shape.

        Records the device and dtype of every tensor it is handed, which is
        the only thing that decides whether the run stayed where it was told.
        """
        import torch
        from torch import nn

        from tests.helpers_adapter import AdapterModuleMixin

        # The mixin supplies the ModelAdapter members this double does not
        # care about (pads, to_species, energy); EnForce_ANI gates on them.
        # It contributes no nn.Parameter, so the premise below still holds.
        class _ParamlessRecordingNNP(AdapterModuleMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.seen: list[dict] = []

            def forward(self, coords, species, charges, atom_mask=None):
                self.seen.append({
                    "device": coords.device,
                    "dtype": coords.dtype,
                    "species_device": species.device,
                    "charge_device": charges.device,
                })
                energy = torch.zeros(coords.shape[0], dtype=coords.dtype)
                # A toy restoring force: non-zero (so the fmax pre-check fails
                # and the ASE calculator -- the cuda-seizing half of the split
                # -- is actually exercised) and geometry-dependent (so BFGS's
                # curvature update does not divide by zero). Detached because
                # ASE converts forces with .numpy().
                forces = (-0.5 * coords).detach()
                return energy, forces

        assert not list(_ParamlessRecordingNNP().parameters()), (
            "test premise: the model must hold no nn.Parameter"
        )
        return _ParamlessRecordingNNP()

    @staticmethod
    def _pretend_cuda_is_available(monkeypatch, count: int = 8):
        """Make the CUDA-availability probe say yes without touching a GPU.

        Both ``is_available`` and ``device_count`` are patched: CI runners have
        no CUDA device at all, and a test that only passes on the 8-GPU
        development box would not defend anything.
        """
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    @staticmethod
    def _atoms():
        from ase import Atoms

        return Atoms("H2", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])

    def test_requested_cpu_device_wins_over_visible_cuda(self, monkeypatch):
        """device=cpu means every tensor is on the CPU, CUDA visible or not."""
        import torch

        from Auto3D.ASE.thermo import Calculator

        self._pretend_cuda_is_available(monkeypatch)
        model = self._paramless_model()
        calc = Calculator(model, 0, device=torch.device("cpu"))

        assert calc.device == torch.device("cpu")
        assert calc.dtype is torch.float32

        atoms = self._atoms()
        atoms.calc = calc
        atoms.get_potential_energy()

        assert model.seen, "the model was never called"
        for call in model.seen:
            assert call["device"].type == "cpu", (
                f"coordinates were built on {call['device']}, not the "
                "requested cpu"
            )
            assert call["dtype"] is torch.float32, (
                f"coordinates were built as {call['dtype']}; the rest of the "
                "call (mol2aimnet_input, the charge tensor) is float32"
            )
            assert call["species_device"].type == "cpu"
            assert call["charge_device"].type == "cpu"

    def test_no_device_argument_never_seizes_a_gpu(self, monkeypatch):
        """With nothing to infer from, CPU is the only safe answer.

        A calculator that picks cuda because a GPU happens to be visible makes
        ``use_gpu=False`` untrue for anyone who does not pass a device.
        """
        import torch

        from Auto3D.ASE.thermo import Calculator

        self._pretend_cuda_is_available(monkeypatch)
        calc = Calculator(self._paramless_model(), 0)

        assert calc.device == torch.device("cpu")
        assert calc.dtype is torch.float32

    def test_a_models_own_parameter_device_is_still_honored(self):
        """Nothing changes for a model that does carry parameters."""
        import torch
        from torch import nn

        from Auto3D.ASE.thermo import Calculator

        from tests.helpers_adapter import AdapterModuleMixin

        class _WithParam(AdapterModuleMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float64))

            def forward(self, coords, species, charges, atom_mask=None):
                return torch.zeros(coords.shape[0]), torch.zeros_like(coords)

        calc = Calculator(_WithParam(), 0)
        assert calc.device == torch.device("cpu")
        assert calc.dtype is torch.float64

    def test_calc_thermo_no_gpu_uses_one_device_and_one_dtype(
        self, monkeypatch, tmp_path
    ):
        """The whole call, both stages, on the device ``use_gpu=False`` means.

        The fmax pre-check goes through ``mol2aimnet_input`` (device from
        ``get_device``) and the relaxation goes through the ASE ``Calculator``.
        Before this fix those two disagreed for a param-less custom NNP.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        import Auto3D.ASE.thermo as thermo_mod

        self._pretend_cuda_is_available(monkeypatch)
        model = self._paramless_model()
        monkeypatch.setattr(thermo_mod, "create_model", lambda *a, **k: model)
        monkeypatch.setattr(
            thermo_mod, "_load_hessian_model", lambda *a, **k: object()
        )

        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "water")
        sdf = tmp_path / "in.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)
        out = tmp_path / "out.sdf"

        thermo_mod.calc_thermo(
            str(sdf), "AIMNET", use_gpu=False, opt_steps=2, out_path=str(out)
        )

        assert len(model.seen) >= 2, (
            "expected both the fmax pre-check and the ASE relaxation to call "
            f"the model; got {len(model.seen)} call(s)"
        )
        devices = {call["device"].type for call in model.seen}
        dtypes = {call["dtype"] for call in model.seen}
        assert devices == {"cpu"}, (
            f"use_gpu=False, but one calc_thermo call spanned devices {devices}"
        )
        assert dtypes == {torch.float32}, (
            f"one calc_thermo call spanned precisions {dtypes}: the geometry "
            "would be relaxed at one precision and the Hessian built at another"
        )


class TestSymmetryNumberIsValidatedNotClamped:
    """An impossible sigma must be reported, not silently turned into 1 or used.

    Every other invalid value in ``_symmetry_number`` warns. A *parseable* but
    impossible one did not: ``max(1, int(...))`` turned "0" and "-3" into sigma=1
    in silence, and there was no upper bound at all, so a mistyped "1000000" was
    accepted and shifted Gibbs energy by R·T·ln(1e6) = 8.2 kcal/mol at 298 K.
    ``_resolve_multiplicity``, two functions below in the same file, already
    bounds and parity-checks its property.
    """

    @pytest.mark.parametrize("bad", ["0", "-3", "1000000", "61"])
    def test_an_impossible_value_warns_and_falls_back_to_one(self, bad, caplog):
        from Auto3D.ASE import thermo as thermo_mod

        mol = _mol("c1ccccc1", symmetry_number=bad)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            sigma = thermo_mod._symmetry_number(mol)

        assert sigma == 1, f"sigma={bad!r} was used or clamped without falling back"
        assert any("symmetry_number" in r.message for r in caplog.records), (
            f"symmetry_number={bad!r} was accepted or clamped with no warning"
        )

    @pytest.mark.parametrize("good", ["1", "2", "6", "12", "60"])
    def test_a_real_symmetry_number_is_used_without_complaint(self, good, caplog):
        """The bound must not reject values real molecules have.

        60 is the top of the range (icosahedral I/Ih -- C60, B12H12(2-)), so it
        has to be accepted; a guard written as ``> 60`` versus ``>= 60`` differs
        exactly here.
        """
        from Auto3D.ASE import thermo as thermo_mod

        mol = _mol("c1ccccc1", symmetry_number=good)
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            sigma = thermo_mod._symmetry_number(mol)

        assert sigma == int(good)
        assert not any("symmetry_number" in r.message for r in caplog.records), (
            f"a valid symmetry_number={good!r} was warned about"
        )


class TestTheHessianPathBoundaryCasesAfterUnification:
    """The two boundary defects the deleted name-keyed dispatch used to own.

    ``aimnet_hessian_helper`` had to convert species itself for its ANI2xt branch
    (``reshape(-1)``, not ``squeeze()``, so a MONATOMIC molecule's ``(1, 1)``
    numbers tensor did not collapse to 0-d and make ``.tolist()`` a bare int) and
    had to cast the charge to the coordinates' dtype for its custom-NNP branch
    (``vib_hessian`` builds ``torch.tensor([charge])``, i.e. int64, while the
    optimization half of the same call feeds float32 through ``pad_from_mols``).

    Both are still properties of the path, but neither is now that function's
    problem, and that is the point of keeping these tests:

    * species conversion happens ONCE, in ``vib_hessian``, over a Python list of
      atomic numbers taken from the mol -- no tensor to collapse, and the shape
      is checked below for a lone atom;
    * the charge cast lives in ``CustomModelAdapter.energy``
      (``charges.to(coords.dtype)``), which is the same code the optimization
      path uses, so the two cannot drift apart. Pinned in
      ``tests/test_model_adapter.py::TestEnergyIsDtypePreserving``; asserted here
      end-to-end through ``vib_hessian``.
    """

    @staticmethod
    def _monatomic():
        from rdkit import Chem

        mol = Chem.MolFromSmiles("[C]")
        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)
        assert mol.GetNumAtoms() == 1, "test premise: exactly one atom"
        return mol

    def test_a_monatomic_species_gets_one_species_index_not_a_scalar(self):
        """``squeeze()`` collapsed a (1, 1) numbers tensor to 0-d.

        ``.tolist()`` on a 0-d tensor is a bare ``int``, and iterating an int
        raises ``TypeError``. Reachable: ``vib_hessian`` builds the Hessian before
        ``_detect_geometry`` runs three lines later, so nothing classifies the
        species monatomic in time to skip it, and the lone atom was reported as
        ``Thermo_failed`` rather than as monatomic.
        """
        import torch

        from Auto3D.ASE import thermo as thermo_mod
        from tests.helpers_adapter import FakeAdapter

        adapter = FakeAdapter()
        seen = {}
        real_to_species = adapter.to_species

        def _record(atomic_numbers):
            seen["numbers"] = list(atomic_numbers)
            return real_to_species(atomic_numbers)

        adapter.to_species = _record

        thermo_mod.vib_hessian(
            self._monatomic(), object(), adapter, torch.device("cpu")
        )

        assert seen["numbers"] == [6], (
            f"one atom must yield one species value, got {seen['numbers']}"
        )

    def test_a_custom_model_sees_a_float_charge_matching_its_coordinates(self):
        """The optimization half of the same run passes float32 (padding.py).

        ``vib_hessian`` builds the charge from a Python int (int64), so a custom
        NNP that does arithmetic on it, or that is dtype-sensitive, used to get
        two different answers within one ``calc_thermo`` call. The cast now lives
        in ``CustomModelAdapter.energy``, i.e. in the same object the
        optimization path calls.
        """
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from torch import nn

        from Auto3D.ASE import thermo as thermo_mod
        from Auto3D.models.adapter import CustomModelAdapter

        seen = {}

        class _Toy(nn.Module):
            def forward(self, species, coords, charges):
                seen["charge_dtype"] = charges.dtype
                return coords.pow(2).sum(dim=(1, 2))

        adapter = CustomModelAdapter.__new__(CustomModelAdapter)
        nn.Module.__init__(adapter)
        adapter.model = _Toy()
        adapter.coord_pad, adapter.species_pad = 0.0, -1

        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        thermo_mod.vib_hessian(mol, object(), adapter, torch.device("cpu"))

        # torch.float64: vib_hessian builds fp64 coordinates, and `energy` is
        # dtype-preserving, so the charge follows the coordinates rather than
        # arriving as the int64 tensor the formal charge started as.
        assert seen["charge_dtype"] is torch.float64, (
            f"custom NNP received charge as {seen['charge_dtype']} beside "
            "float64 coordinates"
        )


class TestLoadHessianModelDispatchFoldsCase:
    """The dtype half of the dispatch still folds case, one function up.

    ``_load_hessian_model("ani2x", device)`` used to miss the ANI branch and fall
    through to the aimnet-registry branch, which then reached for a
    ``.calculator`` attribute an ANI2xAdapter does not have -- so the run died in
    the generic "Unexpected Error" panel at exit 1, after model construction had
    already been paid for. The reach-through is gone, but the branch still exists
    and still decides fp64-vs-fp32, so the case folding still matters: an ANI
    engine that missed it would be differentiated in fp32.
    """

    @pytest.mark.parametrize("spelling", ["ANI2x", "ani2x", "ANI2xt", "ani2xt"])
    def test_an_ani_name_takes_the_ani_branch_in_any_case(self, spelling, monkeypatch):
        import torch

        from Auto3D.ASE import thermo as thermo_mod
        from tests.helpers_adapter import AdapterModuleMixin

        class _FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        class _FakeAdapter(AdapterModuleMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeModel()

            def forward(self, coords, species, charges, atom_mask=None):
                raise AssertionError("not called by _load_hessian_model")

        seen = {}

        def _create(name, device, compile_model=None, use_cache=True):
            seen["use_cache"] = use_cache
            return _FakeAdapter()

        monkeypatch.setattr(thermo_mod, "create_model", _create)

        result = thermo_mod._load_hessian_model(spelling, torch.device("cpu"))

        # The ANI branch, identified by what only it does: fp64 in place, and the
        # factory cache disabled so the shared fp32 instance is not upcast too.
        assert result.model.weight.dtype == torch.float64
        assert seen["use_cache"] is False


class TestDerivedMultiplicityIsAlsoChecked:
    """The bounds and parity checks only ever ran on a *supplied* multiplicity.

    Both sit inside ``if mol.HasProp("multiplicity")``, so a multiplicity Auto3D
    derived from the radical-electron count was returned unchecked -- and 2S+1
    requires odd multiplicity for an even-electron species, even for an odd-electron
    one, which the radical count can violate when the drawing hides an open shell.
    Getting it wrong is worth R*ln 3 = 0.65 kcal/mol in T*S_elec.

    RDKit re-derives radical electrons on sanitization and is self-consistent for
    every ordinary species (checked: CH2, CH3, O2, N, ethanol), which is why this
    needed a deliberately inconsistent molecule rather than a realistic one --
    and why the check is worth having: the case it catches is precisely the one no
    ordinary input produces.
    """

    def test_a_parity_inconsistent_derived_multiplicity_warns(self, caplog):
        from rdkit import Chem

        from Auto3D.ASE import thermo as thermo_mod

        # CH3 has 9 electrons (odd), so 2S+1 must be even. Forcing two unpaired
        # electrons makes the derived multiplicity 3 -- odd, and impossible.
        mol = Chem.MolFromSmiles("[CH3]")
        mol.GetAtomWithIdx(0).SetNumRadicalElectrons(2)
        mol.SetProp("_Name", "impossible_radical")
        assert thermo_mod._electron_count(mol) % 2 == 1, "test premise: odd electrons"

        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            multiplicity = thermo_mod._resolve_multiplicity(mol)

        assert multiplicity == 3, "the derived value is still returned, not replaced"
        assert any("parity" in r.message for r in caplog.records), (
            f"an impossible derived multiplicity was returned with no warning: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_a_consistent_derived_multiplicity_is_not_warned_about(self, caplog):
        """Every ordinary radical must stay quiet, or the check is noise."""
        from rdkit import Chem

        from Auto3D.ASE import thermo as thermo_mod

        for smiles in ("[CH3]", "[CH2]", "CCO", "[N]"):
            caplog.clear()
            mol = Chem.MolFromSmiles(smiles)
            with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
                thermo_mod._resolve_multiplicity(mol)
            assert not any("parity" in r.message for r in caplog.records), (
                f"{smiles} triggered a parity warning it should not have"
            )
