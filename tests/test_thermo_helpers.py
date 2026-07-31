"""Fast unit tests for thermochemistry helper functions.

These cover the pure-Python helpers in Auto3D.ASE.thermo and do not run any
neural-network potential or thermodynamic calculation, so they stay in the
fast test suite (the main tests/test_thermo.py module is marked slow).

The two AIMNET Hessian-model checks below are marked ``slow``: each requires a
real ~9s NNP load (a separate model from the conftest ``aimnet_model`` adapter,
since ``_load_hessian_model`` returns the bare AIMNet2Calculator). They share a
module-scoped ``aimnet_hessian_model`` fixture so that, in the slow suite, the
model still loads only once instead of twice.
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

    def test_unlabeled_species_uses_ordinary_masses(self):
        """No isotopes set -> the symbol-only path, byte-for-byte unchanged."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from Auto3D.ASE.thermo import mol2atoms

        mol = Chem.AddHs(Chem.MolFromSmiles("C#N"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        atoms = mol2atoms(mol)
        assert atoms.get_chemical_symbols() == [a.GetSymbol() for a in mol.GetAtoms()]
        assert list(atoms.get_masses()) == list(Atoms(atoms.get_chemical_symbols()).get_masses())


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
    m = aimnet_hessian_model
    # An AIMNet2Calculator from the aimnet registry (not a bundled .jpt);
    # vib_hessian routes it through the calculator's full-pipeline analytic Hessian.
    assert m is not None
    assert hasattr(m, "model")  # the calculator wraps the underlying nn.Module


@pytest.mark.slow
def test_load_hessian_model_aimnet_is_fp32(aimnet_hessian_model):
    import torch
    # The underlying aimnet module stays fp32 (no whole-graph fp64 upcast).
    p = next(aimnet_hessian_model.model.parameters())
    assert p.dtype == torch.float32


from Auto3D.ASE.thermo import analyze_vibrations  # noqa: E402

EV_PER_CM = 1.0 / 8065.54429  # eV per wavenumber


def _ev(*wavenumbers_cm):
    """Build a vibrational-energy array in eV from wavenumbers.

    A negative wavenumber is an imaginary mode, which ASE represents as a
    complex energy with a nonzero imaginary part.
    """
    out = []
    for w in wavenumbers_cm:
        if w < 0:
            out.append(complex(0.0, abs(w) * EV_PER_CM))
        else:
            out.append(complex(w * EV_PER_CM, 0.0))
    return np.array(out)


class TestVibrationAnalysis:
    def test_a_clean_spectrum_is_untouched(self):
        result = analyze_vibrations(_ev(200, 800, 1600, 3000))
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert result.n_raised == 0

    def test_a_small_imaginary_mode_is_counted_but_tolerated(self):
        """A -15 cm-1 artifact is the reason ignore_imag_modes exists."""
        result = analyze_vibrations(_ev(-15, 800, 1600))
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(15.0, abs=0.5)
        assert result.is_transition_state is False

    def test_a_large_imaginary_mode_is_a_transition_state(self):
        """-400 cm-1 is a reaction coordinate, not numerical noise.

        ASE sorts by absolute value and deletes both indiscriminately, so
        without this distinction a saddle point is reported as a minimum.
        """
        result = analyze_vibrations(_ev(-400, 800, 1600))
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(400.0, abs=1.0)
        assert result.is_transition_state is True

    def test_the_largest_imaginary_mode_decides(self):
        result = analyze_vibrations(_ev(-12, -350, 900))
        assert result.n_imag == 2
        assert result.max_imag_cm == pytest.approx(350.0, abs=1.0)
        assert result.is_transition_state is True

    def test_low_frequencies_are_raised_to_the_cutoff(self):
        """A 10 cm-1 torsion contributes ~2.4 kcal/mol to -T*S at 298 K."""
        result = analyze_vibrations(_ev(10, 40, 800), low_freq_cutoff_cm=100.0)
        real_cm = sorted(round(e.real / EV_PER_CM) for e in result.energies)
        assert real_cm == [100, 100, 800], real_cm
        assert result.n_raised == 2

    def test_raising_is_off_by_default_at_zero_cutoff(self):
        result = analyze_vibrations(_ev(10, 40, 800), low_freq_cutoff_cm=0.0)
        real_cm = sorted(round(e.real / EV_PER_CM) for e in result.energies)
        assert real_cm == [10, 40, 800], real_cm
        assert result.n_raised == 0

    def test_imaginary_modes_are_not_raised(self):
        """Raising applies to real low frequencies, never to imaginary ones."""
        result = analyze_vibrations(_ev(-20, 10, 800), low_freq_cutoff_cm=100.0)
        assert result.n_raised == 1
        assert result.n_imag == 1


import logging  # noqa: E402

from rdkit import Chem  # noqa: E402

from Auto3D.ASE.thermo import _resolve_multiplicity, _symmetry_number  # noqa: E402


def _mol(smiles, **props):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for key, value in props.items():
        mol.SetProp(key, str(value))
    return mol


class TestSymmetryNumber:
    def test_defaulting_warns_prominently(self, caplog):
        """sigma=1 biases G by RT*ln(sigma) and does not cancel between isomers."""
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
        """The fallback value is already covered; the warning is new."""
        with caplog.at_level(logging.WARNING, logger="Auto3D.ASE.thermo"):
            assert _symmetry_number(_mol("CCO", symmetry_number="not-a-number")) == 1
        assert any("symmetry_number" in r.message for r in caplog.records)


class TestMultiplicity:
    def test_a_malformed_property_falls_back_to_the_radical_count(self):
        """The accessor was unguarded where _symmetry_number's is guarded."""
        mol = _mol("CCO")
        mol.SetProp("multiplicity", "triplet")
        assert _resolve_multiplicity(mol) == 1

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
