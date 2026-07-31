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
    """analyze_vibrations must count imaginary modes over the vibration-only
    subset (3N-6 / 3N-5), mirroring IdealGasThermo's own slice exactly, while
    still returning the full 3N-mode set in ``.energies`` for IdealGasThermo
    to slice itself. Most cases below use n_atoms chosen so 3N-6 (nonlinear)
    equals the number of modes given, i.e. every mode supplied is a genuine
    vibration and none of it is trans/rot -- the trans/rot exclusion itself is
    covered separately below.
    """

    def test_a_clean_spectrum_is_untouched(self):
        result = analyze_vibrations(
            _ev(200, 800, 1600, 3000), n_atoms=4, geometry="nonlinear"
        )
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert len(result.energies) == 4

    def test_a_small_imaginary_mode_is_counted_but_tolerated(self):
        """A -15 cm-1 artifact is the reason ignore_imag_modes exists."""
        result = analyze_vibrations(
            _ev(-15, 800, 1600), n_atoms=3, geometry="nonlinear"
        )
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(15.0, abs=0.5)
        assert result.is_transition_state is False

    def test_a_large_imaginary_mode_is_a_transition_state(self):
        """-400 cm-1 is a reaction coordinate, not numerical noise.

        ASE sorts by absolute value and deletes both indiscriminately, so
        without this distinction a saddle point is reported as a minimum.
        """
        result = analyze_vibrations(
            _ev(-400, 800, 1600), n_atoms=3, geometry="nonlinear"
        )
        assert result.n_imag == 1
        assert result.max_imag_cm == pytest.approx(400.0, abs=1.0)
        assert result.is_transition_state is True

    def test_the_largest_imaginary_mode_decides(self):
        """The largest-magnitude imaginary mode comes FIRST in a real
        spectrum, not last: VibrationsData.get_energies() diagonalizes with
        np.linalg.eigh, which returns eigenvalues ascending, so the most
        negative omega^2 (largest imaginary energy) sorts to the front. The
        previous version of this test put the largest imaginary mode last,
        which would not have caught a max(...) -> last-wins regression.
        """
        result = analyze_vibrations(
            _ev(-350, -12, 900), n_atoms=3, geometry="nonlinear"
        )
        assert result.n_imag == 2
        assert result.max_imag_cm == pytest.approx(350.0, abs=1.0)
        assert result.is_transition_state is True

    def test_energies_preserve_the_full_mode_count(self):
        """.energies must stay the full 3N set, imaginary modes included.

        IdealGasThermo is handed this same list and performs its own
        equivalent 3N-6/3N-5 slice internally; if analyze_vibrations instead
        trimmed .energies down to the vibration-only subset, that second cut
        would delete genuine vibrations, but every other test in this class
        would still pass since none of them checks the length of .energies.
        """
        modes = _ev(-400, 10, 20, 800, 1600, 3000)
        result = analyze_vibrations(modes, n_atoms=4, geometry="nonlinear")
        assert len(result.energies) == len(modes)
        assert any(abs(e.imag) > 0.0 for e in result.energies), (
            "an imaginary mode did not survive into .energies"
        )

    def test_translation_rotation_pseudo_imaginary_modes_are_excluded(self):
        """The critical bug: VibrationsData.get_energies() returns all 3N
        modes, including translation/rotation. Those eigenvalues should be
        exactly zero but come out as small positive or negative numerical
        noise, so several of them routinely present as spurious "imaginary"
        modes. Measured case: a 5-atom Lennard-Jones cluster at Auto3D's own
        0.01 eV/A convergence threshold reports 5 spurious imaginary modes up
        to 19i cm-1 counting over the raw 3N set, while ASE's own
        IdealGasThermo -- which performs the 3N-6 cut before counting --
        reports 0. A user filtering N_imaginary_modes == 0 would discard
        every valid conformer without this fix.
        """
        # 5 atoms, nonlinear: 15 modes total, 6 trans/rot (of which several
        # are spuriously "imaginary", all tiny in magnitude) and 9 genuine,
        # entirely real vibrations.
        trans_rot = _ev(-19, -12, -8, 3, 5, 7)
        vibrational = _ev(120, 300, 450, 600, 800, 1000, 1400, 1800, 3000)
        modes = np.concatenate([trans_rot, vibrational])
        result = analyze_vibrations(modes, n_atoms=5, geometry="nonlinear")
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert len(result.energies) == len(modes)

    def test_monatomic_has_no_vibrational_modes(self):
        """A single atom has 3N=3 modes, all translation -- nothing to cut."""
        result = analyze_vibrations(_ev(5, -3, 2), n_atoms=1, geometry="monatomic")
        assert result.n_imag == 0
        assert result.max_imag_cm == pytest.approx(0.0)
        assert len(result.energies) == 3

    def test_a_raised_cutoff_suppresses_the_transition_state_flag(self):
        """imag_cutoff_cm must actually be read by is_transition_state, not
        just accepted and ignored: at the 50 cm-1 default, a 100 cm-1
        imaginary mode is a real artifact; raising the cutoff to 500 must
        suppress the flag."""
        result = analyze_vibrations(
            _ev(-100, 800, 1600), n_atoms=3, geometry="nonlinear",
            imag_cutoff_cm=500.0,
        )
        assert result.is_transition_state is False

    def test_a_lowered_cutoff_triggers_the_transition_state_flag(self):
        """A 30 cm-1 imaginary mode is noise at the 50 cm-1 default, but must
        trigger the flag once the cutoff is lowered below it."""
        result = analyze_vibrations(
            _ev(-30, 800, 1600), n_atoms=3, geometry="nonlinear",
            imag_cutoff_cm=20.0,
        )
        assert result.is_transition_state is True


import logging  # noqa: E402

from rdkit import Chem  # noqa: E402

from Auto3D.ASE.thermo import _resolve_multiplicity, _symmetry_number  # noqa: E402


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

        # A bare object is enough: the AIMNet2Calculator isinstance check fails
        # for it, so the autograd branch is taken and we stop once the model
        # call raises (model=None is not callable) -- well after both the
        # Atoms object and the Hessian's coordinate tensor were built.
        try:
            thermo_mod.vib_hessian(
                mol, _FakeCalculator(), model=None,
                model_name="AIMNET", positions=relaxed,
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

        n_modes = 3 * mol.GetNumAtoms()

        class _FakeVib:
            def get_energies(self):
                return [0.01 + 0j] * n_modes

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise ValueError("synthetic thermo failure")

        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib())
        monkeypatch.setattr(thermo_mod, "IdealGasThermo", _Boom)

        with pytest.raises(ValueError):
            thermo_mod.do_mol_thermo(mol, atoms, model=None, model_name="AIMNET")

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

        n_atoms = mol.GetNumAtoms()
        n_modes = 3 * n_atoms
        vib_values = [1e-6] * 6 + [0.05 + 0.01 * i for i in range(n_modes - 6)]

        class _FakeVib:
            def get_energies(self):
                return [complex(v) for v in vib_values]

        monkeypatch.setattr(thermo_mod, "vib_hessian", lambda *a, **k: _FakeVib())

        result = thermo_mod.do_mol_thermo(mol, atoms, model=None, model_name="AIMNET")

        assert result is mol
        assert mol.HasProp("G_hartree")
        after = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        np.testing.assert_allclose(after, relaxed)
        assert not np.allclose(after, original), (
            "a converged record's conformer was not updated to the relaxed "
            "geometry"
        )


class TestHessianHelperDispatch:
    """An unrecognized model name must raise, not return None."""

    def test_an_unknown_name_raises(self):
        import torch

        from Auto3D.ASE.thermo import aimnet_hessian_helper

        with pytest.raises(ValueError, match="not-a-real-model"):
            aimnet_hessian_helper(
                torch.zeros(1, 1, 3),
                numbers=torch.ones(1, 1, dtype=torch.long),
                charge=torch.zeros(1),
                model=None,
                model_name="not-a-real-model",
            )

    def test_a_registry_alias_raises_rather_than_returning_none(self):
        """aimnet2-2025 matched no branch and fell off the end as None.

        None then flowed into torch.autograd.functional.hessian, whose error
        names neither the model nor the dispatch.
        """
        import torch

        from Auto3D.ASE.thermo import aimnet_hessian_helper

        with pytest.raises(ValueError, match="aimnet2-2025"):
            aimnet_hessian_helper(
                torch.zeros(1, 1, 3),
                numbers=torch.ones(1, 1, dtype=torch.long),
                charge=torch.zeros(1),
                model=None,
                model_name="aimnet2-2025",
            )


class TestLoadHessianModelRouting:
    """ANI2xt/ANI2x/custom-path branches of _load_hessian_model now route
    through ModelFactory instead of hand-rolling the dispatch (M40).

    These monkeypatch ``create_model`` itself, so no real NNP is loaded and
    torchani need not be installed -- only the wiring (which arguments reach
    ModelFactory, and that the returned module is the adapter's raw
    ``.model``, upcast to fp64) is under test. The AIMNET/registry branch is
    deliberately NOT exercised here: constructing it for real would load an
    actual NNP (forbidden in this environment), and its contract is already
    pinned by the slow ``test_load_hessian_model_aimnet*`` tests above.
    """

    def _install_fake_factory(self, monkeypatch):
        import torch

        from Auto3D.ASE import thermo as thermo_mod

        calls = {}

        class _FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        class _FakeAdapter:
            def __init__(self):
                self.model = _FakeModule()

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
        assert result.weight.dtype == torch.float64

    def test_custom_path_routes_through_model_factory(self, monkeypatch, tmp_path):
        import torch

        calls, thermo_mod, device = self._install_fake_factory(monkeypatch)
        fake_path = tmp_path / "custom_model.pt"
        fake_path.write_text("stands in for a model file; never actually loaded")

        result = thermo_mod._load_hessian_model(str(fake_path), device)

        assert calls["args"] == (str(fake_path), device, False, False)
        assert result.weight.dtype == torch.float64
