"""Synthetic Hessians with a prescribed vibrational spectrum, and the two ASE
mode-selection rules written out as pure functions.

Nothing here loads a neural network potential, downloads a model, or touches a
GPU: a Hessian with any spectrum you like is a few lines of linear algebra, and
that is what every thermochemistry test in this repo is built on.

The translation/rotation basis is deliberately constructed here from scratch
rather than imported from ``Auto3D.ASE.thermo._external_mode_basis``. Only the
*span* of those six vectors matters for building a Hessian with a prescribed
spectrum, so an independent construction (the rotations are written with the
opposite cross-product order, giving the same span with the opposite sign)
keeps the fixture from agreeing with the code under test by construction: a
sign, ordering or center-of-mass bug in the production basis would still leave
these Hessians correct, and the projection tests would fail.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.constants import EV_PER_WAVENUMBER

#: The eigenvalue -> energy conversion ASE applies to a mass-weighted Hessian
#: (eV/A^2, amu). Written out from ase.units rather than imported from Auto3D
#: so the fixture does not inherit a mistake in the constant under test.
def _conversion() -> float:
    from ase import units

    return units._hbar * units.m / (units._e * units._amu) ** 0.5


#: Translational + rotational degrees of freedom per geometry class. Stated
#: here independently of the production table for the same reason.
EXTERNAL_DOF = {"monatomic": 3, "linear": 5, "nonlinear": 6}


def n_vib_expected(n_atoms: int, geometry: str) -> int:
    """3N-6 / 3N-5 / 0, stated independently of the production helper."""
    return max(0, 3 * n_atoms - EXTERNAL_DOF[geometry])


def _independent_external_basis(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """3N x 6 mass-weighted translations and infinitesimal rotations."""
    sqrt_m = np.sqrt(masses)
    center = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    offsets = positions - center
    columns = []
    for axis in range(3):
        block = np.zeros_like(positions)
        block[:, axis] = sqrt_m
        columns.append(block.reshape(-1))
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        # (r_i - r_cm) x e_a, i.e. the negative of the production convention:
        # same span, opposite sign.
        columns.append((np.cross(offsets, unit) * sqrt_m[:, None]).reshape(-1))
    return np.column_stack(columns)


def eigenvalue_for(wavenumber_cm: float) -> float:
    """Mass-weighted-Hessian eigenvalue whose ASE energy is ``wavenumber_cm``.

    A negative wavenumber means an imaginary mode, i.e. a negative eigenvalue.
    """
    magnitude = (abs(wavenumber_cm) * EV_PER_WAVENUMBER / _conversion()) ** 2
    return -magnitude if wavenumber_cm < 0 else magnitude


def hessian_with_spectrum(
    atoms: Atoms,
    vibrations_cm,
    external_cm,
    geometry: str = "nonlinear",
) -> np.ndarray:
    """A Cartesian Hessian (eV/A^2) with exactly the spectrum asked for.

    ``vibrations_cm`` are the 3N-6 (or 3N-5) genuine vibrations and
    ``external_cm`` the 6 (or 5) translation/rotation eigenvalues -- which a
    real Hessian puts at small nonzero values rather than exactly zero, and
    which are the modes the old magnitude heuristic had to guess at. Negative
    entries are imaginary modes.
    """
    n_atoms = len(atoms)
    n_external = EXTERNAL_DOF[geometry]
    n_vib = n_vib_expected(n_atoms, geometry)
    assert len(vibrations_cm) == n_vib, (
        f"expected {n_vib} vibrations for {n_atoms} {geometry} atoms, "
        f"got {len(vibrations_cm)}"
    )
    assert len(external_cm) == n_external, (
        f"expected {n_external} translation/rotation modes, got {len(external_cm)}"
    )
    masses = np.asarray(atoms.get_masses(), dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    basis = _independent_external_basis(positions, masses)
    left = np.linalg.svd(basis, full_matrices=True)[0]
    external, internal = left[:, :n_external], left[:, n_external:]
    mass_weighted = (
        internal @ np.diag([eigenvalue_for(w) for w in vibrations_cm]) @ internal.T
        + external @ np.diag([eigenvalue_for(w) for w in external_cm]) @ external.T
    )
    root = np.repeat(np.sqrt(masses), 3)
    return root[:, None] * mass_weighted * root[None, :]


def wavenumbers(energies) -> list[float]:
    """cm^-1 from a list of complex eV energies; imaginary modes come back negative."""
    return [
        (value.real if abs(complex(value).imag) == 0.0 else -abs(complex(value).imag))
        / EV_PER_WAVENUMBER
        for value in map(complex, energies)
    ]


def energies_ev(*wavenumbers_cm) -> list[complex]:
    """Complex eV energies from wavenumbers; a negative one is imaginary."""
    return [
        complex(0.0, abs(w) * EV_PER_WAVENUMBER)
        if w < 0
        else complex(w * EV_PER_WAVENUMBER, 0.0)
        for w in wavenumbers_cm
    ]


# --- ASE's two mode-selection rules, transcribed from its own sources -------
#
# These exist so a test can exercise BOTH rules on whichever single ASE version
# happens to be installed. Each is a verbatim transcription of the selection
# ``IdealGasThermo.__init__`` performs; the reference is quoted next to it so a
# reader can check the transcription without installing that version.


def ase_selection_le_3_27(vib_energies, n_vib: int) -> list[complex]:
    """ASE 3.23.0-3.27.x: ``sort(key=np.abs)`` then keep the last ``3N-6``.

    ``ase/thermochemistry.py``, ``IdealGasThermo.__init__``::

        vib_energies = list(vib_energies)
        vib_energies.sort(key=np.abs)
        if natoms:
            vib_energies = vib_energies[-(3 * natoms - 6):]
    """
    ordered = sorted((complex(v) for v in vib_energies), key=np.abs)
    return ordered[-n_vib:] if n_vib > 0 else []


def ase_selection_ge_3_28(vib_energies, n_vib: int) -> list[complex]:
    """ASE >= 3.28.0, ``vib_selection='highest'`` (the default).

    ``ase/thermochemistry.py``, ``IdealGasThermo.__init__``::

        vib_energies = list(vib_energies)
        vib_energies.sort(key=lambda f: (f ** 2).real)
        ...
        vib_energies = vib_energies[-num_vibs:]

    Under this key every imaginary mode sorts BELOW every real one --
    ``(0+30j)**2 = -900`` against ``30**2 = +900`` -- so a genuine imaginary
    mode is dropped by the selection and a translation/rotation noise mode is
    promoted into the vibrational partition function to fill the quota.
    """
    ordered = sorted((complex(v) for v in vib_energies), key=lambda f: (f ** 2).real)
    return ordered[-n_vib:] if n_vib > 0 else []


ASE_SELECTION_RULES = {
    "abs-sort (ASE 3.23-3.27)": ase_selection_le_3_27,
    "square-sort (ASE >=3.28)": ase_selection_ge_3_28,
}


# --- probe molecules --------------------------------------------------------


def probe_mol(smiles: str = "CCO", name: str = "probe") -> Chem.Mol:
    """An embedded RDKit molecule with a conformer and a name."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", name)
    return mol


def atoms_for(mol: Chem.Mol, potential_energy: float = -1234.5) -> Atoms:
    """ASE atoms for ``mol`` with a stubbed calculator and potential energy."""
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions)
    atoms.get_calculator = lambda: None
    atoms.get_potential_energy = lambda: potential_energy
    return atoms


class FakeVib:
    """Stands in for ``vib_hessian``'s ``VibrationsData`` return value."""

    def __init__(self, hessian: np.ndarray):
        self._hessian = np.asarray(hessian, dtype=float)

    def get_hessian_2d(self) -> np.ndarray:
        return self._hessian


def fake_vib_for(atoms: Atoms, vibrations_cm, external_cm, geometry="nonlinear"):
    """``FakeVib`` whose Hessian has exactly the requested spectrum."""
    return FakeVib(hessian_with_spectrum(atoms, vibrations_cm, external_cm, geometry))


# --- a real force-field Hessian, for the non-synthetic anchor ---------------


def mmff_hessian(smiles: str, *, displacement: np.ndarray | None = None):
    """(atoms, Hessian in eV/A^2) for an MMFF-minimized molecule.

    Built from central differences of the MMFF **energy**, not of
    ``ForceField.CalcGrad(positions)``: verified against this repo's RDKit
    (2025.09.6), ``CalcGrad`` ignores the positions it is handed and returns
    the gradient at the force field's own stored geometry, so a
    gradient-based finite-difference Hessian is silently wrong (it puts
    n-butane's four lowest modes at -140 to -47 cm-1 instead of +123 to +365).
    ``CalcEnergy(positions)`` does honor its argument.

    ``displacement`` (a flat 3N array, in Angstrom) moves the geometry off the
    stationary point, which is where the magnitude heuristic breaks down.
    """
    kcal_per_mol_to_ev = 1.0 / 23.060547830619026
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    force_field = AllChem.MMFFGetMoleculeForceField(
        mol, AllChem.MMFFGetMoleculeProperties(mol)
    )
    force_field.Minimize(maxIts=200000, forceTol=1e-10, energyTol=1e-14)
    flat = np.asarray(force_field.Positions(), dtype=float)
    if displacement is not None:
        flat = flat + np.asarray(displacement, dtype=float)
    n_atoms = mol.GetNumAtoms()
    step = 1e-3
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(3 * n_atoms):
        for j in range(i, 3 * n_atoms):
            total = 0.0
            for si, sj, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                shifted = flat.copy()
                shifted[i] += si * step
                shifted[j] += sj * step
                total += sign * force_field.CalcEnergy(list(shifted))
            value = total / (4 * step * step) * kcal_per_mol_to_ev
            hessian[i, j] = hessian[j, i] = value
    atoms = Atoms(
        [a.GetSymbol() for a in mol.GetAtoms()], flat.reshape(-1, 3)
    )
    atoms.set_masses([a.GetMass() for a in mol.GetAtoms()])
    return atoms, hessian
