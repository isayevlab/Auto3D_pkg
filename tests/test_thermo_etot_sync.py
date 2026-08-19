"""``calc_thermo`` relaxes the geometry, so it must not leave a stale ``E_tot``.

``DEFAULT_THERMO_CONVERGENCE_THRESHOLD`` (2e-4 eV/A) is 50x tighter than the
conformer-generation threshold ``DEFAULT_CONVERGENCE_THRESHOLD`` (0.01), so on
the canonical ``main() -> calc_thermo`` workflow the pre-check essentially
always fails and BFGS always moves the structure. ``do_mol_thermo`` then writes
the relaxed coordinates into ``mol``'s conformer and writes ``E_hartree`` for
that geometry -- but ``E_tot``/``E_tot(Hartree)`` were read off the *input* SDF
and described the geometry before the relaxation.

The result was one record carrying two disagreeing electronic energies for the
same coordinates. ``ConformerRanker`` and ``select_tautomers`` both read
``E_tot``, so feeding a thermo output to either ranked on an energy belonging to
a structure no longer in the file.

Nothing here loads a neural network potential.
"""

from __future__ import annotations

import ase
import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.foundation.constants import EV_TO_HARTREE
from Auto3D.foundation.utils.energy import e_tot_ev, set_e_tot_from_ev

CPU = torch.device("cpu")

# The energy the stand-in calculator reports for the relaxed geometry. Distinct
# from anything a stale record would carry, so a test that passes cannot be
# passing on a coincidence.
RELAXED_ENERGY_EV = -7.5
STALE_ENERGY_EV = -3.25


class _FixedEnergyCalculator(Calculator):
    """Reports a constant energy and zero forces, so the geometry is stationary."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": RELAXED_ENERGY_EV,
            "forces": np.zeros((len(atoms), 3)),
        }


class _HarmonicAdapter:
    """A ``ModelAdapter`` double whose Hessian is a positive-definite constant.

    ``E = sum(coords**2)`` gives a Hessian of ``2*I``, so every projected mode is
    real and ``analyze_vibrations`` reports a genuine minimum -- which is what
    ``do_mol_thermo`` needs in order to reach the property writes at all.
    """

    coord_pad = 0.0
    species_pad = -1

    def to_species(self, numbers):
        return numbers

    def energy(self, coords, species, charges, atom_mask=None):
        return coords.pow(2).sum(dim=(1, 2))

    def forward(self, coords, species, charges, atom_mask=None):
        coords = coords if coords.requires_grad else coords.requires_grad_(True)
        energy = self.energy(coords, species, charges, atom_mask)
        grad = torch.autograd.grad([energy.sum()], [coords])[0]
        return energy, -grad

    def analytic_hessian(self, coords, species, charges):
        n = coords.shape[-2] * 3
        return torch.eye(n, dtype=torch.double) * 2.0


def _water_with_stale_energy() -> tuple[Chem.Mol, ase.Atoms]:
    """A record as it leaves the conformer pipeline: geometry plus an ``E_tot``."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", "water_1")
    # What the optimizer wrote for the PRE-relaxation geometry. Written through
    # the owner rather than by hand: `E_tot` is Hartree on disk and eV in
    # memory, and setting the property directly from an eV number is exactly
    # the confusion `utils/energy.py` exists to prevent.
    set_e_tot_from_ev(mol, STALE_ENERGY_EV)

    positions = mol.GetConformer().GetPositions()
    atoms = ase.Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=positions)
    atoms.calc = _FixedEnergyCalculator()
    return mol, atoms


def test_do_mol_thermo_does_not_leave_a_stale_e_tot():
    """After the relaxation, ``E_tot`` must describe the geometry in the file."""
    from Auto3D.entry.ASE.thermo.driver import do_mol_thermo

    mol, atoms = _water_with_stale_energy()

    result = do_mol_thermo(mol, atoms, _HarmonicAdapter(), CPU)

    assert result.HasProp("E_hartree"), "the premise: do_mol_thermo writes E_hartree"
    e_hartree = float(result.GetProp("E_hartree"))

    assert result.HasProp("E_tot"), (
        "E_tot survived from the input SDF but now describes a geometry that "
        "is no longer in the record; it must be updated or cleared"
    )
    assert e_tot_ev(result) == pytest.approx(RELAXED_ENERGY_EV), (
        "E_tot still carries the pre-relaxation energy"
    )
    assert e_tot_ev(result) * EV_TO_HARTREE == pytest.approx(e_hartree), (
        "E_tot and E_hartree disagree for the same coordinates"
    )


def test_do_mol_thermo_clears_the_relative_energy_it_cannot_recompute():
    """``E_rel(kcal/mol)`` must not outlive the ``E_tot`` it was derived from.

    ``ranking.run`` writes ``E_rel(kcal/mol)`` as an energy *relative to the
    best conformer of that molecule*, computed from the pre-relaxation
    ``E_tot``. ``calc_thermo`` then relaxes the geometry and rewrites ``E_tot``,
    which leaves the relative value describing a comparison that no longer
    exists -- the same "one record, two disagreeing energies" defect the
    ``E_tot`` write was added to fix, one property over.

    Clearing rather than recomputing: the quantity is defined against the whole
    conformer group, and ``do_mol_thermo`` sees one molecule at a time, so it
    has no reference to recompute against.
    """
    from Auto3D.entry.ASE.thermo.driver import do_mol_thermo

    mol, atoms = _water_with_stale_energy()
    mol.SetProp("E_rel(kcal/mol)", "1.234")

    result = do_mol_thermo(mol, atoms, _HarmonicAdapter(), CPU)

    assert not result.HasProp("E_rel(kcal/mol)"), (
        "E_rel(kcal/mol) survived a relaxation that replaced the E_tot it was "
        f"computed from: {result.GetProp('E_rel(kcal/mol)')}"
    )
