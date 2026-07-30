"""Thermochemistry must be computed at a converged stationary point.

BFGS mutates the ASE atoms in place, but mol's conformer is synced only at the
end of do_mol_thermo -- so vib_hessian re-reads pre-optimization coordinates
while the energy and moments of inertia come from the relaxed structure (C5).
Nothing checks opt.run()'s return value, so G is reported for structures the
optimizer never converged (M8). And one unparseable SDF record aborts a batch
that may already have computed hundreds of Hessians (M13).
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

pytest.importorskip("ase")
pytestmark = pytest.mark.slow


def _write_mol(path, smiles="CCO", name="ethanol", optimize=True):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    if optimize:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    mol.SetProp("_Name", name)
    with Chem.SDWriter(str(path)) as w:
        w.write(mol)
    return str(path)


class TestBatchRobustness:
    """One malformed record must not destroy a batch of Hessians."""

    @pytest.mark.xfail(
        strict=True,
        reason="M13: SDMolSupplier yields a None entry for an unparseable "
        "record, and GetConformer(), GetProp('_Name') and set_calculator all "
        "run before the try: at thermo.py:457 -- SPE.py:73-82 filters None "
        "entries for exactly this reason, thermo.py does not",
    )
    def test_malformed_record_does_not_abort_the_batch(self, job_dir):
        """A None record between two valid ones must be skipped, not crash.

        The corrupt block must sit BETWEEN two valid records, not after the
        last one: verified against this repo's RDKit (2025.09.6), a forward
        ``SDMolSupplier`` iterator (what ``list(...)`` and ``calc_thermo``
        itself use) only ever surfaces an explicit ``None`` for a corrupt
        record when a further valid record follows it in the file. A corrupt
        trailing record with nothing after it is silently dropped without
        producing a ``None`` at all, so the code path under test (an
        unguarded ``None.GetConformer()``) would never fire and the test
        would pass today for the wrong reason.
        """
        from Auto3D.ASE.thermo import calc_thermo

        good1 = job_dir / "good1.sdf"
        good2 = job_dir / "good2.sdf"
        _write_mol(good1, smiles="CCO", name="ethanol")
        _write_mol(good2, smiles="CCCO", name="propanol")

        # Sandwich a deliberately corrupt record between two valid ones.
        combined = job_dir / "mixed.sdf"
        combined.write_text(
            good1.read_text() + "this is not a molecule\n$$$$\n" + good2.read_text()
        )

        out = calc_thermo(str(combined), "AIMNET", use_gpu=False)

        results = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m]
        assert any(m.HasProp("G_hartree") for m in results), (
            "the valid molecules produced no thermo result"
        )


class TestStationaryPointGating:
    """G must not be reported for a structure the optimizer did not converge."""

    @pytest.mark.xfail(
        strict=True,
        reason="M8: opt.run()'s return value is never checked, so a geometry "
        "that exhausts opt_steps proceeds to a Hessian and its G is reported "
        "as if converged; opt_tol is used only in the except ValueError "
        "fallback, and the entry gate at thermo.py:464 hardcodes 0.01",
    )
    def test_unconverged_geometry_is_flagged_or_refused(self, job_dir):
        """With opt_steps=1 nothing can converge, so no G may be emitted unflagged."""
        from Auto3D.ASE.thermo import calc_thermo

        # Deliberately unoptimized (no MMFF pass): a raw ETKDG embedding has
        # bond lengths/angles far enough from equilibrium that any reasonable
        # potential (classical or NNP) reports a force well above both the
        # 0.01 eV/A entry gate and the tighter opt_tol -- so this is
        # guaranteed to either skip straight past the gate or fail to
        # converge in a single BFGS step, regardless of which of those two
        # code paths is taken.
        path = _write_mol(job_dir / "raw.sdf", smiles="CCCCO", optimize=False)

        out = calc_thermo(str(path), "AIMNET", opt_steps=1, use_gpu=False)

        for mol in Chem.SDMolSupplier(str(out), removeHs=False):
            if mol is None:
                continue
            if mol.HasProp("G_hartree"):
                assert mol.HasProp("Thermo_converged") or mol.HasProp("Thermo_warning"), (
                    "G was reported for a structure that could not have converged in "
                    "one step, with no flag distinguishing it"
                )


class TestHessianGeometry:
    """The Hessian and the energy must be evaluated at the same geometry."""

    @pytest.mark.xfail(
        strict=True,
        reason="C5: vib_hessian re-reads mol.GetConformer() at thermo.py:225 "
        "-- the geometry as it stood before BFGS ran -- instead of the "
        "positions actually held by the atoms object it was handed, whose "
        "energy is read at thermo.py:272 after BFGS has moved it",
    )
    def test_hessian_geometry_matches_relaxed_atoms(self, job_dir):
        """vib_hessian must evaluate the Hessian at atoms' current geometry.

        A tolerance-banded comparison of two independently-computed G values
        is a coin flip: whether they differ by more than an arbitrary
        threshold depends on how far the input happened to sit from the
        model's minimum. Instead, compare a quantity that is EXACTLY
        determined by the code path, with no numerical judgment call: the
        Cartesian geometry vib_hessian's returned VibrationsData was built
        from vs. the geometry the same `atoms` object actually holds after
        BFGS has moved it. If the Hessian and the energy come from the same
        structure, these must be bit-for-bit identical (the same array
        round-tripped through the same object); if vib_hessian re-reads a
        stale mol conformer, they will differ by the full relaxation
        displacement -- typically tenths of an Angstrom for a raw,
        non-force-field-relaxed embedding, i.e. many orders of magnitude
        above float64 noise. There is no threshold to tune and no way for
        this to pass by numerical accident.
        """
        from ase.optimize import BFGS

        from Auto3D.ASE.thermo import (
            _load_hessian_model,
            mol2atoms,
            model_name2model_calculator,
            vib_hessian,
        )
        from Auto3D.model_factory import get_device

        # Raw ETKDG embedding, no MMFF relaxation: guarantees a large initial
        # force, so BFGS is guaranteed to actually move the atoms before
        # vib_hessian is ever called.
        path = _write_mol(job_dir / "raw.sdf", smiles="CCO", optimize=False)
        mol = next(m for m in Chem.SDMolSupplier(path, removeHs=False) if m)

        device = get_device(0, use_gpu=False)
        hessian_model = _load_hessian_model("AIMNET", device)
        _, calculator = model_name2model_calculator("AIMNET", device)
        calculator.set_charge(Chem.GetFormalCharge(mol))

        # This is exactly what calc_thermo's optimize branch does: BFGS
        # mutates `atoms` in place. mol's RDKit conformer is untouched.
        atoms = mol2atoms(mol)
        atoms.set_calculator(calculator)
        opt = BFGS(atoms)
        opt.run(fmax=3e-3, steps=100)

        vib = vib_hessian(mol, atoms.get_calculator(), hessian_model, device,
                          model_name="AIMNET")

        # The geometry the Hessian was computed at must be the geometry the
        # energy (and moments of inertia) were computed at: the same `atoms`
        # object, post-relaxation.
        assert np.allclose(vib.atoms.get_positions(), atoms.get_positions(), atol=1e-8), (
            "vib_hessian's Hessian was evaluated at a different geometry than "
            "the relaxed atoms object -- the Hessian is stale relative to the "
            "reported energy"
        )
