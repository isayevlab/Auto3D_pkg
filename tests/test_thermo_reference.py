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

    def test_malformed_record_does_not_abort_the_batch(self, job_dir):
        """A None record between two valid ones must be skipped, not crash.

        Two properties of RDKit's ``SDMolSupplier`` shape this input, both
        measured against this repo's version (2025.09.6) rather than assumed:

        1. The corrupt block must sit BETWEEN two valid records, not after the
           last one. A forward iterator (what ``list(...)`` and ``calc_thermo``
           both use) only surfaces an explicit ``None`` for a corrupt record
           when a further valid record follows it; a corrupt trailing record is
           dropped silently, so the guarded path would never fire.

        2. The corrupt block must be a **well-delimited record** -- header,
           counts line, garbage where coordinates belong, ``M  END``, ``$$$$``.
           Loose garbage does not work: on ``"this is not a molecule\\n$$$$\\n"``
           the supplier logs "moving to the beginning of the next molecule" and
           **consumes the following record while resynchronizing**, yielding
           ``[ethanol, None]`` with propanol never handed over at all. The
           assertion below would then be impossible to satisfy for a reason that
           has nothing to do with the code under test.
        """
        from Auto3D.ASE.thermo import calc_thermo

        good1 = job_dir / "good1.sdf"
        good2 = job_dir / "good2.sdf"
        _write_mol(good1, smiles="CCO", name="ethanol")
        _write_mol(good2, smiles="CCCO", name="propanol")

        # A corrupt record that ends cleanly at its own $$$$, so the supplier
        # yields exactly [ethanol, None, propanol]. Verified: the counts line
        # promises two atoms and the atom block does not deliver them.
        corrupt = (
            "corrupt\n"
            "     RDKit          3D\n"
            "\n"
            "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
            "NOT_A_COORD_LINE\n"
            "  1  2  1  0\n"
            "M  END\n"
            "$$$$\n"
        )
        combined = job_dir / "mixed.sdf"
        combined.write_text(good1.read_text() + corrupt + good2.read_text())

        out = calc_thermo(str(combined), "AIMNET", use_gpu=False)

        results = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m]
        # `any(...)` passes even if only one of the two valid records survived
        # -- e.g. a regression that aborts the batch partway through, right
        # after the corrupt record, would still satisfy it. Both good records
        # must independently reach the output with a Gibbs energy, and the
        # corrupt one must not have been counted or duplicated into a third
        # record.
        with_g = {
            m.GetProp("_Name") for m in results if m.HasProp("G_hartree")
        }
        assert with_g == {"ethanol", "propanol"}, (
            f"expected both valid molecules to produce a thermo result, got {with_g}"
        )
        assert len(results) == 2, (
            f"expected exactly the two valid records, got {len(results)}"
        )


class TestStationaryPointGating:
    """G must not be reported for a structure the optimizer did not converge."""

    def test_unconverged_geometry_is_flagged_or_refused(self, job_dir):
        """With opt_steps=1 nothing can converge, so no G may be emitted at all.

        This test used to check `mol.HasProp("Thermo_converged") or
        mol.HasProp("Thermo_warning")` for the (hypothetical) case where a
        structure that failed to converge still got a flagged G -- but
        neither property is ever set anywhere in the source tree, so that
        loop could only fail, never pass on its own merits; it only appeared
        to pass because `with_g` is always empty under the resolution this
        codebase actually implements (`calc_thermo`'s stationary-point gate
        withholds G entirely and sets `Thermo_failed="not_converged"`
        instead). The final non-vacuity check was equally dead: every record
        `_write_thermo_output` writes -- success or failure -- always ends up
        with `Thermo_failed` set (successes get `""` if not already set,
        failures get it set explicitly before being appended), so
        `m.HasProp("G_hartree") or m.HasProp("Thermo_failed")` was true for
        any record calc_thermo could possibly write, regardless of what it
        actually computed.

        Replaced with an assertion of the one behavior calc_thermo
        implements: no G_hartree anywhere, and the record explicitly marked
        `Thermo_failed == "not_converged"`.
        """
        from Auto3D.ASE.thermo import calc_thermo

        # Deliberately unoptimized (no MMFF pass): a raw ETKDG embedding has
        # bond lengths/angles far enough from equilibrium that any reasonable
        # potential (classical or NNP) reports a force well above both the
        # 0.01 eV/A entry gate and the tighter opt_tol -- so this is
        # guaranteed to either skip straight past the gate or fail to
        # converge in a single BFGS step, regardless of which of those two
        # code paths is taken.
        path = _write_mol(job_dir / "raw.sdf", smiles="CCCCO", name="butanol", optimize=False)

        out = calc_thermo(str(path), "AIMNET", opt_steps=1, use_gpu=False)

        results = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m]
        assert results, "calc_thermo produced no output records at all"

        with_g = [m for m in results if m.HasProp("G_hartree")]
        assert not with_g, (
            "G_hartree was reported for a structure that could not have "
            "converged in a single BFGS step; calc_thermo's only implemented "
            "resolution for an unconverged geometry is to withhold G entirely"
        )
        assert all(
            m.GetProp("Thermo_failed") == "not_converged" for m in results
        ), "the unconverged record was not marked Thermo_failed='not_converged'"


class TestHessianGeometry:
    """The Hessian and the energy must be evaluated at the same geometry."""

    def test_hessian_geometry_matches_relaxed_atoms(self, job_dir, monkeypatch):
        """do_mol_thermo's Hessian must come from the same geometry as its energy.

        A tolerance-banded comparison of two independently-computed G values
        is a coin flip: whether they differ by more than an arbitrary
        threshold depends on how far the input happened to sit from the
        model's minimum. Instead, compare a quantity that is EXACTLY
        determined by the code path, with no numerical judgment call: the
        Cartesian geometry the Hessian was built from vs. the geometry the
        same `atoms` object actually holds when the energy is read
        (thermo.py:272). If the Hessian and the energy come from the same
        structure, these must be bit-for-bit identical (the same array
        round-tripped through the same object); if the Hessian is stale, they
        differ by the full relaxation displacement -- typically tenths of an
        Angstrom for a raw, non-force-field-relaxed embedding, many orders of
        magnitude above float64 noise. There is no threshold to tune and no
        way for this to pass by numerical accident.

        This drives `do_mol_thermo` itself -- the sole production caller of
        `vib_hessian` (thermo.py:270) -- rather than calling `vib_hessian`
        directly, and only records what the real, unmodified `vib_hessian`
        returns (via a pass-through spy) rather than asserting on its
        internals. That binds the test to both plausible fix locations: if
        the fix reorders `do_mol_thermo`'s mol-conformer sync to happen
        before it calls `vib_hessian` (the sync currently at :318-320), the
        real `vib_hessian` it calls will see an already-synced `mol` and this
        assertion will pass; if instead the fix changes `vib_hessian` itself
        to source its geometry from `atoms` rather than from `mol`, the
        spied return value reflects that directly too. Either fix location
        makes this XPASS; only calling `vib_hessian` directly (bypassing
        `do_mol_thermo`) would have missed a fix made by reordering the sync.
        """
        from ase.optimize import BFGS

        from Auto3D.ASE import thermo as thermo_mod
        from Auto3D.ASE.thermo import (
            _load_hessian_model,
            mol2atoms,
            model_name2model_calculator,
        )
        from Auto3D.model_factory import get_device

        # Raw ETKDG embedding, no MMFF relaxation: guarantees a large initial
        # force, so BFGS is guaranteed to actually move the atoms before
        # do_mol_thermo calls vib_hessian.
        path = _write_mol(job_dir / "raw.sdf", smiles="CCO", optimize=False)
        mol = next(m for m in Chem.SDMolSupplier(path, removeHs=False) if m)

        device = get_device(0, use_gpu=False)
        hessian_model = _load_hessian_model("AIMNET", device)
        _, calculator = model_name2model_calculator("AIMNET", device)
        calculator.set_charge(Chem.GetFormalCharge(mol))

        # This is exactly what calc_thermo's optimize branch does: BFGS
        # mutates `atoms` in place. mol's RDKit conformer is untouched -- the
        # only sync back into mol happens inside do_mol_thermo, at the very
        # end (thermo.py:318-320).
        atoms = mol2atoms(mol)
        atoms.set_calculator(calculator)
        opt = BFGS(atoms)
        opt.run(fmax=3e-3, steps=100)

        # Spy on the module-level vib_hessian: call straight through to the
        # real implementation and record the geometry it actually built the
        # Hessian from. do_mol_thermo resolves `vib_hessian` as a bare global
        # at call time (thermo.py:270), so patching the module attribute
        # intercepts that specific call without altering its behavior.
        real_vib_hessian = thermo_mod.vib_hessian
        captured: dict[str, np.ndarray] = {}

        def _spy(*args, **kwargs):
            vib = real_vib_hessian(*args, **kwargs)
            # ``VibrationsData`` stores its Atoms privately as ``_atoms`` and
            # exposes it only through ``get_atoms()``, which returns a copy.
            # This spy originally read ``vib.atoms`` -- an attribute ASE has
            # never had -- so the test raised AttributeError on its very first
            # execution rather than checking anything. It is slow-marked and
            # needs a loaded potential, so that first execution was in CI,
            # long after it was written.
            captured["positions"] = vib.get_atoms().get_positions().copy()
            return vib

        monkeypatch.setattr(thermo_mod, "vib_hessian", _spy)

        thermo_mod.do_mol_thermo(mol, atoms, hessian_model, device, model_name="AIMNET")

        assert "positions" in captured, "do_mol_thermo never called vib_hessian"
        # The geometry the Hessian was computed at must be the geometry the
        # energy (and moments of inertia) were computed at: the same `atoms`
        # object, post-relaxation.
        assert np.allclose(captured["positions"], atoms.get_positions(), atol=1e-8), (
            "do_mol_thermo's Hessian was evaluated at a different geometry "
            "than the relaxed atoms object it read the energy from -- the "
            "Hessian is stale relative to the reported energy"
        )
