"""``E_tot`` is Hartree at every writer, and every reader agrees.

Before this, ``batch_opt.optimizing.run`` wrote ``E_tot`` in **eV** and
``ASE/geometry.opt_geometry`` rewrote the same tag in **Hartree**, while the
in-package consumers (``ranking`` and both filters in ``filtering``) all
hard-coded eV. The property name meant two different things depending on which
entry point produced the file, and nothing in the file said which.

The consequence these tests pin is the one a user hits without doing anything
unusual: feed an ``opt_geometry`` output to ``ConformerRanker(window=2.0)``.
Reading a Hartree number as eV makes the window 27.211x too wide, so a
3-conformer set that should yield 2 yields 3, ``E_rel`` reads 0.037 kcal/mol
where the truth is 1.000, and the ranker's own eV->Hartree conversion runs on
an already-Hartree number, dividing by 27.211 twice.

No neural network potential is loaded: ``ensemble_opt`` is stubbed and the
adapter is a conforming double, so the real padder, the real ``optimizing.run``
writer, the real ``_annotate_and_rewrite`` and the real ranker all execute.

``optimizing`` no longer builds its own adapter (audit M41), so the direct tests
inject one and the ``opt_geometry`` tests stub ``create_model`` where
``opt_geometry`` itself reads it -- ``Auto3D.ASE.geometry`` -- rather than at a
seam inside ``batch_opt``.
"""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_EV
from Auto3D.utils.energy import E_TOT_HARTREE_PROP, E_TOT_PROP, e_tot_ev

#: Three conformers of one species, 0 / 1 / 3 kcal/mol apart. A 2 kcal/mol
#: window must admit the first two and refuse the third.
BASE_EV = -10.0
ENERGIES_EV = [
    BASE_EV,
    BASE_EV + 1.0 / EV_TO_KCAL_PER_MOL,
    BASE_EV + 3.0 / EV_TO_KCAL_PER_MOL,
]


def _write_input(path, names) -> None:
    """Three embedded conformers of ethanol, named <species>_<isomer>_<conf>."""
    with Chem.SDWriter(str(path)) as w:
        for i, name in enumerate(names):
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(mol, randomSeed=i + 1)
            mol.SetProp("_Name", name)
            w.write(mol)


def _stub_model_boundary(monkeypatch, energies_ev):
    """Replace the NNP with a table of energies; everything else stays real.

    Returns the conforming adapter double, for callers that construct
    ``optimizing`` directly. ``Auto3D.ASE.geometry.create_model`` is stubbed to
    hand back the same object, because that is where ``opt_geometry`` now builds
    the adapter it injects.
    """
    import Auto3D.ASE.geometry as geo
    import Auto3D.batch_opt.batchopt as bo
    from tests.helpers_adapter import FakeAdapter

    def fake_ensemble_opt(
        net, coord, numbers, charges, param, device, atom_mask=None, progress_cb=None
    ):
        n = len(coord)
        return dict(
            coord=coord.tolist(),
            ids=list(range(n)),
            energy=list(energies_ev[:n]),
            fmax=[0.0] * n,
            he=[],
            close=[],
            timing={},
            numbers=numbers.tolist(),
            converged_mask=[True] * n,
            oscillating_count=[0] * n,
        )

    adapter = FakeAdapter()
    monkeypatch.setattr(bo, "ensemble_opt", fake_ensemble_opt)
    monkeypatch.setattr(geo, "create_model", lambda *a, **k: adapter)
    return adapter


class TestOptimizerWritesHartree:
    """``optimizing.run`` is the first writer; it owns the conversion."""

    def test_e_tot_is_the_model_energy_in_hartree(self, tmp_path, monkeypatch):
        import Auto3D.batch_opt.batchopt as bo

        adapter = _stub_model_boundary(monkeypatch, ENERGIES_EV)
        inp = tmp_path / "in.sdf"
        out = tmp_path / "out.sdf"
        _write_input(inp, ["spec_0_0", "spec_0_1", "spec_0_2"])

        bo.optimizing(
            str(inp),
            str(out),
            adapter=adapter,
            device=torch.device("cpu"),
            config={"opt_steps": 1, "opttol": 0.01, "patience": 1, "batchsize_atoms": 1024},
        ).run()

        mols = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m]
        assert len(mols) == 3
        for mol, energy_ev in zip(mols, ENERGIES_EV, strict=True):
            stored = float(mol.GetProp(E_TOT_PROP))
            # Hartree, not eV: the eV value is 27.211x larger in magnitude.
            assert stored == pytest.approx(energy_ev / HARTREE_TO_EV, rel=1e-9)
            assert e_tot_ev(mol) == pytest.approx(energy_ev, rel=1e-9)


class TestOptGeometryDoesNotConvertTwice:
    """``opt_geometry`` annotates the unit; it must not re-divide."""

    def test_output_energy_is_the_model_energy_in_hartree(self, tmp_path, monkeypatch):
        import Auto3D.ASE.geometry as geo

        _stub_model_boundary(monkeypatch, ENERGIES_EV)
        inp = tmp_path / "mols.sdf"
        _write_input(inp, ["spec_0_0", "spec_0_1", "spec_0_2"])

        out = geo.opt_geometry(str(inp), "AIMNET", use_gpu=False)

        mols = [m for m in Chem.SDMolSupplier(out, removeHs=False) if m]
        assert len(mols) == 3
        for mol, energy_ev in zip(mols, ENERGIES_EV, strict=True):
            assert float(mol.GetProp(E_TOT_PROP)) == pytest.approx(
                energy_ev / HARTREE_TO_EV, rel=1e-9
            )
            # The property name states its unit, and states it truthfully.
            assert mol.GetProp(E_TOT_HARTREE_PROP) == mol.GetProp(E_TOT_PROP)


class TestOptGeometryOutputRanksCorrectly:
    """The chain the brief demonstrates: opt_geometry -> ConformerRanker."""

    def test_a_2_kcal_window_admits_two_of_three(self, tmp_path, monkeypatch):
        import Auto3D.ASE.geometry as geo
        from Auto3D.ranking import ConformerRanker

        _stub_model_boundary(monkeypatch, ENERGIES_EV)
        inp = tmp_path / "mols.sdf"
        _write_input(inp, ["spec_0_0", "spec_0_1", "spec_0_2"])
        optimized = geo.opt_geometry(str(inp), "AIMNET", use_gpu=False)

        ranked = tmp_path / "ranked.sdf"
        results = ConformerRanker(
            input_path=optimized,
            out_path=str(ranked),
            threshold=0.3,
            window=2.0,
        ).run()

        # 2, not 3: the third conformer is 3 kcal/mol up, outside a 2 kcal/mol
        # window. Reading Hartree as eV widens the window 27.211x and keeps it.
        assert len(results) == 2

        rel = [float(m.GetProp("E_rel(kcal/mol)")) for m in results]
        assert rel[0] == pytest.approx(0.0, abs=1e-9)
        # 1.000, not 0.037.
        assert rel[1] == pytest.approx(1.0, abs=1e-6)

    def test_ranked_energy_is_not_divided_twice(self, tmp_path, monkeypatch):
        import Auto3D.ASE.geometry as geo
        from Auto3D.ranking import ConformerRanker

        _stub_model_boundary(monkeypatch, ENERGIES_EV)
        inp = tmp_path / "mols.sdf"
        _write_input(inp, ["spec_0_0", "spec_0_1", "spec_0_2"])
        optimized = geo.opt_geometry(str(inp), "AIMNET", use_gpu=False)

        ranked = tmp_path / "ranked.sdf"
        ConformerRanker(
            input_path=optimized,
            out_path=str(ranked),
            threshold=0.3,
            k=1,
        ).run()

        out = [m for m in Chem.SDMolSupplier(str(ranked), removeHs=False) if m]
        assert len(out) == 1
        expected_hartree = min(ENERGIES_EV) / HARTREE_TO_EV
        assert float(out[0].GetProp(E_TOT_PROP)) == pytest.approx(expected_hartree, rel=1e-9)
        # Dividing by 27.211 a second time would land here instead.
        assert float(out[0].GetProp(E_TOT_PROP)) != pytest.approx(
            expected_hartree / HARTREE_TO_EV, rel=1e-3
        )
        assert out[0].GetProp(E_TOT_HARTREE_PROP) == out[0].GetProp(E_TOT_PROP)


class TestTautomerSelectionReadsTheSameUnit:
    """``select_tautomers`` has always read Hartree; now every writer agrees."""

    def test_relative_tautomer_energy_is_kcal_per_mol(self, tmp_path, monkeypatch):
        import Auto3D.batch_opt.batchopt as bo
        from Auto3D.tautomer import select_tautomers

        adapter = _stub_model_boundary(monkeypatch, ENERGIES_EV[:2])
        inp = tmp_path / "in.sdf"
        out = tmp_path / "opt.sdf"
        _write_input(inp, ["id1@taut0_0_0", "id1@taut1_0_0"])
        bo.optimizing(
            str(inp),
            str(out),
            adapter=adapter,
            device=torch.device("cpu"),
            config={"opt_steps": 1, "opttol": 0.01, "patience": 1, "batchsize_atoms": 1024},
        ).run()

        selected = select_tautomers(str(out), k=2)
        mols = [m for m in Chem.SDMolSupplier(selected, removeHs=False) if m]
        rel = sorted(float(m.GetProp("E_tautomer_relative(kcal/mol)")) for m in mols)
        assert rel[0] == pytest.approx(0.0, abs=1e-9)
        assert rel[1] == pytest.approx(1.0, abs=1e-6)
