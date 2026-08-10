"""Ranking on Gibbs free energy, opt-in.

Conformer selection is electronic by default and must stay that way: a Gibbs
energy exists only after a thermochemistry run, which costs a Hessian per
conformer, so a default that depended on one would turn the cheap path into the
slow one.

But once a user *has* paid for that -- a ``calc_thermo`` output -- selecting on
G is the more defensible choice, because a population goes as ``exp(-dG/RT)``
and the lowest-G conformer need not be the lowest-E one. ``ConformerRanker``
therefore takes a basis, defaulting to the electronic energy.

Nothing here loads a neural network potential.
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.constants import HARTREE_TO_EV
from Auto3D.exceptions import InputValidationError
from Auto3D.ranking import RANK_BY_ELECTRONIC, RANK_BY_GIBBS, ConformerRanker
from Auto3D.utils.energy import set_e_tot_from_ev


def _mol(name, e_ev, g_hartree=None, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    mol.SetProp("_Name", name)
    set_e_tot_from_ev(mol, e_ev)
    mol.SetProp("Converged", "True")
    if g_hartree is not None:
        mol.SetProp("G_hartree", str(g_hartree))
        mol.SetProp("T_K", "298.15")
    return mol


def _write(mols, path):
    with Chem.SDWriter(str(path)) as w:
        for m in mols:
            w.write(m)
    return str(path)


def _ranker(tmp_path, mols, **kwargs):
    return ConformerRanker(
        input_path=_write(mols, tmp_path / "in.sdf"),
        out_path=str(tmp_path / "out.sdf"),
        threshold=0.3,
        **kwargs,
    )


def test_the_default_basis_is_the_electronic_energy():
    """Selection must not acquire a thermochemistry dependency by default."""
    import inspect

    default = inspect.signature(ConformerRanker.__init__).parameters["rank_by"].default
    assert default == RANK_BY_ELECTRONIC


def test_ranking_by_gibbs_can_pick_a_different_conformer_than_energy(tmp_path):
    """The whole point: the G-minimum need not be the E-minimum.

    Conformer A is lower in electronic energy, B is lower in G. ``k=1`` must
    return A on the default basis and B on the Gibbs one, from the same input.
    """
    low_e = _mol("m", e_ev=-20.0, g_hartree=-99.0, seed=1)
    low_g = _mol("m", e_ev=-10.0, g_hartree=-100.0, seed=7)

    by_e = _ranker(tmp_path, [low_e, low_g], k=1).run()
    assert len(by_e) == 1
    assert float(by_e[0].GetProp("E_tot")) == pytest.approx(-20.0 / HARTREE_TO_EV)

    out2 = tmp_path / "out2.sdf"
    by_g = ConformerRanker(
        input_path=_write([low_e, low_g], tmp_path / "in2.sdf"),
        out_path=str(out2), threshold=0.3, k=1, rank_by=RANK_BY_GIBBS,
    ).run()
    assert len(by_g) == 1
    assert float(by_g[0].GetProp("G_hartree")) == pytest.approx(-100.0), (
        "ranking by Gibbs returned the lowest-E conformer, not the lowest-G one"
    )


def test_ranking_by_gibbs_reports_a_gibbs_relative_energy(tmp_path):
    """The published relative energy must name the basis it came from."""
    mols = [_mol("m", e_ev=-10.0, g_hartree=-100.0, seed=1),
            _mol("m", e_ev=-10.0, g_hartree=-99.5, seed=7)]

    out = ConformerRanker(
        input_path=_write(mols, tmp_path / "in.sdf"),
        out_path=str(tmp_path / "out.sdf"), threshold=0.3, k=2,
        rank_by=RANK_BY_GIBBS,
    ).run()

    assert all(m.HasProp("G_rel(kcal/mol)") for m in out)
    assert not any(m.HasProp("E_rel(kcal/mol)") for m in out), (
        "an electronic relative energy was published for a Gibbs ranking"
    )


def test_the_default_basis_still_reports_the_electronic_relative_energy(tmp_path):
    mols = [_mol("m", e_ev=-10.0, seed=1), _mol("m", e_ev=-9.5, seed=7)]

    out = _ranker(tmp_path, mols, k=2).run()

    assert all(m.HasProp("E_rel(kcal/mol)") for m in out)
    assert not any(m.HasProp("G_rel(kcal/mol)") for m in out)


def test_a_missing_gibbs_energy_is_refused_with_a_usable_message(tmp_path):
    """The likely mistake: ranking an optimizer output on G."""
    mols = [_mol("m", e_ev=-10.0)]  # no G_hartree

    with pytest.raises(InputValidationError) as excinfo:
        ConformerRanker(
            input_path=_write(mols, tmp_path / "in.sdf"),
            out_path=str(tmp_path / "out.sdf"), threshold=0.3, k=1,
            rank_by=RANK_BY_GIBBS,
        ).run()

    assert "G_hartree" in str(excinfo.value)
    # The remedy lives on `hint`, which is what the CLI renders beneath the
    # message -- not in str(), so assert on the attribute the user actually sees.
    assert "calc_thermo" in excinfo.value.hint, (
        "the hint should say where a Gibbs energy comes from"
    )


def test_an_unknown_basis_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="rank_by"):
        _ranker(tmp_path, [_mol("m", -10.0)], k=1, rank_by="entropy")


def test_the_window_is_measured_on_the_selected_basis(tmp_path):
    """A 1 kcal/mol window on G must not be applied to E, or vice versa."""
    # Identical electronic energies; G differs by ~3 kcal/mol (0.00478 Hartree).
    mols = [_mol("m", e_ev=-10.0, g_hartree=-100.0, seed=1),
            _mol("m", e_ev=-10.0, g_hartree=-99.99522, seed=7)]

    out = ConformerRanker(
        input_path=_write(mols, tmp_path / "in.sdf"),
        out_path=str(tmp_path / "out.sdf"), threshold=0.3, window=1.0,
        rank_by=RANK_BY_GIBBS,
    ).run()

    assert len(out) == 1, (
        "the second conformer is ~3 kcal/mol above in G and outside a 1 kcal/mol "
        f"window, but {len(out)} were kept"
    )


def test_the_default_path_has_no_thermochemistry_dependency(tmp_path):
    """The property the default protects: ranking must work without any G.

    An optimizer output carries no ``G_hartree`` at all. If selection ever
    defaulted to Gibbs -- or silently fell back to it -- the cheap path would
    require a Hessian per conformer to produce a result. Ranking a G-free file
    with default settings must simply work.

    This replaces an earlier check that grepped ``top_k`` for ``e_tot_ev``. That
    was a proxy for the same property and it broke the moment the read was
    routed through a basis, without anything actually regressing.
    """
    mols = [_mol("m", e_ev=-10.0, seed=1), _mol("m", e_ev=-9.0, seed=7)]
    assert not any(m.HasProp("G_hartree") for m in mols)

    out = _ranker(tmp_path, mols, k=1).run()

    assert len(out) == 1
    assert float(out[0].GetProp("E_tot")) == pytest.approx(-10.0 / HARTREE_TO_EV)
