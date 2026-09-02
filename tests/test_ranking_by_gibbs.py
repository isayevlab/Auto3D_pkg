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

from Auto3D.domain.ranking import RANK_BY_ELECTRONIC, RANK_BY_GIBBS, ConformerRanker
from Auto3D.foundation.constants import HARTREE_TO_EV, HARTREE_TO_KCAL_PER_MOL
from Auto3D.foundation.exceptions import InputValidationError
from Auto3D.foundation.utils.energy import set_e_tot_from_ev


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
        out_path=str(out2),
        threshold=0.3,
        k=1,
        rank_by=RANK_BY_GIBBS,
    ).run()
    assert len(by_g) == 1
    assert float(by_g[0].GetProp("G_hartree")) == pytest.approx(-100.0), (
        "ranking by Gibbs returned the lowest-E conformer, not the lowest-G one"
    )


def test_ranking_by_gibbs_with_k2_returns_the_two_lowest_g_conformers(tmp_path):
    """``k>1`` must truncate on the SELECTED basis, not always on E_tot.

    Three conformers, E strictly ascending (A < B < C) and G strictly
    descending over the same triple (so the E-ascending and G-ascending
    orders are exact reverses of each other -- maximal disagreement).
    ``top_k``'s ``k>1`` branch delegates to ``filter_conformers``, which
    always sorts its ``kept`` list on E_tot (duplicate detection is a
    geometry notion and compares electronic energies unconditionally -- see
    filtering.py); before the fix, ``ranking.py`` truncated that E-ordered
    list to k, so ``k=2`` returned {A, B} -- the two lowest-E conformers,
    dropping C, the actual G-minimum.
    """
    low_e = _mol("m", e_ev=-20.0, g_hartree=-99.0, seed=1)  # A: lowest E, highest G
    mid = _mol("m", e_ev=-15.0, g_hartree=-99.5, seed=13)  # B: middle E, middle G
    low_g = _mol("m", e_ev=-10.0, g_hartree=-100.0, seed=7)  # C: highest E, lowest G

    out = ConformerRanker(
        input_path=_write([low_e, mid, low_g], tmp_path / "in3.sdf"),
        out_path=str(tmp_path / "out3.sdf"),
        threshold=0.3,
        k=2,
        rank_by=RANK_BY_GIBBS,
    ).run()

    g_values = [float(m.GetProp("G_hartree")) for m in out]
    assert g_values == pytest.approx([-100.0, -99.5]), (
        "k=2 on the Gibbs basis must return the two LOWEST-G conformers, in "
        f"ascending G order (expected [-100.0, -99.5], got {g_values}) -- "
        "not the two lowest-E ones"
    )


def test_ranking_by_gibbs_reports_a_gibbs_relative_energy(tmp_path):
    """The published relative energy must name the basis it came from, be
    measured from the G-minimum (never negative, zero on the minimum), and
    reproduce the value a correct upstream ``calc_thermo`` run would have
    written.

    E and G orderings deliberately disagree here (``low_e`` is the E-minimum
    but the G-*maximum* of the pair): giving both conformers the same E_tot,
    as the original version of this test did, cannot distinguish "correct"
    from "referenced to the wrong (E-minimum) conformer" -- with only one
    E_tot value, ranking.py's old ``kept[0]`` (an E-minimum) and the true
    G-minimum happened to coincide half the time, which is exactly what let
    the ranking.py:361 bug (G_rel referenced to the lowest-E conformer, not
    the lowest-G one) through undetected.
    """
    low_e = _mol("m", e_ev=-20.0, g_hartree=-99.0, seed=1)  # E-min, G-max
    low_g = _mol("m", e_ev=-10.0, g_hartree=-100.0, seed=7)  # E-max, G-min
    # Simulate a correct upstream `calc_thermo(relative_gibbs=True)` output:
    # G_rel already correctly measured from the G-minimum (low_g = 0.0).
    expected_g_rel_low_e = (-99.0 - -100.0) * HARTREE_TO_KCAL_PER_MOL
    low_e.SetProp("G_rel(kcal/mol)", str(expected_g_rel_low_e))
    low_g.SetProp("G_rel(kcal/mol)", "0.0")

    out = ConformerRanker(
        input_path=_write([low_e, low_g], tmp_path / "in.sdf"),
        out_path=str(tmp_path / "out.sdf"),
        threshold=0.3,
        k=2,
        rank_by=RANK_BY_GIBBS,
    ).run()

    assert all(m.HasProp("G_rel(kcal/mol)") for m in out)
    assert not any(m.HasProp("E_rel(kcal/mol)") for m in out), (
        "an electronic relative energy was published for a Gibbs ranking"
    )

    g_rel_by_g = {float(m.GetProp("G_hartree")): float(m.GetProp("G_rel(kcal/mol)")) for m in out}
    assert all(v >= 0 for v in g_rel_by_g.values()), (
        f"a relative Gibbs energy went negative: {g_rel_by_g} -- G_rel is "
        "being referenced to the lowest-E conformer instead of the lowest-G one"
    )
    assert g_rel_by_g[-100.0] == pytest.approx(0.0), (
        "the G-minimum conformer must be its own reference (G_rel == 0.0)"
    )
    # ranking.py recomputes G_rel from the SELECTED set rather than trusting
    # whatever the file already carried, but recomputing over the same group
    # (nothing was truncated: k=2 == the whole group) must reproduce the same
    # value a correct upstream run already wrote -- not corrupt it.
    assert g_rel_by_g[-99.0] == pytest.approx(expected_g_rel_low_e)


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
            out_path=str(tmp_path / "out.sdf"),
            threshold=0.3,
            k=1,
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
    """A window on G must not be applied to E, or vice versa -- and the
    ``break`` that stops the scan early must be sound on whichever basis is
    selected.

    Three conformers, E strictly ascending, with a NON-MONOTONE G sequence
    over that same E order: 0.00 / 5.00 / 0.50 kcal/mol. Giving both
    conformers identical E_tot, as the original version of this test did,
    made E-order and G-order trivially compatible with each other and could
    not catch ``top_window``'s ``break`` firing on the wrong (E-sorted)
    order: with 5.00 kcal/mol (outside the window) sorted ahead of 0.50
    kcal/mol (inside it), the old code stopped at the 5.00 entry and silently
    dropped the 0.50 one, which a monotone G sequence can never expose.
    """
    g0 = -99.0
    mols = [
        _mol("m", e_ev=-10.00, g_hartree=g0 + 0.0 / HARTREE_TO_KCAL_PER_MOL, seed=1),
        _mol("m", e_ev=-9.50, g_hartree=g0 + 5.0 / HARTREE_TO_KCAL_PER_MOL, seed=13),
        _mol("m", e_ev=-9.00, g_hartree=g0 + 0.5 / HARTREE_TO_KCAL_PER_MOL, seed=7),
    ]

    out = ConformerRanker(
        input_path=_write(mols, tmp_path / "in.sdf"),
        out_path=str(tmp_path / "out.sdf"),
        threshold=0.3,
        window=2.0,
        rank_by=RANK_BY_GIBBS,
    ).run()

    g_rel = sorted(float(m.GetProp("G_rel(kcal/mol)")) for m in out)
    assert g_rel == pytest.approx([0.0, 0.5], abs=1e-6), (
        "window=2.0 kcal/mol on G must keep the 0.00 and 0.50 kcal/mol members "
        f"(the 5.00 kcal/mol one is genuinely outside the window), got {g_rel}"
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
