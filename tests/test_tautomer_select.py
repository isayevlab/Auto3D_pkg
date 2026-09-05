"""Fast tests for tautomer selection (no NN, no GPU).

These live in a separate module from test_tauto.py because that module carries
a module-level ``pytestmark = pytest.mark.slow``; tests here must run in the
default fast suite.
"""


def test_select_tautomers_groups_by_id(tmp_path):
    """select_tautomers must not crash on pandas 3.x and must keep top-k per id."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for name, e in [("molA@taut1", -1.0), ("molA@taut2", -0.5), ("molB@taut1", -2.0)]:
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", name)
            m.SetProp("E_tot", str(e))
            m.SetProp("E_rel(kcal/mol)", "0.0")
            w.write(m)

    out = select_tautomers(str(sdf), k=1)
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    names = sorted(m.GetProp("_Name") for m in mols)
    assert names == ["molA", "molB"]  # one top tautomer per id
    assert not any(m.HasProp("E_rel(kcal/mol)") for m in mols)

    # Names alone don't prove *which* tautomer survived: after selection both
    # groups are renamed to their bare id, so a reversed sort (highest energy
    # kept instead of lowest) would still produce ["molA", "molB"] here. E_tot
    # is not cleared on write, so check it directly: molA must keep taut1
    # (-1.0, the lower/more stable energy), not taut2 (-0.5).
    by_name = {m.GetProp("_Name"): m for m in mols}
    assert float(by_name["molA"].GetProp("E_tot")) == -1.0
    assert float(by_name["molB"].GetProp("E_tot")) == -2.0
    # The kept tautomer is each group's own reference, so its relative energy
    # to itself must be exactly zero.
    assert float(by_name["molA"].GetProp("E_tautomer_relative(kcal/mol)")) == 0.0
    assert float(by_name["molB"].GetProp("E_tautomer_relative(kcal/mol)")) == 0.0


def test_select_tautomers_does_not_cross_rank_different_species_sharing_an_id(tmp_path, caplog):
    """Issue 12: acetic acid and acetate sharing one base id must both survive
    selection, each ranked against its own species/charge partition -- not
    cross-ranked, which always let the neutral member "win" by hundreds of
    kcal/mol and silently defeated tautomer/pKa enumeration's whole point.
    """
    import logging

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        # Neutral acetic acid: much higher (less negative) E_tot than the
        # anion below, exactly the "always wins on a raw energy comparison"
        # shape the review flagged.
        acid = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
        AllChem.EmbedMolecule(acid, randomSeed=1)
        acid.SetProp("_Name", "acetic_acid@taut1")
        acid.SetProp("E_tot", "-227.0")
        w.write(acid)

        # Acetate anion, same base id -- a pKa-normalized conjugate base kept
        # under one id by tautomer enumeration.
        acetate = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
        AllChem.EmbedMolecule(acetate, randomSeed=1)
        acetate.SetProp("_Name", "acetic_acid@taut2")
        acetate.SetProp("E_tot", "-226.5")  # higher electronic energy than the acid
        w.write(acetate)

    with caplog.at_level(logging.WARNING, logger="Auto3D.entry.tautomer"):
        out = select_tautomers(str(sdf), k=1)

    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    charges = sorted(Chem.GetFormalCharge(m) for m in mols)
    assert charges == [-1, 0], (
        f"both species must survive selection under separate partitions, got charges {charges}"
    )
    # Each partition is its own reference (only one member per partition
    # here), so both keep a relative energy of exactly zero.
    assert all(float(m.GetProp("E_tautomer_relative(kcal/mol)")) == 0.0 for m in mols)
    assert any("distinct species" in r.message for r in caplog.records), (
        "the species split must be logged, naming the group"
    )


def test_select_tautomers_same_species_case_is_unchanged(tmp_path):
    """Negative control for the fix above: a group that is genuinely one
    species/charge state must still be ranked and truncated to top-k exactly
    as before (see test_select_tautomers_groups_by_id for the full check)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for name, e in [("molA@taut1", -1.0), ("molA@taut2", -0.5)]:
            m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            AllChem.EmbedMolecule(m, randomSeed=1)
            m.SetProp("_Name", name)
            m.SetProp("E_tot", str(e))
            w.write(m)

    out = select_tautomers(str(sdf), k=1)
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    assert len(mols) == 1  # top-1 within the single (same-species) partition
    assert float(mols[0].GetProp("E_tot")) == -1.0


def test_select_tautomers_ranks_across_genuine_tautomers(tmp_path):
    """Regression for the slow-suite failure in test_tauto.py::test_get_stable_tautomers2.

    Keto (``CC(=O)C``, acetone) and enol (``CC(O)=C``, prop-1-en-2-ol) share
    one base id, as real tautomer enumeration output does. They are
    constitutional isomers of each other by definition -- same formula
    (C3H6O), same formal charge (0), different connectivity -- which is
    exactly the case a connectivity-sensitive partition key (a canonical
    SMILES, e.g. ``species_key``) gets wrong: it would put each in its own
    singleton partition, so *both* survive a ``k=1`` selection instead of only
    the lower-energy one. That silently turns "select the most stable
    tautomer" into "keep every tautomer", and is what let a bare enol (no
    C=O) reach test_tauto.py's output alongside its keto partner.

    The partition key must instead be (element composition, formal charge)
    only, so these two stay comparable and only the keto form (given the
    lower energy here) survives.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        keto = Chem.AddHs(Chem.MolFromSmiles("CC(=O)C"))
        AllChem.EmbedMolecule(keto, randomSeed=1)
        keto.SetProp("_Name", "acetone@taut1")
        keto.SetProp("E_tot", "-1.0")  # lower (more stable)
        w.write(keto)

        enol = Chem.AddHs(Chem.MolFromSmiles("CC(O)=C"))
        AllChem.EmbedMolecule(enol, randomSeed=1)
        enol.SetProp("_Name", "acetone@taut2")
        enol.SetProp("E_tot", "-0.5")  # higher (less stable)
        w.write(enol)

    out = select_tautomers(str(sdf), k=1)
    mols = list(Chem.SDMolSupplier(out, removeHs=False))
    assert len(mols) == 1, (
        "keto and enol share formula and charge and must be ranked against "
        "each other, not each kept as its own partition's sole survivor"
    )
    substructure = Chem.MolFromSmarts("C=O")
    assert mols[0].HasSubstructMatch(substructure), (
        "the surviving tautomer must be the lower-energy keto form"
    )
    assert float(mols[0].GetProp("E_tautomer_relative(kcal/mol)")) == 0.0


def test_select_tautomers_rejects_nonpositive_k(tmp_path):
    """k < 1 used to silently drop every tautomer (out_mols0[:0]); now rejected."""
    import pytest
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers
    from Auto3D.foundation.exceptions import ConfigurationError

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "molA@taut1")
        m.SetProp("E_tot", "-1.0")
        w.write(m)

    with pytest.raises(ConfigurationError, match="tauto_k"):
        select_tautomers(str(sdf), k=0)


def test_select_tautomers_rejects_k_and_window_together(tmp_path):
    """M29: this used to be a bare ValueError, not an Auto3DError -- so the
    CLI's differentiated exit code/hint never applied to a direct Python API
    call, only to the CLI's own pre-check in execute_tautomers."""
    import pytest
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers
    from Auto3D.foundation.exceptions import ConfigurationError

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "molA@taut1")
        m.SetProp("E_tot", "-1.0")
        w.write(m)

    with pytest.raises(ConfigurationError, match="Only k OR window"):
        select_tautomers(str(sdf), k=1, window=2.0)


def test_select_tautomers_rejects_neither_k_nor_window(tmp_path):
    """Same M29 gap as above, for the neither-given branch."""
    import pytest
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from Auto3D.entry.tautomer import select_tautomers
    from Auto3D.foundation.exceptions import ConfigurationError

    sdf = tmp_path / "in.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "molA@taut1")
        m.SetProp("E_tot", "-1.0")
        w.write(m)

    with pytest.raises(ConfigurationError, match="Either k OR window"):
        select_tautomers(str(sdf))
