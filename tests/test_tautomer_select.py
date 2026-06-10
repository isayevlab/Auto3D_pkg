"""Fast tests for tautomer selection (no NN, no GPU).

These live in a separate module from test_tauto.py because that module carries
a module-level ``pytestmark = pytest.mark.slow``; tests here must run in the
default fast suite.
"""


def test_select_tautomers_groups_by_id(tmp_path):
    """select_tautomers must not crash on pandas 3.x and must keep top-k per id."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Auto3D.tautomer import select_tautomers

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
