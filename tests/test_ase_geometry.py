def test_opt_geometry_names_output_by_model(monkeypatch, tmp_path):
    """Output filename must reflect the model, not always 'userNNP'."""
    import Auto3D.ASE.geometry as geo

    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")  # contents irrelevant; we stub optimizing + supplier

    class _Stub:
        def __init__(self, *a, **k): pass
        def run(self): pass
    monkeypatch.setattr(geo, "optimizing", _Stub)
    monkeypatch.setattr(geo.Chem, "SDMolSupplier", lambda *a, **k: [])
    monkeypatch.setattr(geo.torch.cuda, "is_available", lambda: False)

    out = geo.opt_geometry(str(sdf), "AIMNET")
    assert out.endswith("mols_AIMNET_opt.sdf")
