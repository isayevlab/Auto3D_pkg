"""Every reader of an SDF agrees on what a record is, on one fixture.

The fixture has one unparseable, one implicit-hydrogen, one dummy-atom, and two good
records. The single-file entry points skip the defective ones (calc_thermo keeps and
marks them); main()'s SDF engine adds hydrogens and re-embeds, so for it only the
dummy-atom record is defective. That divergence is deliberate and pinned here.
"""

from __future__ import annotations

from rdkit import Chem

from Auto3D.foundation.utils.sdf_io import classify_records
from tests.helpers_records import write_mixed_sdf


def test_classify_records_is_the_baseline(tmp_path):
    """What every other row in this module is compared against."""
    classified = classify_records(str(write_mixed_sdf(tmp_path / "mixed.sdf")))
    assert classified.names() == ["ethanol", "ethane"]


def test_select_tautomers_keeps_exactly_the_kept_records(tmp_path):
    """Tautomer selection reads through `iter_conformer_records`, so it sees
    exactly the classifier's `kept` records.

    Each fixture name carries no ``@tautN`` suffix, so every record is its own
    tautomer group and a ``k=1`` selection keeps all of them -- which makes the
    output names a direct readout of what the reader accepted.
    """
    from Auto3D.entry.tautomer import select_tautomers

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = select_tautomers(str(path), k=1)

    names = [m.GetProp("_Name") for m in Chem.SDMolSupplier(out, removeHs=False) if m is not None]
    kept = classify_records(str(path)).names()
    assert set(names) <= set(kept), f"a record the classifier rejected survived: {names}"
    assert "skeleton" not in names, "a heavy-atom skeleton was ranked by electronic energy"
    assert "frag" not in names, "an R-group placeholder was ranked as a species"


def test_the_sdf_isomer_engine_re_embeds_implicit_h_and_skips_only_the_dummy(tmp_path, caplog):
    """``main()``'s SDF path is the one reader an implicit-H record is fine for.

    ``RDKitSdfIsomer`` calls ``Chem.AddHs`` and re-embeds every record it keeps,
    so a heavy-atom skeleton is legitimate input rather than a defect -- the
    deliberate divergence from the single-file entry points above. Only the
    dummy-atom record is refused (N-M3: adding hydrogens cannot turn an R-group
    placeholder into a species), and the unparseable one is reported once.
    """
    import logging

    from Auto3D.engines.isomers import IsomerEngineFactory

    path = write_mixed_sdf(tmp_path / "mixed.sdf")
    out = tmp_path / "enumerated.sdf"
    with caplog.at_level(logging.WARNING, logger="Auto3D"):
        IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(path),
            output_path=str(out),
            max_confs=1,
            threshold=0.3,
            n_jobs=1,
            enumerate_isomers=False,
        ).run()

    # Names are <species>_<isomer>_<conformer>; no fixture name contains '_'.
    prefixes = {
        m.GetProp("_Name").split("_")[0]
        for m in Chem.SDMolSupplier(str(out), removeHs=False)
        if m is not None
    }
    assert prefixes == {"ethanol", "skeleton", "ethane"}
    assert "frag" not in prefixes

    messages = [r.getMessage() for r in caplog.records]
    assert sum("dummy atom" in m for m in messages) == 1, messages
    assert any("frag" in m for m in messages), messages
    # Wording-agnostic: this engine reports the unparseable record in its own
    # words today, and what is pinned here is that it reports it exactly once.
    assert sum("parse" in m for m in messages) == 1, messages
