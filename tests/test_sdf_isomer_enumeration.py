"""The SDF input path enumerates stereoisomers like the SMILES path.

Every test drives the production ``rdkit_sdf`` engine through
``IsomerEngineFactory.create`` -- the same path ``auto3D.py`` and
``workflow_workers.py`` use -- not RDKit in isolation, and inspects the SDF file
Auto3D actually writes.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from Auto3D.isomers import IsomerEngineFactory


def _write_sdf(path, smiles: str, name: str, three_d: bool) -> None:
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    if three_d:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=3)
    else:
        AllChem.Compute2DCoords(mol)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def _run_engine(job_dir, smiles, name, three_d, **kwargs):
    """Run the engine and return {species_name: {cip codes seen}}."""
    input_sdf = job_dir / f"{name}_in.sdf"
    output_sdf = job_dir / f"{name}_out.sdf"
    _write_sdf(input_sdf, smiles, name, three_d)

    engine = IsomerEngineFactory.create(
        "rdkit_sdf",
        input_path=str(input_sdf),
        output_path=str(output_sdf),
        max_confs=6,
        threshold=0.3,
        n_jobs=1,
        **kwargs,
    )
    engine.run()

    per_species: dict[str, set[str]] = {}
    for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
        if mol is None:
            continue
        species = mol.GetProp("_Name").rsplit("_", 1)[0]
        Chem.AssignStereochemistryFrom3D(mol)
        codes = {code for _, code in
                 Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)}
        per_species.setdefault(species, set()).update(codes)
    return per_species


class TestUnspecifiedCentersEnumerate:
    def test_flat_sdf_yields_two_consistent_species(self, job_dir):
        """A flat threonine gives two diastereomeric species, each consistent.

        Threonine has two independent stereocenters, so its four raw
        enumerated stereoisomers form two enantiomeric pairs; the enantiomer
        filter reduces each pair to one representative, leaving two
        diastereomers (unlike alanine's single stereocenter, whose R/S pair
        is now collapsed to a single representative -- see
        ``test_enantiomers_are_removed_but_geometric_isomers_are_not``).
        "Consistent" here means every conformer of a given species reports
        the identical per-atom CIP assignment, not that every code letter is
        identical -- a two-center molecule legitimately has one R and one S
        center at once, which is not a mixture.
        """
        input_sdf = job_dir / "threonine_in.sdf"
        output_sdf = job_dir / "threonine_out.sdf"
        _write_sdf(input_sdf, "CC(O)C(N)C(=O)O", "threonine", three_d=False)
        IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=6,
            threshold=0.3,
            n_jobs=1,
        ).run()

        per_species: dict[str, set[tuple[tuple[int, str], ...]]] = {}
        for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
            if mol is None:
                continue
            species = mol.GetProp("_Name").rsplit("_", 1)[0]
            Chem.AssignStereochemistryFrom3D(mol)
            assignment = tuple(
                sorted(Chem.FindMolChiralCenters(mol, useLegacyImplementation=False))
            )
            per_species.setdefault(species, set()).add(assignment)

        assert len(per_species) == 2, f"expected two diastereomers, got {per_species}"
        for name, assignments in per_species.items():
            assert len(assignments) == 1, (
                f"{name} mixes configurations across conformers: {assignments}"
            )

    def test_species_names_have_three_components(self, job_dir):
        """Names are <species>_<isomer>_<conformer>, as the SMILES path emits.

        ConformerRanker groups on the first underscore-delimited component, so
        a two-component name would put both configurations back in one group.
        Threonine survives enantiomer dedup with two isomers (a diastereomeric
        pair); alanine's single-stereocenter R/S pair now collapses to one
        representative, so it no longer exercises the isomer-index component.
        """
        input_sdf = job_dir / "threonine3_in.sdf"
        output_sdf = job_dir / "threonine3_out.sdf"
        _write_sdf(input_sdf, "CC(O)C(N)C(=O)O", "threonine", three_d=False)
        IsomerEngineFactory.create(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=4,
            threshold=0.3,
            n_jobs=1,
        ).run()

        names = [m.GetProp("_Name")
                 for m in Chem.SDMolSupplier(str(output_sdf), removeHs=False)
                 if m is not None]
        assert names, "the engine wrote nothing"
        for name in names:
            assert name.count("_") == 2, f"unexpected name shape: {name}"
            assert name.split("_")[0] == "threonine", name
        assert {name.split("_")[1] for name in names} == {"0", "1"}, names

    def test_enantiomers_are_removed_but_geometric_isomers_are_not(self, job_dir):
        """One species per enantiomeric pair, both species for cis/trans.

        Mirror images are degenerate under a reflection-invariant potential, so
        the SMILES path drops one of each pair and this path must match. A
        reflection cannot change E/Z, so geometric isomers are not a pair and
        both must survive -- the same distinction the enantiomer filter draws.
        """
        chiral = _run_engine(job_dir, "CC(N)C(=O)O", "alanine_ec", three_d=False)
        assert len(chiral) == 1, f"an enantiomer was kept: {sorted(chiral)}"

        geometric = _run_engine(job_dir, "CC=CC", "butene_ec", three_d=False)
        assert len(geometric) == 2, f"a geometric isomer was dropped: {sorted(geometric)}"


class TestSpecifiedStereoIsNotDisturbed:
    def test_3d_sdf_with_a_specified_center_stays_one_species(self, job_dir):
        """3D SDF input was already safe and must remain a single species."""
        per_species = _run_engine(
            job_dir, "C[C@H](N)C(=O)O", "lalanine", three_d=True
        )
        assert len(per_species) == 1, f"a specified center was enumerated: {per_species}"
        codes = next(iter(per_species.values()))
        assert codes == {"S"}, f"the specified configuration changed: {codes}"


class TestEnumerationDisabled:
    def test_disabled_enumeration_warns_about_the_mixture(self, job_dir, caplog):
        """With enumeration off the user is told the output is a mixture."""
        import logging

        with caplog.at_level(logging.WARNING, logger="Auto3D.isomer_engine"):
            per_species = _run_engine(
                job_dir, "CC(N)C(=O)O", "alanine_off", three_d=False,
                enumerate_isomers=False,
            )
        assert len(per_species) == 1, f"enumeration ran while disabled: {per_species}"
        assert any("unspecified stereo" in record.message for record in caplog.records), (
            f"no warning about the mixture: {[r.message for r in caplog.records]}"
        )

    def test_disabled_enumeration_warns_about_an_unspecified_double_bond(
        self, job_dir, caplog
    ):
        """A flat, undrawn C=C must trigger the mixture warning too.

        RDKit reports a double bond with no drawn geometry as
        ``Chem.StereoSpecified.Unknown``, not ``Unspecified``. A count that
        only checks ``Unspecified`` misses every unspecified C=C -- exactly
        the case this warning exists for: a flat 2D fumaric/maleic-acid SDF
        (``OC(=O)C=CC(=O)O``) would otherwise be written as one species
        silently mixing two geometries ~5 kcal/mol apart.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="Auto3D.isomer_engine"):
            per_species = _run_engine(
                job_dir, "OC(=O)C=CC(=O)O", "fumaric_off", three_d=False,
                enumerate_isomers=False,
            )
        assert len(per_species) == 1, f"enumeration ran while disabled: {per_species}"
        assert any(
            "1 unspecified stereo element" in record.message
            and "fumaric_off" in record.message
            for record in caplog.records
        ), (
            "no warning naming the unspecified stereo element: "
            f"{[r.message for r in caplog.records]}"
        )
