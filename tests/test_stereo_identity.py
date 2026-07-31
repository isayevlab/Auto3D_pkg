"""Stereochemical identity must survive the pipeline.

Auto3D's primary value proposition is stereoisomer enumeration, so a molecule
emitted with a different configuration than it was given is a correctness
failure, not a quality one. Every test here is hermetic: these defects occur
during enumeration, before any neural network potential runs.

Findings: C1 (E/Z collapse), C2 (tautomer stereo loss), M19 (SDF path).
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from Auto3D.utils.stereochemistry import enantiomer, enantiomer_helper


def _enumerate(smiles: str) -> list[str]:
    """Enumerate unassigned stereocenters the way the pipeline does."""
    opts = StereoEnumerationOptions(unique=True, maxIsomers=64, onlyUnassigned=True)
    mol = Chem.MolFromSmiles(smiles)
    return sorted(Chem.MolToSmiles(m) for m in EnumerateStereoisomers(mol, options=opts))


class TestEnantiomerPredicate:
    """The enantiomer predicate must not treat 'no stereocenters' as 'enantiomers'."""

    def test_two_achiral_molecules_are_not_enantiomers(self):
        """Molecules with no stereocenters cannot be an enantiomeric pair."""
        assert enantiomer([], []) is False


class TestEZIsomersSurvive:
    """E/Z configuration is invariant under reflection, so it is never enantiomeric."""

    def test_but_2_ene_keeps_both_geometric_isomers(self):
        """CC=CC must yield both E and Z after enantiomer filtering."""
        enumerated = _enumerate("CC=CC")
        assert enumerated == ["C/C=C/C", "C/C=C\\C"], f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"

    def test_fumaric_and_maleic_acid_both_survive(self):
        """The two diacids are distinct compounds, not an enantiomeric pair."""
        enumerated = _enumerate("OC(=O)C=CC(=O)O")
        assert len(enumerated) == 2, f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"


class TestTautomerStereoPreservation:
    """Tautomer enumeration must not silently erase a specified stereocenter."""

    def test_specified_center_survives_tautomer_enumeration(self, job_dir):
        """At least one output tautomer must retain the input's specified center.

        This drives Auto3D's real ``rdkit`` tautomer engine -- the same
        ``TautomerEngine.rd_taut()`` the pipeline dispatches to, reached via
        ``Auto3D.isomers.factory.create_tautomer_engine`` -- rather than a
        bare RDKit ``TautomerEnumerator``, so the defect is attributed to
        Auto3D's tautomer path and not to RDKit in isolation.

        This center is itself alpha to the ketone, so a tautomer genuinely
        produced by enolizing through the stereocenter's own alpha-hydrogen
        could legitimately racemize it -- that would be real chemistry, not a
        bug. The defect under test is that RDKit's ``SetRemoveSp3Stereo(True)``
        strips the center indiscriminately from EVERY output tautomer,
        including ones produced by enolizing the ketone's other,
        non-stereogenic alpha carbon, which cannot touch this center at all.
        That is unconditional information loss, not equilibrium modeling, and
        the eventual fix must not over-correct to "preserve stereo across all
        tautomers unconditionally."
        """
        from Auto3D.isomers.factory import create_tautomer_engine

        in_smi = job_dir / "taut_stereo.smi"
        in_smi.write_text("C[C@H](C(=O)C)N taut_test\n")
        out_smi = job_dir / "taut_stereo_out.smi"

        create_tautomer_engine(
            "rdkit", str(in_smi), str(out_smi), pka_norm=False
        ).run()

        outputs = out_smi.read_text().splitlines()
        assert outputs, "tautomer enumeration returned nothing"
        assert any("@" in line for line in outputs), (
            f"every tautomer lost the specified stereocenter: {sorted(outputs)}"
        )


class TestSdfInputStereo:
    """A 2D SDF with an unspecified center must not be silently randomized."""

    def test_unspecified_center_is_enumerated_or_refused(self, job_dir):
        """Drive Auto3D's real SDF isomer engine on a flat, unspecified center.

        This writes a genuine flat (2D, no wedge bonds, no parity flags) SDF
        record for alanine to disk and feeds it through the production
        ``rdkit_sdf`` engine -- the same ``RDKitSdfIsomerAdapter`` /
        ``RDKitSdfIsomer.run()`` the pipeline dispatches to for SDF input --
        via ``Auto3D.isomers.factory.create_isomer_engine``. It then inspects
        the SDF file Auto3D actually writes, grouped by species name (the
        conformer-index suffix stripped). Either the two configurations must
        come out as distinct, internally consistent species, or ambiguous
        input must be explicitly refused (a ``ValueError``). Instead, both
        configurations are written as numbered conformers under one species
        name -- the defect this test targets.
        """
        from rdkit.Chem import AllChem

        from Auto3D.isomers.factory import create_isomer_engine

        # Alanine drawn flat (2D), with no stereo specified anywhere: no
        # wedge/hash bonds, no parity flags in the mol block.
        mol = Chem.MolFromSmiles("CC(N)C(=O)O")
        mol.SetProp("_Name", "alanine_flat")
        AllChem.Compute2DCoords(mol)

        input_sdf = job_dir / "alanine_flat.sdf"
        with Chem.SDWriter(str(input_sdf)) as writer:
            writer.write(mol)

        output_sdf = job_dir / "alanine_enumerated.sdf"
        engine = create_isomer_engine(
            "rdkit_sdf",
            input_path=str(input_sdf),
            output_path=str(output_sdf),
            max_confs=12,
            threshold=0.3,
            n_jobs=1,
        )

        try:
            engine.run()
        except ValueError:
            # Explicit refusal of ambiguous stereochemistry is an acceptable
            # resolution; there is nothing further to check.
            return

        per_species: dict[str, set[str]] = {}
        for out_mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False):
            if out_mol is None:
                continue
            name = out_mol.GetProp("_Name")
            species = name.rsplit("_", 1)[0]
            Chem.AssignStereochemistryFrom3D(out_mol)
            found = Chem.FindMolChiralCenters(out_mol, useLegacyImplementation=False)
            per_species.setdefault(species, set()).update(code for _, code in found)

        mixed = {
            name: sorted(codes) for name, codes in per_species.items() if len(codes) > 1
        }
        assert not mixed, (
            f"RDKitSdfIsomer wrote a stereochemical mixture under a single species "
            f"name: {mixed}; the pipeline must enumerate distinct species or refuse "
            f"ambiguous input instead of silently mixing configurations as "
            f"conformers of one species"
        )
