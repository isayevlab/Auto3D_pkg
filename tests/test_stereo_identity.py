"""Stereochemical identity must survive the pipeline.

Auto3D's primary value proposition is stereoisomer enumeration, so a molecule
emitted with a different configuration than it was given is a correctness
failure, not a quality one. Every test here is hermetic: these defects occur
during enumeration, before any neural network potential runs.

Findings: C1 (E/Z collapse), C2 (tautomer stereo loss), M19 (SDF path),
C9 (no post-optimization validation).
"""
from __future__ import annotations

import pytest
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

    @pytest.mark.xfail(
        strict=True,
        reason="C1: enantiomer([], []) returns True vacuously -- the loop body "
        "never executes and `indicator` stays True",
    )
    def test_two_achiral_molecules_are_not_enantiomers(self):
        """Molecules with no stereocenters cannot be an enantiomeric pair."""
        assert enantiomer([], []) is False


class TestEZIsomersSurvive:
    """E/Z configuration is invariant under reflection, so it is never enantiomeric."""

    @pytest.mark.xfail(
        strict=True,
        reason="C1: FindMolChiralCenters does not report double-bond stereo, so "
        "both descriptor lists are empty and one geometric isomer is discarded",
    )
    def test_but_2_ene_keeps_both_geometric_isomers(self):
        """CC=CC must yield both E and Z after enantiomer filtering."""
        enumerated = _enumerate("CC=CC")
        assert enumerated == ["C/C=C/C", "C/C=C\\C"], f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"

    @pytest.mark.xfail(
        strict=True,
        reason="C1: fumaric and maleic acid differ by ~5 kcal/mol and one is "
        "discarded as an 'enantiomer' of the other",
    )
    def test_fumaric_and_maleic_acid_both_survive(self):
        """The two diacids are distinct compounds, not an enantiomeric pair."""
        enumerated = _enumerate("OC(=O)C=CC(=O)O")
        assert len(enumerated) == 2, f"enumeration changed: {enumerated}"

        kept = enantiomer_helper(enumerated)
        assert len(kept) == 2, f"a geometric isomer was discarded: kept {kept}"


class TestTautomerStereoPreservation:
    """Tautomer enumeration must not silently erase a specified stereocenter."""

    @pytest.mark.xfail(
        strict=True,
        reason="C2: RDKit TautomerEnumerator defaults to SetRemoveSp3Stereo(True), "
        "so rd_taut writes stereo-stripped SMILES that are then re-enumerated "
        "as unassigned -- a 50% chance of the wrong enantiomer",
    )
    def test_specified_center_survives_tautomer_enumeration(self):
        """At least one output tautomer must retain the input's specified center."""
        from rdkit.Chem.MolStandardize import rdMolStandardize

        source = "C[C@H](C(=O)C)N"
        enumerator = rdMolStandardize.TautomerEnumerator()
        outputs = [
            Chem.MolToSmiles(t)
            for t in enumerator.Enumerate(Chem.MolFromSmiles(source))
        ]

        assert outputs, "tautomer enumeration returned nothing"
        assert any("@" in smi for smi in outputs), (
            f"every tautomer lost the specified stereocenter: {sorted(outputs)}"
        )


class TestSdfInputStereo:
    """A 2D SDF with an unspecified center must not be silently randomized."""

    @pytest.mark.xfail(
        strict=True,
        reason="M19: RDKitSdfIsomer calls only AddHs + EmbedMultipleConfs, so "
        "ETKDG returns a mixture of configurations written as conformers of one "
        "species; RDKitSdfIsomerAdapter does not even accept enumerate_isomers",
    )
    def test_unspecified_center_is_enumerated_or_refused(self, job_dir):
        """Either both configurations appear as separate species, or it is refused."""
        from rdkit.Chem import AllChem

        # Alanine drawn flat, with no stereo specified.
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(N)C(=O)O"))
        mol.SetProp("_Name", "alanine_flat")
        AllChem.EmbedMultipleConfs(mol, numConfs=12, randomSeed=42)

        codes = set()
        for conf in mol.GetConformers():
            single = Chem.Mol(mol, confId=conf.GetId())
            Chem.AssignStereochemistryFrom3D(single)
            found = Chem.FindMolChiralCenters(single, useLegacyImplementation=False)
            codes.update(code for _, code in found)

        assert len(codes) <= 1, (
            f"embedding produced a stereochemical mixture {sorted(codes)} labeled as "
            f"conformers of one species; the pipeline must enumerate or refuse instead"
        )
