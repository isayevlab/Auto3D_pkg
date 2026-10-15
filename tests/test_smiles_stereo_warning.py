"""The SMILES path must see unspecified double-bond geometry, not just atom centers.

``check_smi_format`` used to count unspecified stereo with
``CalcNumUnspecifiedAtomStereoCenters``, which reports **atom** centers only and
never double-bond geometry. The SDF path was already fixed for exactly this gap
(``RDKitSdfIsomer.count_unspecified_stereo``); the SMILES path was not, so with
``enumerate_isomers=False`` a molecule whose only open stereo element is a C=C
went through silently.

The consequence is a molecule the user did not submit, or the loss of one they
did -- measured below, by structure:

* ``OC(=O)C=CC(=O)O`` embeds as **fumaric and maleic acid together**, two
  species ~5 kcal/mol apart, under a single species id. They then compete on
  energy in ranking, so ``k=1`` returns whichever happens to be lower.
* ``CC=CC`` embeds as a mixture of cis- and trans-2-butene under one
  unlabeled name: the user gets a species they did not specify, without
  knowing which geometry (or which mixture of the two) they received.

Every assertion here is on the emitted *structures*, not on a count: a count of
2 is satisfied by two copies of one isomer. Nothing in this module loads a
neural network potential -- ETKDG embedding and RDKit stereo perception only.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

from rdkit import Chem

from Auto3D.engines.isomers.rdkit_smi import RDKitIsomer
from Auto3D.foundation.config import Auto3DOptions
from Auto3D.orchestration.pipeline.input_checks import check_smi_format

#: Canonical SMILES of the two 2-butene geometries, for readability below.
TRANS_2_BUTENE = "C/C=C/C"
CIS_2_BUTENE = "C/C=C\\C"
FUMARIC_ACID = "O=C(O)/C=C/C(=O)O"
MALEIC_ACID = "O=C(O)/C=C\\C(=O)O"


def _configuration_from_3d(mol: Chem.Mol) -> str:
    """Canonical isomeric SMILES of the configuration ``mol``'s geometry holds.

    The stereo flags on the record cannot be read directly: ``SDWriter`` marks
    a double bond of undefined geometry with the crossed-bond ("either") flag,
    which comes back as ``Chem.BondStereo.STEREOANY``, and
    ``AssignStereochemistryFrom3D`` deliberately leaves an existing STEREOANY
    flag untouched rather than deriving E/Z from the coordinates (see
    ``Auto3D.domain.stereo_check``). Clearing the flag first is what makes the
    geometry ETKDG actually built visible, and therefore what makes an
    assertion on the emitted structures possible at all.
    """
    work = Chem.Mol(mol)
    for bond in work.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.AssignStereochemistryFrom3D(work)
    return Chem.MolToSmiles(Chem.RemoveHs(work))


def _configurations_emitted(job_dir: Path, smiles: str, *, enumerate_isomers: bool) -> Counter[str]:
    """Run the real SMILES isomer/conformer engine and report what it wrote.

    Returns a ``Counter`` keyed by canonical isomeric SMILES, so both *which*
    configurations were emitted and how many conformers each got are visible.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    smi_path = job_dir / "in.smi"
    smi_path.write_text(f"{smiles}\tprobe\n")
    engine = RDKitIsomer(
        smi=str(smi_path),
        smiles_enumerated=str(job_dir / "enumerated.smi"),
        smiles_enumerated_reduced=str(job_dir / "reduced.smi"),
        smiles_hashed=str(job_dir / "hashed.smi"),
        enumerated_sdf=str(job_dir / "enumerated.sdf"),
        job_name=str(job_dir),
        max_confs=None,
        threshold=0.3,
        n_threads=1,
        enumerate_isomers=enumerate_isomers,
    )
    out_sdf = engine.run()
    emitted: Counter[str] = Counter()
    for mol in Chem.SDMolSupplier(out_sdf, removeHs=False):
        assert mol is not None, "the engine wrote a record RDKit cannot parse"
        emitted[_configuration_from_3d(mol)] += 1
    assert emitted, f"the engine emitted no conformers at all for {smiles!r}"
    return emitted


def _warnings_for(job_dir: Path, smiles: str) -> list[str]:
    """``check_smi_format`` warning texts for one SMILES, enumeration disabled."""
    job_dir.mkdir(parents=True, exist_ok=True)
    smi_path = job_dir / "probe.smi"
    smi_path.write_text(f"{smiles}\tprobe\n")
    args = Auto3DOptions(path=str(smi_path), k=1, use_gpu=False, enumerate_isomers=False)
    args.input_format = "smi"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_smi_format(args)
    return [str(record.message) for record in caught]


class TestUnspecifiedDoubleBondOnTheSmilesPath:
    """The two cases the SDF path already covers, on the SMILES path."""

    def test_fumaric_and_maleic_are_emitted_together_and_the_input_is_warned_about(self, tmp_path):
        """``OC(=O)C=CC(=O)O`` embeds as both geometric isomers at once.

        Asserting the structures, not ``len(...) == 2``: two conformers of one
        isomer would satisfy a count and would be a perfectly fine result.
        """
        emitted = _configurations_emitted(
            tmp_path / "embed", "OC(=O)C=CC(=O)O", enumerate_isomers=False
        )
        assert set(emitted) == {FUMARIC_ACID, MALEIC_ACID}, (
            "expected the documented fumaric+maleic mixture under one species "
            f"id, got {dict(emitted)}"
        )

        messages = _warnings_for(tmp_path / "warn", "OC(=O)C=CC(=O)O")
        stereo_warnings = [m for m in messages if "unspecified stereo element" in m]
        assert len(stereo_warnings) == 1, (
            "the input that emits two geometric isomers under one id was not "
            f"warned about: {messages}"
        )
        assert "OC(=O)C=CC(=O)O" in stereo_warnings[0]
        assert "1 unspecified stereo element" in stereo_warnings[0]

    def test_2_butene_unspecified_geometry_is_warned_about(self, tmp_path):
        """``CC=CC`` without enumeration gets whatever geometry ETKDG samples, and a warning.

        The conformer budget for a rotor-free 4-heavy-atom molecule such as
        this one requests 16 conformers; under that request, this package's
        embedding settings sample a mixture of both 2-butene geometries under
        one unlabeled name, so the user gets a species they did not specify
        either way the sampling happens to land. The enumerated run is the
        non-vacuity guard: it shows both geometries are reachable for this
        molecule, so whatever the embedding alone produced is a sampling
        outcome rather than an artifact of the comparison.
        """
        both = _configurations_emitted(tmp_path / "enumerated", "CC=CC", enumerate_isomers=True)
        assert set(both) == {TRANS_2_BUTENE, CIS_2_BUTENE}, (
            f"enumeration did not produce both 2-butene geometries: {dict(both)}"
        )

        emitted = _configurations_emitted(tmp_path / "embed", "CC=CC", enumerate_isomers=False)
        assert set(emitted) <= {TRANS_2_BUTENE, CIS_2_BUTENE}, (
            f"unexpected 2-butene geometry: {dict(emitted)}"
        )

        messages = _warnings_for(tmp_path / "warn", "CC=CC")
        stereo_warnings = [m for m in messages if "unspecified stereo element" in m]
        assert len(stereo_warnings) == 1, (
            f"the input whose geometry is sampled rather than specified was not warned about: "
            f"{messages}"
        )
        assert "CC=CC" in stereo_warnings[0]


class TestTheWarningStillDiscriminates:
    """A predicate that warns about everything would pass the tests above."""

    def test_a_specified_double_bond_is_not_warned_about(self, tmp_path):
        """``C/C=C/C`` names its geometry, so there is nothing to enumerate."""
        messages = _warnings_for(tmp_path / "warn", "C/C=C/C")
        assert [m for m in messages if "unspecified stereo" in m] == [], (
            f"a fully specified input was warned about: {messages}"
        )

    def test_an_unspecified_atom_center_is_still_warned_about(self, tmp_path):
        """The atom-center case the old predicate did cover must not regress."""
        messages = _warnings_for(tmp_path / "warn", "CC(O)CC")
        stereo_warnings = [m for m in messages if "unspecified stereo element" in m]
        assert len(stereo_warnings) == 1, (
            f"an unspecified tetrahedral center stopped warning: {messages}"
        )
        assert "CC(O)CC" in stereo_warnings[0]

    def test_a_molecule_with_no_stereo_at_all_is_not_warned_about(self, tmp_path):
        messages = _warnings_for(tmp_path / "warn", "CCO")
        assert [m for m in messages if "unspecified stereo" in m] == [], (
            f"ethanol was warned about: {messages}"
        )


class TestUnspecifiedStereoWarningsAggregatePerFile:
    """P-m32: a file with several under-specified records must get ONE
    UserWarning naming all of them, not one warning per record flooding the
    log -- the same aggregation `_warn_about_dummy_atom_records` already does
    for dummy-atom records."""

    def test_three_underspecified_records_produce_one_aggregated_warning(self, tmp_path):
        smi_path = tmp_path / "probe.smi"
        smi_path.write_text("CC(O)CC s1\nCC(N)CC s2\nCC(F)CC s3\n")
        args = Auto3DOptions(path=str(smi_path), k=1, use_gpu=False, enumerate_isomers=False)
        args.input_format = "smi"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_smi_format(args)

        messages = [str(record.message) for record in caught]
        stereo_warnings = [m for m in messages if "unspecified stereo element" in m]
        assert len(stereo_warnings) == 1, (
            f"expected exactly one aggregated warning, got: {messages}"
        )
        assert "3" in stereo_warnings[0]
        assert "s1" in stereo_warnings[0]
        assert "s2" in stereo_warnings[0]
