"""Tautomer enumeration: the one engine that spans both backends.

``RDKitOrOEChemTautomerEngine`` dispatches on ``mode`` (``rdkit``/``oechem``)
inside a single class, which is why this module -- alone among the backend
modules beside it -- carries the OpenEye import guard as well as rdkit. Splitting
the class in two along that seam is a separate change with its own compatibility
question; this move does not make it.
"""

from __future__ import annotations

from rdkit import Chem

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)
from rdkit.Chem.MolStandardize import rdMolStandardize

from Auto3D.foundation.utils.smi_io import (
    combine_smi,
    iter_smi_records,
)

try:
    from openeye import oechem, oequacpac
except ImportError:
    pass


def _contradicts_reference_stereo(reference: Chem.Mol, tautomer: Chem.Mol) -> bool:
    """True if ``tautomer`` is the input's skeleton with a different configuration.

    Preserving sp3 stereo through enumeration is only safe for single-step
    flattening. Across a multi-step path -- D-erythrose reaching the 2,3-enediol,
    which flattens both of its centers -- RDKit restores a DEFINITE tag instead
    of leaving the center unspecified, and for one output that tag is the
    input's mirror image. Unfiltered, D-erythrose yields L-erythrose as a
    "tautomer": the wrong-identity defect that preserving stereo exists to
    prevent.

    The test is deliberately narrow. A tautomer whose constitution differs from
    the input is a genuinely different species, and its stereo descriptors are
    not comparable to the input's -- a keto/enol shift can relabel an untouched
    center from R to S purely by changing a neighboring branch's CIP priority,
    which is a relabeling and not an inversion. Only when the constitution is
    identical does "different configuration" mean the molecule came back wrong.

    Comparing canonical SMILES rather than per-atom descriptors also means this
    depends on no assumption about the enumerator preserving atom ordering.
    """
    if Chem.MolToSmiles(tautomer, isomericSmiles=False) != Chem.MolToSmiles(
        reference, isomericSmiles=False
    ):
        return False
    return Chem.MolToSmiles(tautomer) != Chem.MolToSmiles(reference)


class RDKitOrOEChemTautomerEngine:
    """Enumerate possible tautomers for input molecules, on either backend.

    Named for its backends, like ``RDKitIsomer``/``RDKitSdfIsomer`` beside it,
    and *not* ``TautomerEngine``: that name belongs to the role Protocol in
    :mod:`Auto3D.engines.isomers.base`, which this class structurally satisfies. The two
    shared the name until 3.0.0, which forced :mod:`Auto3D.engines.isomers.factory` to
    import this one under a local alias to keep them apart within a single file
    -- a per-file workaround for a package-wide collision, and one whose alias
    said "Omega" although tautomers come from ``oequacpac`` and Omega is the
    conformer generator. ``run()`` dispatches on ``mode``, so one class covers
    both backends.

    Args:
        mode: Tautomer engine to use: 'rdkit' or 'oechem'.
        input_f: Path to input SMI file.
        out: Path for output SMI file.
        pKaNorm: Normalize ionization state to pH ~7.4 (oechem only).
    """

    def __init__(self, mode: str, input_f: str, out: str, pKaNorm: bool) -> None:
        self.mode = mode
        self.input_f = input_f
        self.output = out
        self.pKaNorm = pKaNorm

    def oe_taut(self) -> None:
        """Enumerate tautomers using OEChem."""
        ifs = oechem.oemolistream()
        ifs.open(self.input_f)

        ofs = oechem.oemolostream()
        ofs.open(self.output)

        tautomerOptions = oequacpac.OETautomerOptions()

        for mol in ifs.GetOEGraphMols():
            for tautomer in oequacpac.OEGetReasonableTautomers(mol, tautomerOptions, self.pKaNorm):
                oechem.OEWriteMolecule(ofs, tautomer)

        # Appending input_f smiles into output
        combine_smi([self.input_f, self.output], self.output)

    def rd_taut(self) -> None:
        """Enumerate tautomers using RDKit."""
        enumerator = rdMolStandardize.TautomerEnumerator()
        # RDKit strips stereo from every atom in the tautomer core, in every
        # output tautomer, including tautomers formed at a site that cannot
        # reach a given center -- enolizing a ketone's other alpha carbon,
        # say. The stripped SMILES are then re-enumerated downstream by
        # EnumerateStereoisomers(onlyUnassigned=True) and one epimer is kept
        # arbitrarily, so a submitted (S) molecule comes back as (R) half the
        # time at identical energy. Disabling that default removal here
        # preserves what the user specified -- but is only safe for
        # single-step flattening: across a multi-step path RDKit can restore
        # a DEFINITE tag on a center it destroyed, and the tag it picks can
        # be the input's mirror image, so contradicting tautomers are
        # filtered out below via _contradicts_reference_stereo().
        enumerator.SetRemoveSp3Stereo(False)
        enumerator.SetRemoveBondStereo(False)
        smiles = []
        for _line_no, smi, idx in iter_smi_records(self.input_f, on_malformed="skip"):
            smiles.append((smi, idx))
        tautomers = []
        for smi_idx in smiles:
            smi, idx = smi_idx
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"Skipping molecule {idx!r}: failed to parse {smi!r}")
                continue
            tauts = enumerator.Enumerate(mol)
            for taut in tauts:
                if _contradicts_reference_stereo(mol, taut):
                    continue
                tautomers.append((Chem.MolToSmiles(taut), idx))
        with open(self.output, "w+") as f:
            for smi_idx in tautomers:
                smi, idx = smi_idx
                line = smi.strip() + " " + str(idx.strip()) + "\n"
                f.write(line)

    def run(self) -> None:
        """Execute tautomer enumeration."""
        if self.mode == "oechem":
            self.oe_taut()
        elif self.mode == "rdkit":
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')
