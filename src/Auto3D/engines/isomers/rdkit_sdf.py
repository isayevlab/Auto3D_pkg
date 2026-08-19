"""``RDKitSdfIsomer``: stereoisomer enumeration for SDF input.

The engine the factory reaches for the registry name ``rdkit_sdf``, including via
the ``rdkit`` + ``input_format="sdf"`` auto-selection.
"""

from __future__ import annotations

from rdkit import Chem

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from tqdm import tqdm

from Auto3D.domain.embedding import embed_params
from Auto3D.foundation.constants import MAX_STEREOISOMERS
from Auto3D.foundation.utils.molprops import calculate_conformer_count
from Auto3D.foundation.utils.stereochemistry import (
    count_unspecified_stereo as _count_unspecified_stereo,
)
from Auto3D.foundation.utils.stereochemistry import (
    enantiomer_key,
)


class RDKitSdfIsomer:
    """Enumerate stereoisomers and conformers from an SDF file.

    Preserves specified stereo centers and enumerates unspecified ones, so each
    output species has one definite configuration. Enantiomeric pairs are
    reduced to one representative -- the same rule the SMILES path applies via
    ``remove_enantiomers`` -- since mirror images are exactly degenerate under
    any reflection-invariant potential. Conformers are named
    ``<name>_<isomer>_<conformer>`` uniformly, including when there is only one
    isomer, so this path's own output has one consistent shape to parse -- the
    same shape the SMILES path emits, with or without ``enumerate_isomers``.
    The isomer component is what
    :func:`Auto3D.domain.id_mapping.decode_ids` relies on to rebuild
    ``<original>_<isomer>_<conformer>`` IDs after the pipeline's numeric-ID
    encoding step; :class:`~Auto3D.domain.ranking.ConformerRanker` groups on the
    leading component only, so it is unaffected by the isomer index.

    Args:
        sdf: Path to input SDF file.
        enumerated_sdf: Path for output SDF file.
        max_confs: Maximum conformers per stereoisomer. None for dynamic.
        threshold: RMSD threshold for duplicate removal (Å).
        np: Number of CPU threads for parallelization.
        flipper: Whether to enumerate unspecified stereocenters. When False,
            a molecule with unspecified stereo is embedded as-is and its
            conformers are a mixture of configurations; a warning says so.
    """

    def __init__(
        self,
        sdf: str,
        enumerated_sdf: str,
        max_confs: int | None,
        threshold: float,
        np: int,
        flipper: bool = True,
    ) -> None:
        self.sdf = sdf
        self.enumerated_sdf = enumerated_sdf
        self.n_conformers = max_confs
        self.threshold = threshold
        self.np = np
        self.flipper = flipper

    @staticmethod
    def count_unspecified_stereo(mol: Chem.Mol) -> int:
        """Count stereo elements the input leaves unspecified.

        Thin alias for :func:`Auto3D.foundation.utils.stereochemistry.count_unspecified_stereo`,
        which owns the predicate so this path and ``check_smi_format`` cannot
        drift apart on what counts as unspecified (they did: the SMILES path
        used to count atom centers only and never saw an unspecified C=C).
        """
        return _count_unspecified_stereo(mol)

    def stereoisomers(self, mol: Chem.Mol, name: str) -> list[Chem.Mol]:
        """Return the distinct configurations to embed for one input record.

        A 3D SDF whose centers are all specified yields exactly one entry, so
        this is a no-op for that input; only unspecified centers enumerate.
        """
        if not self.flipper:
            unspecified = self.count_unspecified_stereo(mol)
            if unspecified:
                logger.warning(
                    f"{name!r} has {unspecified} unspecified stereo element(s) "
                    "and stereoisomer enumeration is disabled, so its conformers "
                    "will be a mixture of configurations. Enable isomer "
                    "enumeration to get one consistent species per configuration."
                )
            return [mol]

        opts = StereoEnumerationOptions(
            unique=True, maxIsomers=MAX_STEREOISOMERS, onlyUnassigned=True
        )
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) >= MAX_STEREOISOMERS:
            logger.warning(
                f"Stereoisomer enumeration hit the cap of {MAX_STEREOISOMERS} "
                f"for {name!r}; results may be truncated."
            )
        # Mirror images are exactly degenerate under any reflection-invariant
        # potential, so optimizing both spends half the budget for nothing and
        # leaves top_k choosing between them on numerical noise. The SMILES
        # path drops them in remove_enantiomers; this is the same rule, applied
        # to the enumerated list directly. Geometric isomers are NOT affected:
        # a reflection cannot change E/Z, so cis and trans keep distinct keys.
        deduplicated: list[Chem.Mol] = []
        seen: set[tuple[str, ...]] = set()
        for isomer in isomers:
            key = enantiomer_key(Chem.MolToSmiles(isomer))
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(isomer)
        # EnumerateStereoisomers returns an empty sequence for a molecule it
        # cannot enumerate; embedding the input unchanged beats dropping it.
        return deduplicated or [mol]

    def run(self) -> str:
        """Enumerate stereoisomers and conformers into the output SDF file.

        Returns:
            Path to the enumerated SDF file.
        """
        supp = Chem.SDMolSupplier(self.sdf, removeHs=False)
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol in tqdm(supp):
                if mol is None:
                    logger.warning(
                        "Skipping molecule: failed to parse (SDMolSupplier yielded None)."
                    )
                    continue
                name = mol.GetProp("_Name")
                for isomer_idx, isomer in enumerate(self.stereoisomers(mol, name)):
                    mol2 = Chem.AddHs(isomer)
                    if self.n_conformers is None:
                        # Compute the conformer budget on the H-complete (AddHs)
                        # mol so the SDF path agrees with the SMILES path on the
                        # RICHER with-H count. AddHs is idempotent for a mol that
                        # already carries explicit Hs (3D SDFs read with
                        # removeHs=False), so this yields the same count
                        # regardless of input format.
                        n_conformers = calculate_conformer_count(mol2)
                    else:
                        n_conformers = self.n_conformers
                    AllChem.EmbedMultipleConfs(
                        mol2,
                        numConfs=n_conformers,
                        params=embed_params(n_threads=self.np, prune_rms_thresh=self.threshold),
                    )
                    if mol2.GetNumConformers() == 0:
                        logger.warning(
                            f"Stereoisomer {isomer_idx} of {name!r} produced no "
                            "conformers; ETKDG could not embed it. This species "
                            "is absent from the output."
                        )
                        continue
                    # Three name components (species _ isomer _ conformer) match
                    # the SMILES path, whose consumers group on the first one.
                    for conf_idx, conf in enumerate(mol2.GetConformers()):
                        conf_name = f"{name}_{isomer_idx}_{conf_idx}"
                        mol2.SetProp("_Name", conf_name)
                        mol2.SetProp("ID", conf_name)
                        writer.write(mol2, confId=conf.GetId())
        return self.enumerated_sdf
