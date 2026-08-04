#!/usr/bin/env python
"""Isomer enumeration engines for stereoisomer and conformer generation."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from rdkit import Chem

from Auto3D.utils.logging_config import get_logger

logger = get_logger(__name__)
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

from Auto3D.clash_relief import relieve_clash
from Auto3D.constants import CONFORMER_RANDOM_SEED, MAX_STEREOISOMERS
from Auto3D.utils.molprops import calculate_conformer_count
from Auto3D.utils.smi_io import (
    combine_smi,
    hash_enumerated_smi_IDs,
    iter_smi_records,
)
from Auto3D.utils.stereochemistry import (
    amend_configuration_w,
    enantiomer_key,
    remove_enantiomers,
)
from Auto3D.utils.stereochemistry import (
    count_unspecified_stereo as _count_unspecified_stereo,
)

try:
    from openeye import oechem, oeomega, oequacpac
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


class TautomerEngine:
    """Enumerate possible tautomers for input molecules.

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
        with open(self.output, 'w+') as f:
            for smi_idx in tautomers:
                smi, idx = smi_idx
                line = smi.strip() + ' ' + str(idx.strip()) + '\n'
                f.write(line)

    def run(self) -> None:
        """Execute tautomer enumeration."""
        if self.mode == "oechem":
            self.oe_taut()
        elif self.mode == "rdkit":
            self.rd_taut()
        else:
            raise ValueError(f'{self.mode} must be one of "oechem" or "rdkit".')

class RDKitIsomer:
    """Enumerate stereoisomers and conformers using RDKit.

    Args:
        smi: Path to SMI file containing SMILES and IDs.
        smiles_enumerated: Output path for enumerated cis/trans isomers.
        smiles_enumerated_reduced: Output path for reduced isomers (no enantiomers).
        smiles_hashed: Output path for hashed SMILES IDs.
        enumerated_sdf: Output path for 3D conformers.
        job_name: Working directory for temporary files.
        max_confs: Maximum conformers per SMILES. None for dynamic.
        threshold: RMSD threshold for duplicate removal (Å).
        np: Number of CPU threads for conformer generation.
        flipper: Whether to enumerate R/S and cis/trans isomers.
        use_parallel_embedding: Whether to use parallel conformer embedding.
        parallel_embedding_threshold: Minimum number of molecules to trigger
            parallel embedding. Default 10.
        parallel_workers: Number of worker processes for parallel embedding.
            Default 4.
    """

    def __init__(
        self,
        smi: str,
        smiles_enumerated: str,
        smiles_enumerated_reduced: str,
        smiles_hashed: str,
        enumerated_sdf: str,
        job_name: str,
        max_confs: int | None,
        threshold: float,
        np: int,
        flipper: bool = True,
        use_parallel_embedding: bool = False,
        parallel_embedding_threshold: int = 10,
        parallel_workers: int = 4,
    ) -> None:
        self.input_f = smi
        self.n_conformers = max_confs
        self.enumerate = {}
        self.enumerated_smi_path = smiles_enumerated
        self.enumerated_smi_path_reduced = smiles_enumerated_reduced
        self.enumerated_smi_hashed_path = smiles_hashed
        self.enumerated_sdf = enumerated_sdf
        self.rdk_tmp = Path(job_name) / 'rdk_tmp'
        self.rdk_tmp.mkdir()
        self.threshold = threshold
        self.np = np
        self.flipper = flipper
        self.use_parallel_embedding = use_parallel_embedding
        self.parallel_embedding_threshold = parallel_embedding_threshold
        self.parallel_workers = parallel_workers

    @staticmethod
    def read(input_f: str) -> dict[str, str]:
        """Read SMILES file and return name->SMILES mapping.

        Lenient line handling (matches encode_ids semantics): whitespace-only
        lines are skipped, and a line is only accepted if it has at least two
        whitespace-separated tokens (SMILES + ID). Extra tokens are ignored.
        Malformed lines are warned about and skipped rather than aborting.
        """
        outputs = {}
        for _line_no, smiles, name in iter_smi_records(input_f, on_malformed="skip"):
            outputs[name] = smiles
        return outputs

    @staticmethod
    def enumerate_func(mol: Chem.Mol) -> list[str]:
        """Enumerate R/S and cis/trans isomers for a molecule.

        Args:
            mol: RDKit molecule object, or None for an unparseable SMILES.

        Returns:
            Sorted list of isomer SMILES strings (empty if ``mol`` is None).
        """
        if mol is None:
            logger.warning("Skipping molecule: failed to parse (MolFromSmiles returned None).")
            return []
        # Set an explicit, high maxIsomers so molecules with many unspecified
        # stereocenters are not silently truncated at RDKit's default of 1024.
        opts = StereoEnumerationOptions(unique=True, maxIsomers=MAX_STEREOISOMERS)
        isomers = tuple(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) >= MAX_STEREOISOMERS:
            logger.warning(
                f"Stereoisomer enumeration hit the cap of {MAX_STEREOISOMERS} "
                f"for {Chem.MolToSmiles(mol)!r}; results may be truncated."
            )
        isomers = sorted(
            Chem.MolToSmiles(x, isomericSmiles=True, doRandom=False) for x in isomers
        )
        return isomers

    def write_enumerated_smi(self) -> None:
        with open(self.enumerated_smi_path, 'w+') as f:
            for name, smi in self.enumerate.items():
                for i, isomer in enumerate(smi):
                    new_name = str(name).strip() + '_' + str(i)
                    line = isomer.strip() + '\t' + new_name + '\n'
                    f.write(line)

    def write_single_isomer_smi(self) -> None:
        """Copy the input .smi through, appending the isomer index 0 to each id.

        The no-enumeration counterpart of :meth:`write_enumerated_smi`: this
        branch produces exactly one "isomer" per input -- the molecule as the
        user wrote it -- so its index is always 0. Writing it keeps the two
        branches' output shape identical, which is what makes a conformer name
        parseable at all (see :func:`Auto3D.ranking.species_id`).

        Streams records instead of going through :meth:`read` so duplicate ids
        still reach ``hash_enumerated_smi_IDs`` (which renames them) exactly as
        they did when this branch handed it the input file directly; ``read()``
        returns a dict and would collapse them into one.
        """
        with open(self.enumerated_smi_path, 'w+') as f:
            for _line_no, smi, name in iter_smi_records(
                self.input_f, on_malformed="skip"
            ):
                f.write(f"{smi.strip()}\t{str(name).strip()}_0\n")

    def embed_conformer(self, smi: str) -> Chem.Mol | None:
        """Embed multiple 3D conformers for a SMILES string.

        Returns None if the SMILES cannot be parsed, mirroring the parallel
        worker (_embed_single) so a single unparseable SMILES is skipped rather
        than crashing the whole serial embedding loop on AddHs(None).
        """
        mol_noh = Chem.MolFromSmiles(smi)
        if mol_noh is None:
            return None
        mol = Chem.AddHs(mol_noh)
        if self.n_conformers is None:
            # Compute the conformer budget on the H-complete (AddHs) mol so the
            # SMILES and SDF paths agree, and on the RICHER side: RDKit's
            # CalcNumRotatableBonds only counts O-H / N-H torsions when hydrogens
            # are explicit, so the with-H count samples hydroxyl/amine rotors
            # that the no-H count drops (e.g. glycerol 238 vs 52 conformers).
            n_conformers = calculate_conformer_count(mol)
            AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                                    randomSeed=CONFORMER_RANDOM_SEED, numThreads=self.np,
                                    pruneRmsThresh=self.threshold)
        else:
            AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers,
                                    randomSeed=CONFORMER_RANDOM_SEED, numThreads=self.np,
                                    pruneRmsThresh=self.threshold)
        return mol

    def run(self) -> str:
        """Enumerate 3D structures and write to output SDF file.

        Returns:
            Path to the enumerated SDF file.
        """
        if self.flipper:
            logger.info("Enumerating cis/tran isomers for unspecified double bonds...")
            logger.info("Enumerating R/S isomers for unspecified atomic centers...")
            smiles_og = self.read(self.input_f)
            for name, smiles in smiles_og.items():
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(
                        f"Skipping molecule {name!r}: failed to parse {smiles!r}"
                    )
                    continue
                isomers = self.enumerate_func(mol)
                if not isomers:
                    continue
                self.enumerate[name] = isomers
            self.write_enumerated_smi()
            logger.info("Removing enantiomers...")
            amend_configuration_w(self.enumerated_smi_path)
            remove_enantiomers(self.enumerated_smi_path, self.enumerated_smi_path_reduced)
            hash_enumerated_smi_IDs(self.enumerated_smi_path_reduced,
                                    self.enumerated_smi_hashed_path)
        else:
            # No stereoisomer enumeration -- but the conformer names must still
            # carry BOTH trailing components, or they cannot be parsed back.
            # Emitting only "<species>_<conformer>" here made "KEY_2_0"
            # ambiguous: species "KEY_2" conformer 0 (this branch) or species
            # "KEY" isomer 2 conformer 0 (the branch above)?
            # `smiles2smi` mints exactly ids like "KEY_2" -- for the SECOND of
            # two DIFFERENT input molecules that share a standard InChIKey --
            # so that neither is dropped. `ranking.species_id` then grouped
            # both under "KEY", and `k=1` returned a single conformer for the
            # pair, possibly the other molecule's geometry under this
            # molecule's name. Writing the isomer index makes the parse
            # unambiguous instead of asking the parser to guess which branch
            # produced the name.
            self.write_single_isomer_smi()
            hash_enumerated_smi_IDs(self.enumerated_smi_path,
                                    self.enumerated_smi_hashed_path)

        logger.info("Enumerating conformers/rotamers, removing duplicates...")
        smiles2 = self.read(self.enumerated_smi_hashed_path)

        smi_name_tuples = [(smi, name) for name, smi in smiles2.items()]

        # Decide whether to use parallel embedding
        use_parallel = (
            self.use_parallel_embedding
            and len(smi_name_tuples) >= self.parallel_embedding_threshold
        )

        if use_parallel:
            self._run_parallel_embedding(smi_name_tuples)
        else:
            self._run_serial_embedding(smi_name_tuples)

        return self.enumerated_sdf

    def _run_serial_embedding(
        self, smi_name_tuples: list[tuple[str, str]]
    ) -> None:
        """Run serial conformer embedding."""
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for smi, name in tqdm(smi_name_tuples):
                mol = self.embed_conformer(smi)
                if mol is None:
                    logger.warning(
                        f"Skipping molecule {name!r}: failed to parse {smi!r}"
                    )
                    continue
                n_written = 0
                for i in range(mol.GetNumConformers()):
                    # Relieve atom clashes (MMFF, UFF fallback) and keep the
                    # conformer only if it ends up clash-free.
                    if relieve_clash(mol, i):
                        conf_id = name.strip() + f"_{i}"
                        mol.SetProp('ID', conf_id)
                        mol.SetProp('_Name', conf_id)
                        writer.write(mol, confId=i)
                        n_written += 1
                if n_written == 0:
                    # Every embedded conformer was rejected by clash relief
                    # (or none embedded at all): the species is silently
                    # absent from the output and never reaches ranking, so
                    # not even "No structure converged" would appear for it.
                    # Name it here, once, mirroring the SDF path's equivalent
                    # warning for a stereoisomer ETKDG could not embed.
                    logger.warning(
                        f"{name!r} produced no conformers after clash relief; "
                        "this species is absent from the output."
                    )

    def _run_parallel_embedding(
        self, smi_name_tuples: list[tuple[str, str]]
    ) -> None:
        """Run parallel conformer embedding using ProcessPoolExecutor."""
        # Function-scope on purpose: tests patch the attribute on
        # ``Auto3D.embedding`` and rely on this lookup re-reading it.
        from Auto3D.embedding import embed_conformers_parallel

        logger.info(f"Using parallel embedding with {self.parallel_workers} workers...")

        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol, conf_idx, conf_id in embed_conformers_parallel(
                smiles_names=smi_name_tuples,
                n_conformers=self.n_conformers,
                threshold=self.threshold,
                np_threads=self.np,
                n_workers=self.parallel_workers,
            ):
                mol.SetProp('ID', conf_id)
                mol.SetProp('_Name', conf_id)
                writer.write(mol, confId=conf_idx)


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
    :func:`Auto3D.id_mapping.decode_ids` relies on to rebuild
    ``<original>_<isomer>_<conformer>`` IDs after the pipeline's numeric-ID
    encoding step; :class:`~Auto3D.ranking.ConformerRanker` groups on the
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

        Thin alias for :func:`Auto3D.utils.stereochemistry.count_unspecified_stereo`,
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
                name = mol.GetProp('_Name')
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
                        randomSeed=CONFORMER_RANDOM_SEED,
                        numThreads=self.np,
                        pruneRmsThresh=self.threshold,
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
                        conf_name = f'{name}_{isomer_idx}_{conf_idx}'
                        mol2.SetProp('_Name', conf_name)
                        mol2.SetProp('ID', conf_name)
                        writer.write(mol2, confId=conf.GetId())
        return self.enumerated_sdf


def oe_flipper(input_f: str, out: str) -> None:
    """Enumerate stereoisomers using OpenEye Flipper."""
    ifs = oechem.oemolistream()
    ifs.open(input_f)
    ofs = oechem.oemolostream()
    ofs.open(out)

    flipperOpts = oeomega.OEFlipperOptions()
    flipperOpts.SetWarts(True)
    flipperOpts.SetMaxCenters(12)
    flipperOpts.SetEnumNitrogen(True)
    flipperOpts.SetEnumBridgehead(True)
    flipperOpts.SetEnumEZ(False)
    flipperOpts.SetEnumRS(False)
    for mol in ifs.GetOEMols():
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), flipperOpts):
            enantiomer = oechem.OEMol(enantiomer)
            oechem.OEWriteMolecule(ofs, enantiomer)

def oe_isomer(
    mode: str,
    input_f: str,
    smiles_enumerated: str,
    smiles_reduced: str,
    smiles_hashed: str,
    output: str,
    max_confs: int | None,
    threshold: float,
    flipper: bool = True,
) -> int:
    """Generate R/S, cis/trans isomers and conformers using OpenEye Omega.

    The OpenEye toolkit's application-options machinery writes ``oeomega_*``
    and ``flipper_*`` logfiles into the **process working directory**, which
    for an ordinary ``cd ~/project && auto3d run mols.smi`` is the user's own
    directory. ``job_layout.housekeeping`` used to sweep those names out
    of the cwd and into the run's ``verbose`` folder, which is tarred and then
    deleted -- so a user file named ``oeomega_settings.txt`` in the cwd was
    destroyed by an ordinary run. The fix is on this side rather than on the
    sweep's: the OpenEye section below runs with the working directory set to
    the directory this call already owns (``output``'s parent, i.e. the
    per-chunk directory Auto3D created for this job), so the logfiles land
    there and ``housekeeping`` collects them with the rest of the chunk's
    metadata.

    Every path argument is made absolute first, because a relative one would
    otherwise be resolved against that new working directory. The change is
    invisible to callers: the pipeline already passes absolute paths derived
    from ``create_chunk_meta_names``.

    Args:
        mode: Omega mode ('classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs').
        input_f: Path to input SMI or SDF file.
        smiles_enumerated: Path for enumerated stereoisomers.
        smiles_reduced: Path for reduced isomers (no enantiomers).
        smiles_hashed: Path for hashed SMILES IDs.
        output: Path for output SDF file. Its parent directory is where the
            OpenEye logfiles are written, and must exist.
        max_confs: Maximum conformers per molecule. None for default (1000).
        threshold: RMSD threshold for duplicate removal.
        flipper: Whether to enumerate stereoisomers.

    Returns:
        0 on success.
    """
    input_f = os.path.abspath(input_f)
    smiles_enumerated = os.path.abspath(smiles_enumerated)
    smiles_reduced = os.path.abspath(smiles_reduced)
    smiles_hashed = os.path.abspath(smiles_hashed)
    output = os.path.abspath(output)

    with contextlib.chdir(os.path.dirname(output)):
        return _oe_isomer_in_owned_cwd(
            mode,
            input_f,
            smiles_enumerated,
            smiles_reduced,
            smiles_hashed,
            output,
            max_confs,
            threshold,
            flipper,
        )


def _oe_isomer_in_owned_cwd(
    mode: str,
    input_f: str,
    smiles_enumerated: str,
    smiles_reduced: str,
    smiles_hashed: str,
    output: str,
    max_confs: int | None,
    threshold: float,
    flipper: bool = True,
) -> int:
    """Body of :func:`oe_isomer`, run with the cwd set to a directory we own.

    Split out only so the ``chdir`` wrapper above stays readable; every path
    reaching this function is already absolute. Do not call it directly --
    doing so puts the OpenEye logfiles back in the caller's cwd.
    """
    input_format = Path(input_f).suffix[1:].strip()
    if max_confs is None:
        max_confs = 1000

    match mode:
        case "classic":
            omegaOpts = oeomega.OEOmegaOptions()
        case "dense":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        case "pose":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Pose)
        case "rocs":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_ROCS)
        case "fast_rocs":
            omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_FastROCS)
        case "macrocycle":
            omegaOpts = oeomega.OEMacrocycleOmegaOptions()
        case _:
            raise ValueError(f"mode has to be 'classic' or 'macrocycle', but received {mode}.")
    omegaOpts.SetParameterVisibility(oechem.OEParamVisibility_Hidden) 
    omegaOpts.SetParameterVisibility("-rms", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-ewindow", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-maxconfs", oechem.OEParamVisibility_Simple)

    if mode == 'macrocycle':
        omegaOpts.SetIterCycleSize(1000)
        omegaOpts.SetMaxIter(2000)   
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)
    else:
        omegaOpts.SetFixRMS(threshold)  #macrocycle mode does not have the attribute 'SetFixRMS'
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetWarts(True)
        omegaOpts.SetMaxConfs(max_confs)
        omegaOpts.SetEnergyWindow(999)   
        omegaOpts.SetRMSRange("0.8, 1.0, 1.2, 1.4")             
    # dense, pose, rocs, fast_rocs mdoes use the default parameters from OEOMEGA:
    # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenConstants/OEOmegaSampling.html 
    opts = oechem.OESimpleAppOptions(omegaOpts, "Omega", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol3D)

    omegaOpts.UpdateValues(opts)
    if mode == "macrocycle":
        omega = oeomega.OEMacrocycleOmega(omegaOpts)
    else:
        omega = oeomega.OEOmega(omegaOpts)
    if input_format == "smi":
        if flipper:
            logger.info("Enumerating stereoisomers.")
            oe_flipper(input_f, smiles_enumerated)
            amend_configuration_w(smiles_enumerated)
            remove_enantiomers(smiles_enumerated, smiles_reduced)
            ifs = oechem.oemolistream()
            ifs.open(smiles_reduced)
        else:
            ifs = oechem.oemolistream()
            ifs.open(input_f)
    elif input_format == "sdf":
            ifs = oechem.oemolistream()
            ifs.open(input_f)        
    ofs = oechem.oemolostream()
    ofs.open(output)

    logger.info("Enumerating conformers.")
    for mol in tqdm(ifs.GetOEMols()):
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    return 0


# Backward compatibility aliases
tautomer_engine = TautomerEngine
rd_isomer = RDKitIsomer
rd_isomer_sdf = RDKitSdfIsomer
