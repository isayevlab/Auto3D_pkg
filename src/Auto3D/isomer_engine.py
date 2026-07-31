#!/usr/bin/env python
"""Isomer enumeration engines for stereoisomer and conformer generation."""
from __future__ import annotations

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

from Auto3D.constants import CONFORMER_RANDOM_SEED, MAX_STEREOISOMERS
from Auto3D.utils import (
    amend_configuration_w,
    hash_enumerated_smi_IDs,
    relieve_clash,
    remove_enantiomers,
)
from Auto3D.utils.chemistry import calculate_conformer_count
from Auto3D.utils.file_ops import combine_smi, iter_smi_records

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
            hash_enumerated_smi_IDs(self.input_f,
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
                for i in range(mol.GetNumConformers()):
                    # Relieve atom clashes (MMFF, UFF fallback) and keep the
                    # conformer only if it ends up clash-free.
                    if relieve_clash(mol, i):
                        conf_id = name.strip() + f"_{i}"
                        mol.SetProp('ID', conf_id)
                        mol.SetProp('_Name', conf_id)
                        writer.write(mol, confId=i)

    def _run_parallel_embedding(
        self, smi_name_tuples: list[tuple[str, str]]
    ) -> None:
        """Run parallel conformer embedding using ProcessPoolExecutor."""
        from Auto3D.isomers.parallel_embed import embed_conformers_parallel

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
    """Enumerate conformers from an SDF file.

    Preserves specified stereo centers and enumerates unspecified ones.

    Args:
        sdf: Path to input SDF file.
        enumerated_sdf: Path for output SDF file.
        max_confs: Maximum conformers per molecule. None for dynamic.
        threshold: RMSD threshold for duplicate removal (Å).
        np: Number of CPU threads for parallelization.
    """

    def __init__(
        self,
        sdf: str,
        enumerated_sdf: str,
        max_confs: int | None,
        threshold: float,
        np: int,
    ) -> None:
        self.sdf = sdf
        self.enumerated_sdf = enumerated_sdf
        self.n_conformers = max_confs
        self.threshold = threshold
        self.np = np

    def run(self) -> str:
        """Enumerate conformers and write to output SDF file.

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
                #enumerate conformers
                mol2 = Chem.AddHs(mol)
                if self.n_conformers is None:
                    # Compute the conformer budget on the H-complete (AddHs) mol
                    # so the SDF path agrees with the SMILES path on the RICHER
                    # with-H count. AddHs is idempotent for a mol that already
                    # carries explicit Hs (3D SDFs read with removeHs=False), so
                    # this yields the same count regardless of input format.
                    n_conformers = calculate_conformer_count(mol2)
                else:
                    n_conformers = self.n_conformers
                AllChem.EmbedMultipleConfs(mol2, numConfs=n_conformers, randomSeed=CONFORMER_RANDOM_SEED, numThreads=self.np, pruneRmsThresh=self.threshold)
                #set conformer names
                name = mol.GetProp('_Name')
                for i, conf in enumerate(mol2.GetConformers()):
                    mol2.SetProp('_Name', f'{name}_{i}')
                    mol2.SetProp('ID', f'{name}_{i}')
                    writer.write(mol2, confId=i)
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

    Args:
        mode: Omega mode ('classic', 'macrocycle', 'dense', 'pose', 'rocs', 'fast_rocs').
        input_f: Path to input SMI or SDF file.
        smiles_enumerated: Path for enumerated stereoisomers.
        smiles_reduced: Path for reduced isomers (no enantiomers).
        smiles_hashed: Path for hashed SMILES IDs.
        output: Path for output SDF file.
        max_confs: Maximum conformers per molecule. None for default (1000).
        threshold: RMSD threshold for duplicate removal.
        flipper: Whether to enumerate stereoisomers.

    Returns:
        0 on success.
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
