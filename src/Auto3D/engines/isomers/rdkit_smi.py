"""``RDKitIsomer``: SMILES in, enumerated stereoisomers and conformers out.

The engine the factory reaches for the registry name ``rdkit``.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from Auto3D.foundation.utils.logging_config import get_logger

logger = get_logger(__name__)
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from tqdm import tqdm

from Auto3D.domain.clash_relief import relieve_clash
from Auto3D.domain.embedding import embed_params
from Auto3D.foundation.constants import MAX_STEREOISOMERS
from Auto3D.foundation.utils.molprops import calculate_conformer_count
from Auto3D.foundation.utils.smi_io import (
    hash_enumerated_smi_IDs,
    iter_smi_records,
)
from Auto3D.foundation.utils.stereochemistry import (
    amend_configuration_w,
    remove_enantiomers,
)


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
        self.rdk_tmp = Path(job_name) / "rdk_tmp"
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
        isomers = sorted(Chem.MolToSmiles(x, isomericSmiles=True, doRandom=False) for x in isomers)
        return isomers

    def write_enumerated_smi(self) -> None:
        with open(self.enumerated_smi_path, "w+") as f:
            for name, smi in self.enumerate.items():
                for i, isomer in enumerate(smi):
                    new_name = str(name).strip() + "_" + str(i)
                    line = isomer.strip() + "\t" + new_name + "\n"
                    f.write(line)

    def write_single_isomer_smi(self) -> None:
        """Copy the input .smi through, appending the isomer index 0 to each id.

        The no-enumeration counterpart of :meth:`write_enumerated_smi`: this
        branch produces exactly one "isomer" per input -- the molecule as the
        user wrote it -- so its index is always 0. Writing it keeps the two
        branches' output shape identical, which is what makes a conformer name
        parseable at all (see :func:`Auto3D.domain.ranking.species_id`).

        Streams records instead of going through :meth:`read` so duplicate ids
        still reach ``hash_enumerated_smi_IDs`` (which renames them) exactly as
        they did when this branch handed it the input file directly; ``read()``
        returns a dict and would collapse them into one.
        """
        with open(self.enumerated_smi_path, "w+") as f:
            for _line_no, smi, name in iter_smi_records(self.input_f, on_malformed="skip"):
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
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_conformers,
                params=embed_params(n_threads=self.np, prune_rms_thresh=self.threshold),
            )
        else:
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.n_conformers,
                params=embed_params(n_threads=self.np, prune_rms_thresh=self.threshold),
            )
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
                    logger.warning(f"Skipping molecule {name!r}: failed to parse {smiles!r}")
                    continue
                isomers = self.enumerate_func(mol)
                if not isomers:
                    continue
                self.enumerate[name] = isomers
            self.write_enumerated_smi()
            logger.info("Removing enantiomers...")
            amend_configuration_w(self.enumerated_smi_path)
            remove_enantiomers(self.enumerated_smi_path, self.enumerated_smi_path_reduced)
            hash_enumerated_smi_IDs(
                self.enumerated_smi_path_reduced, self.enumerated_smi_hashed_path
            )
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
            hash_enumerated_smi_IDs(self.enumerated_smi_path, self.enumerated_smi_hashed_path)

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

    def _run_serial_embedding(self, smi_name_tuples: list[tuple[str, str]]) -> None:
        """Run serial conformer embedding."""
        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for smi, name in tqdm(smi_name_tuples):
                mol = self.embed_conformer(smi)
                if mol is None:
                    logger.warning(f"Skipping molecule {name!r}: failed to parse {smi!r}")
                    continue
                n_written = 0
                for i in range(mol.GetNumConformers()):
                    # Relieve atom clashes (MMFF, UFF fallback) and keep the
                    # conformer only if it ends up clash-free.
                    if relieve_clash(mol, i):
                        conf_id = name.strip() + f"_{i}"
                        mol.SetProp("ID", conf_id)
                        mol.SetProp("_Name", conf_id)
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

    def _run_parallel_embedding(self, smi_name_tuples: list[tuple[str, str]]) -> None:
        """Run parallel conformer embedding using ProcessPoolExecutor."""
        # Function-scope on purpose: tests patch the attribute on
        # ``Auto3D.domain.embedding`` and rely on this lookup re-reading it.
        from Auto3D.domain.embedding import embed_conformers_parallel

        logger.info(f"Using parallel embedding with {self.parallel_workers} workers...")

        with Chem.SDWriter(self.enumerated_sdf) as writer:
            for mol, conf_idx, conf_id in embed_conformers_parallel(
                smiles_names=smi_name_tuples,
                n_conformers=self.n_conformers,
                threshold=self.threshold,
                np_threads=self.np,
                n_workers=self.parallel_workers,
            ):
                mol.SetProp("ID", conf_id)
                mol.SetProp("_Name", conf_id)
                writer.write(mol, confId=conf_idx)


# ``rd_isomer``/``rd_isomer_sdf`` (2.x-era aliases of ``RDKitIsomer``/
# ``RDKitSdfIsomer``) were removed for the same reason ``tautomer_engine`` was
# removed above: ``rd_isomer_sdf`` had zero importers anywhere in the repo and
# ``rd_isomer`` had exactly one, tests/test_isomer_engine.py, which now imports
# ``RDKitIsomer`` directly. Neither is documented at any public path
# (docs/source/api.rst). Keeping an alias nothing but one test module
# exercises preserves a hazard under a spelling the rest of the codebase
# never uses -- cheaper to remove now, before 3.1.0, than later.
