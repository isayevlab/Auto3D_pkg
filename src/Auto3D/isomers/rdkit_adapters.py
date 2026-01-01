"""Adapters for RDKit isomer engines."""
from __future__ import annotations

from Auto3D.isomers.base import BaseIsomerEngine


class RDKitIsomerAdapter(BaseIsomerEngine):
    """Adapter wrapping the RDKitIsomer class.

    This adapter provides a consistent interface for the RDKit SMI-based
    isomer generation, following the same pattern as OmegaIsomerAdapter.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        smiles_enumerated: str,
        smiles_reduced: str,
        smiles_hashed: str,
        job_dir: str,
        max_confs: int | None = None,
        threshold: float = 0.3,
        n_jobs: int = 4,
        enumerate_isomers: bool = True,
        use_parallel_embedding: bool = False,
        parallel_embedding_threshold: int = 10,
        parallel_workers: int = 4,
    ) -> None:
        """Initialize the RDKit SMI isomer adapter.

        Args:
            input_path: Path to input SMI file.
            output_path: Path for output SDF file.
            smiles_enumerated: Path for enumerated stereoisomers.
            smiles_reduced: Path for reduced isomers (no enantiomers).
            smiles_hashed: Path for hashed SMILES IDs.
            job_dir: Working directory for temporary files.
            max_confs: Maximum conformers per molecule.
            threshold: RMSD threshold for duplicate removal.
            n_jobs: Number of CPU threads for conformer generation.
            enumerate_isomers: Whether to enumerate R/S and cis/trans isomers.
            use_parallel_embedding: Whether to use parallel conformer embedding.
            parallel_embedding_threshold: Minimum molecules for parallel embedding.
            parallel_workers: Number of worker processes for parallel embedding.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            max_confs=max_confs,
            threshold=threshold,
            n_jobs=n_jobs,
        )
        self.smiles_enumerated = smiles_enumerated
        self.smiles_reduced = smiles_reduced
        self.smiles_hashed = smiles_hashed
        self.job_dir = job_dir
        self.enumerate_isomers = enumerate_isomers
        self.use_parallel_embedding = use_parallel_embedding
        self.parallel_embedding_threshold = parallel_embedding_threshold
        self.parallel_workers = parallel_workers

    def run(self) -> str:
        """Execute RDKit SMI-based isomer generation.

        Returns:
            Path to the output SDF file.
        """
        from Auto3D.isomer_engine import RDKitIsomer

        engine = RDKitIsomer(
            smi=self.input_path,
            smiles_enumerated=self.smiles_enumerated,
            smiles_enumerated_reduced=self.smiles_reduced,
            smiles_hashed=self.smiles_hashed,
            enumerated_sdf=self.output_path,
            job_name=self.job_dir,
            max_confs=self.max_confs,
            threshold=self.threshold,
            np=self.n_jobs,
            flipper=self.enumerate_isomers,
            use_parallel_embedding=self.use_parallel_embedding,
            parallel_embedding_threshold=self.parallel_embedding_threshold,
            parallel_workers=self.parallel_workers,
        )
        return engine.run()


class RDKitSdfIsomerAdapter(BaseIsomerEngine):
    """Adapter wrapping the RDKitSdfIsomer class.

    This adapter provides a consistent interface for the RDKit SDF-based
    conformer generation, following the same pattern as OmegaIsomerAdapter.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_confs: int | None = None,
        threshold: float = 0.3,
        n_jobs: int = 4,
    ) -> None:
        """Initialize the RDKit SDF isomer adapter.

        Args:
            input_path: Path to input SDF file.
            output_path: Path for output SDF file.
            max_confs: Maximum conformers per molecule.
            threshold: RMSD threshold for duplicate removal.
            n_jobs: Number of CPU threads for conformer generation.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            max_confs=max_confs,
            threshold=threshold,
            n_jobs=n_jobs,
        )

    def run(self) -> str:
        """Execute RDKit SDF-based conformer generation.

        Returns:
            Path to the output SDF file.
        """
        from Auto3D.isomer_engine import RDKitSdfIsomer

        engine = RDKitSdfIsomer(
            sdf=self.input_path,
            enumerated_sdf=self.output_path,
            max_confs=self.max_confs,
            threshold=self.threshold,
            np=self.n_jobs,
        )
        return engine.run()
