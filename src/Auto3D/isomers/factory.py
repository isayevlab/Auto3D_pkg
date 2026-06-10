"""Factory classes and functions for creating isomer engines."""
from __future__ import annotations

from typing import TYPE_CHECKING

from Auto3D.isomers.base import BaseIsomerEngine, IsomerEngine, TautomerEngine

if TYPE_CHECKING:
    pass


class IsomerEngineFactory:
    """Factory class for creating isomer engine adapters.

    This class follows the same pattern as ModelFactory, providing a unified
    interface for creating different isomer engine implementations.

    Example:
        >>> engine = IsomerEngineFactory.create(
        ...     "rdkit",
        ...     input_path="input.smi",
        ...     output_path="output.sdf",
        ...     smiles_enumerated="enum.smi",
        ...     job_dir="/path/to/job",
        ... )
        >>> output = engine.run()
    """

    # Registry of available adapters
    _adapters: dict[str, type[BaseIsomerEngine]] = {}

    @classmethod
    def _ensure_registered(cls) -> None:
        """Lazily register adapters on first use."""
        if cls._adapters:
            return

        from Auto3D.isomers.omega_adapter import OmegaIsomerAdapter
        from Auto3D.isomers.rdkit_adapters import (
            RDKitIsomerAdapter,
            RDKitSdfIsomerAdapter,
        )

        cls._adapters = {
            "rdkit": RDKitIsomerAdapter,
            "rdkit_sdf": RDKitSdfIsomerAdapter,
            "omega": OmegaIsomerAdapter,
        }

    @classmethod
    def available_engines(cls) -> list[str]:
        """Return list of available engine types.

        Returns:
            List of supported engine type names.
        """
        cls._ensure_registered()
        return list(cls._adapters.keys())

    @classmethod
    def create(
        cls,
        engine_type: str,
        input_path: str,
        output_path: str,
        *,
        input_format: str = "smi",
        smiles_enumerated: str = "",
        smiles_reduced: str = "",
        smiles_hashed: str = "",
        job_dir: str = "",
        max_confs: int | None = None,
        threshold: float = 0.3,
        n_jobs: int = 4,
        enumerate_isomers: bool = True,
        mode: str = "classic",
        use_parallel_embedding: bool = False,
        parallel_embedding_threshold: int = 10,
        parallel_workers: int = 4,
    ) -> BaseIsomerEngine:
        """Create an isomer engine adapter.

        Args:
            engine_type: Engine type ('rdkit', 'rdkit_sdf', or 'omega').
            input_path: Path to input file.
            output_path: Path for output SDF file.
            input_format: Input format ('smi' or 'sdf'). Used for auto-selection
                of rdkit vs rdkit_sdf when engine_type is 'rdkit'.
            smiles_enumerated: Path for enumerated SMILES (rdkit/omega).
            smiles_reduced: Path for reduced SMILES (rdkit/omega).
            smiles_hashed: Path for hashed SMILES (rdkit/omega).
            job_dir: Working directory (rdkit only).
            max_confs: Maximum conformers per molecule.
            threshold: RMSD threshold for duplicate removal.
            n_jobs: Number of parallel jobs.
            enumerate_isomers: Whether to enumerate stereoisomers.
            mode: Omega mode ('classic', 'macrocycle', etc.) for omega engine.
            use_parallel_embedding: Use parallel conformer embedding (rdkit only).
            parallel_embedding_threshold: Minimum molecules for parallel embedding.
            parallel_workers: Number of worker processes for parallel embedding.

        Returns:
            Configured isomer engine adapter instance.

        Raises:
            ValueError: If engine_type is not recognized.
        """
        cls._ensure_registered()
        engine_type = engine_type.lower()

        # Auto-select rdkit_sdf for SDF input when rdkit is requested
        if engine_type == "rdkit" and input_format.lower() == "sdf":
            engine_type = "rdkit_sdf"

        if engine_type not in cls._adapters:
            available = ", ".join(f"'{e}'" for e in cls._adapters.keys())
            raise ValueError(
                f"Unknown isomer engine type: '{engine_type}'. "
                f"Supported types: {available}"
            )

        adapter_class = cls._adapters[engine_type]

        # Create adapter with appropriate parameters
        if engine_type == "rdkit":
            return adapter_class(
                input_path=input_path,
                output_path=output_path,
                smiles_enumerated=smiles_enumerated,
                smiles_reduced=smiles_reduced,
                smiles_hashed=smiles_hashed,
                job_dir=job_dir,
                max_confs=max_confs,
                threshold=threshold,
                n_jobs=n_jobs,
                enumerate_isomers=enumerate_isomers,
                use_parallel_embedding=use_parallel_embedding,
                parallel_embedding_threshold=parallel_embedding_threshold,
                parallel_workers=parallel_workers,
            )
        elif engine_type == "rdkit_sdf":
            return adapter_class(
                input_path=input_path,
                output_path=output_path,
                max_confs=max_confs,
                threshold=threshold,
                n_jobs=n_jobs,
            )
        elif engine_type == "omega":
            return adapter_class(
                mode=mode,
                input_path=input_path,
                smiles_enumerated=smiles_enumerated,
                smiles_reduced=smiles_reduced,
                smiles_hashed=smiles_hashed,
                output_path=output_path,
                max_confs=max_confs,
                threshold=threshold,
                enumerate_isomers=enumerate_isomers,
            )

        # Should not reach here, but for type safety
        raise ValueError(f"Unknown engine type: {engine_type}")


def create_isomer_engine(
    engine_type: str,
    input_path: str,
    output_path: str,
    *,
    smiles_enumerated: str = "",
    smiles_reduced: str = "",
    smiles_hashed: str = "",
    job_dir: str = "",
    max_confs: int | None = None,
    threshold: float = 0.3,
    n_jobs: int = 4,
    enumerate_isomers: bool = True,
    mode: str = "classic",
    use_parallel_embedding: bool = False,
    parallel_embedding_threshold: int = 10,
    parallel_workers: int = 4,
) -> IsomerEngine:
    """Create an isomer engine based on the specified type.

    This is a convenience wrapper around IsomerEngineFactory.create().

    Args:
        engine_type: Engine type ('rdkit', 'rdkit_sdf', or 'omega').
        input_path: Path to input file.
        output_path: Path for output SDF file.
        smiles_enumerated: Path for enumerated SMILES (rdkit only).
        smiles_reduced: Path for reduced SMILES (rdkit only).
        smiles_hashed: Path for hashed SMILES (rdkit only).
        job_dir: Working directory (rdkit only).
        max_confs: Maximum conformers per molecule.
        threshold: RMSD threshold for duplicate removal.
        n_jobs: Number of parallel jobs.
        enumerate_isomers: Whether to enumerate stereoisomers.
        mode: Omega mode ('classic', 'macrocycle', etc.) for omega engine.
        use_parallel_embedding: Whether to use parallel conformer embedding
            (rdkit only). Default False.
        parallel_embedding_threshold: Minimum molecules for parallel embedding
            (rdkit only). Default 10.
        parallel_workers: Number of worker processes for parallel embedding
            (rdkit only). Default 4.

    Returns:
        Configured isomer engine instance.

    Raises:
        ValueError: If engine_type is not recognized.

    Example:
        >>> engine = create_isomer_engine(
        ...     "rdkit",
        ...     input_path="input.smi",
        ...     output_path="output.sdf",
        ...     smiles_enumerated="enum.smi",
        ...     smiles_reduced="reduced.smi",
        ...     smiles_hashed="hashed.smi",
        ...     job_dir="/path/to/job",
        ... )
        >>> output = engine.run()
    """
    return IsomerEngineFactory.create(
        engine_type=engine_type,
        input_path=input_path,
        output_path=output_path,
        smiles_enumerated=smiles_enumerated,
        smiles_reduced=smiles_reduced,
        smiles_hashed=smiles_hashed,
        job_dir=job_dir,
        max_confs=max_confs,
        threshold=threshold,
        n_jobs=n_jobs,
        enumerate_isomers=enumerate_isomers,
        mode=mode,
        use_parallel_embedding=use_parallel_embedding,
        parallel_embedding_threshold=parallel_embedding_threshold,
        parallel_workers=parallel_workers,
    )


def create_tautomer_engine(
    engine_type: str,
    input_path: str,
    output_path: str,
    pka_norm: bool = True,
) -> TautomerEngine:
    """Create a tautomer engine based on the specified type.

    Args:
        engine_type: Engine type ('rdkit' or 'oechem').
        input_path: Path to input SMI file.
        output_path: Path for output SMI file.
        pka_norm: Normalize ionization to pH ~7.4 (oechem only).

    Returns:
        Configured tautomer engine instance.

    Raises:
        ValueError: If engine_type is not recognized.

    Example:
        >>> engine = create_tautomer_engine("rdkit", "input.smi", "output.smi")
        >>> engine.run()
    """
    engine_type = engine_type.lower()

    if engine_type in ("rdkit", "oechem"):
        from Auto3D.isomer_engine import TautomerEngine as TautEngine

        return TautEngine(
            mode=engine_type,
            input_f=input_path,
            out=output_path,
            pKaNorm=pka_norm,
        )

    else:
        raise ValueError(
            f"Unknown tautomer engine type: '{engine_type}'. "
            f"Supported types: 'rdkit', 'oechem'"
        )
