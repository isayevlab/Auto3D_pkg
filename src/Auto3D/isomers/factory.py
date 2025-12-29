"""Factory functions for creating isomer engines."""
from __future__ import annotations

from typing import TYPE_CHECKING

from Auto3D.isomers.base import IsomerEngine, TautomerEngine

if TYPE_CHECKING:
    pass


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
) -> IsomerEngine:
    """Create an isomer engine based on the specified type.

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
    engine_type = engine_type.lower()

    if engine_type == "rdkit":
        from Auto3D.isomer_engine import RDKitIsomer

        return RDKitIsomer(
            smi=input_path,
            smiles_enumerated=smiles_enumerated,
            smiles_enumerated_reduced=smiles_reduced,
            smiles_hashed=smiles_hashed,
            enumerated_sdf=output_path,
            job_name=job_dir,
            max_confs=max_confs,
            threshold=threshold,
            np=n_jobs,
            flipper=enumerate_isomers,
        )

    elif engine_type == "rdkit_sdf":
        from Auto3D.isomer_engine import RDKitSdfIsomer

        return RDKitSdfIsomer(
            sdf=input_path,
            enumerated_sdf=output_path,
            max_confs=max_confs,
            threshold=threshold,
            np=n_jobs,
        )

    elif engine_type == "omega":
        # Create a wrapper for the oe_isomer function
        from Auto3D.isomers.omega_adapter import OmegaIsomerAdapter

        return OmegaIsomerAdapter(
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

    else:
        raise ValueError(
            f"Unknown isomer engine type: '{engine_type}'. "
            f"Supported types: 'rdkit', 'rdkit_sdf', 'omega'"
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
