"""Factory classes and functions for creating isomer engines."""
from __future__ import annotations

from collections.abc import Callable

from Auto3D.isomer_engine import (
    RDKitIsomer,
    RDKitSdfIsomer,
    oe_isomer,
)
from Auto3D.isomer_engine import (
    TautomerEngine as _RDKitOrOmegaTautomerEngine,
)
from Auto3D.isomers.base import IsomerEngine, TautomerEngine

#: Engine names :meth:`IsomerEngineFactory.create` accepts. ``rdkit`` is also
#: reachable as ``rdkit_sdf`` via the ``input_format="sdf"`` auto-selection
#: below.
_ENGINE_TYPES = ("rdkit", "rdkit_sdf", "omega")


class _DeferredIsomerEngine:
    """The ``.run() -> str`` object :meth:`IsomerEngineFactory.create` returns.

    Holds nothing but the zero-argument callable that builds the concrete
    engine and runs it. Construction stays deferred to ``run()`` on purpose and
    is not an implementation detail: ``RDKitIsomer.__init__`` creates its
    ``rdk_tmp`` working directory, so building the engine at ``create()`` time
    would move a filesystem side effect (and its ``FileExistsError``) earlier
    than every caller was written for.

    This one class replaces three adapter classes and an abstract base that
    existed only to copy the same eight-to-thirteen keyword arguments twice --
    once into an adapter's ``__init__``, once out of it into the real engine.
    ``create`` below now maps them once, at the single site that knows what
    each engine takes.
    """

    __slots__ = ("_build_and_run", "output_path")

    def __init__(self, build_and_run: Callable[[], str], output_path: str) -> None:
        self._build_and_run = build_and_run
        self.output_path = output_path

    def run(self) -> str:
        """Build the concrete engine, run it, and return its output path."""
        return self._build_and_run()


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

    @classmethod
    def available_engines(cls) -> list[str]:
        """Return list of available engine types.

        Returns:
            List of supported engine type names.
        """
        return list(_ENGINE_TYPES)

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
    ) -> IsomerEngine:
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
            Configured isomer engine, whose ``run()`` builds and drives the
            concrete engine and returns the output path.

        Raises:
            ValueError: If engine_type is not recognized.
        """
        engine_type = engine_type.lower()

        # Auto-select rdkit_sdf for SDF input when rdkit is requested
        if engine_type == "rdkit" and input_format.lower() == "sdf":
            engine_type = "rdkit_sdf"

        if engine_type not in _ENGINE_TYPES:
            available = ", ".join(f"'{e}'" for e in _ENGINE_TYPES)
            raise ValueError(
                f"Unknown isomer engine type: '{engine_type}'. "
                f"Supported types: {available}"
            )

        # The one kwarg-mapping site. Each branch names exactly the arguments
        # its engine takes, so an argument no engine reads cannot survive here
        # unnoticed the way it could when three adapters each held a partial
        # copy of the same signature.
        if engine_type == "rdkit":
            def build_and_run() -> str:
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
                    use_parallel_embedding=use_parallel_embedding,
                    parallel_embedding_threshold=parallel_embedding_threshold,
                    parallel_workers=parallel_workers,
                ).run()

        elif engine_type == "rdkit_sdf":
            def build_and_run() -> str:
                return RDKitSdfIsomer(
                    sdf=input_path,
                    enumerated_sdf=output_path,
                    max_confs=max_confs,
                    threshold=threshold,
                    np=n_jobs,
                    flipper=enumerate_isomers,
                ).run()

        else:  # "omega" -- the only remaining possibility, checked above
            def build_and_run() -> str:
                # oe_isomer is a function returning 0, not an engine object, so
                # this is the one branch that still has an adapting step: run it
                # and report the path it wrote.
                oe_isomer(
                    mode=mode,
                    input_f=input_path,
                    smiles_enumerated=smiles_enumerated,
                    smiles_reduced=smiles_reduced,
                    smiles_hashed=smiles_hashed,
                    output=output_path,
                    max_confs=max_confs,
                    threshold=threshold,
                    flipper=enumerate_isomers,
                )
                return output_path

        return _DeferredIsomerEngine(build_and_run, output_path)


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
        return _RDKitOrOmegaTautomerEngine(
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
