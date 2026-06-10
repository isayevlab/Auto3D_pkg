"""Base protocols for isomer enumeration engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class IsomerEngine(Protocol):
    """Protocol for isomer enumeration engines.

    All isomer engines must implement this interface to be compatible
    with the Auto3D workflow.

    Example:
        >>> class MyIsomerEngine:
        ...     def run(self) -> str:
        ...         # Generate isomers and return output path
        ...         return "/path/to/output.sdf"
    """

    @abstractmethod
    def run(self) -> str:
        """Execute isomer enumeration.

        Returns:
            Path to the output file containing enumerated isomers.
        """
        ...


@runtime_checkable
class TautomerEngine(Protocol):
    """Protocol for tautomer enumeration engines.

    Example:
        >>> class MyTautomerEngine:
        ...     def run(self) -> None:
        ...         # Enumerate tautomers and write to output
        ...         pass
    """

    @abstractmethod
    def run(self) -> None:
        """Execute tautomer enumeration."""
        ...


class BaseIsomerEngine(ABC):
    """Abstract base class for isomer engines.

    Provides common functionality for all isomer engine implementations.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_confs: int | None = None,
        threshold: float = 0.3,
        n_jobs: int = 4,
    ) -> None:
        """Initialize the isomer engine.

        Args:
            input_path: Path to input file (SMI or SDF).
            output_path: Path for output SDF file.
            max_confs: Maximum conformers per molecule. None for dynamic.
            threshold: RMSD threshold for duplicate removal (Å).
            n_jobs: Number of parallel jobs for conformer generation.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.max_confs = max_confs
        self.threshold = threshold
        self.n_jobs = n_jobs

    @abstractmethod
    def run(self) -> str:
        """Execute isomer enumeration.

        Returns:
            Path to the output SDF file.
        """
        ...
