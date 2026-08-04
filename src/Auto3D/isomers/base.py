"""Base protocols for isomer enumeration engines."""
from __future__ import annotations

from abc import abstractmethod
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
