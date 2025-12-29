"""I/O utilities with Repository Pattern for molecule file handling.

This module provides a unified interface for reading and writing
molecular data in different formats (SDF, SMI).

Example:
    >>> from Auto3D.io import SDFRepository, SMIRepository
    >>> repo = SDFRepository()
    >>> molecules = repo.read("input.sdf")
    >>> repo.write("output.sdf", molecules)
"""
from __future__ import annotations

from Auto3D.io.repositories import (
    MoleculeRepository,
    SDFRepository,
    SMIRepository,
    read_molecules,
    write_molecules,
)

__all__ = [
    "MoleculeRepository",
    "SDFRepository",
    "SMIRepository",
    "read_molecules",
    "write_molecules",
]
