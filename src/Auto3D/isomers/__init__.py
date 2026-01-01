"""Isomer enumeration engines with Strategy Pattern.

This module provides a unified interface for different isomer enumeration
backends (RDKit, OpenEye Omega) through the Strategy Pattern.

Example:
    >>> from Auto3D.isomers import IsomerEngineFactory
    >>> engine = IsomerEngineFactory.create("rdkit", input_path="input.smi", ...)
    >>> engine.run()

    # Or using the convenience function
    >>> from Auto3D.isomers import create_isomer_engine
    >>> engine = create_isomer_engine("rdkit", input_path="input.smi", ...)
"""
from __future__ import annotations

from Auto3D.isomers.base import BaseIsomerEngine, IsomerEngine, TautomerEngine
from Auto3D.isomers.factory import (
    IsomerEngineFactory,
    create_isomer_engine,
    create_tautomer_engine,
)

__all__ = [
    # Protocols and base classes
    "IsomerEngine",
    "TautomerEngine",
    "BaseIsomerEngine",
    # Factory
    "IsomerEngineFactory",
    # Convenience functions
    "create_isomer_engine",
    "create_tautomer_engine",
]
