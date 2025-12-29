"""Isomer enumeration engines with Strategy Pattern.

This module provides a unified interface for different isomer enumeration
backends (RDKit, OpenEye Omega) through the Strategy Pattern.

Example:
    >>> from Auto3D.isomers import create_isomer_engine, IsomerEngine
    >>> engine = create_isomer_engine("rdkit", input_path="input.smi", ...)
    >>> engine.run()
"""
from __future__ import annotations

from Auto3D.isomers.base import IsomerEngine, TautomerEngine
from Auto3D.isomers.factory import create_isomer_engine, create_tautomer_engine

__all__ = [
    "IsomerEngine",
    "TautomerEngine",
    "create_isomer_engine",
    "create_tautomer_engine",
]
