"""Isomer enumeration engines with Strategy Pattern.

This package provides a unified interface for different isomer enumeration
backends (RDKit, OpenEye Omega) through the Strategy Pattern.

``IsomerEngineFactory`` is the one name re-exported here, because
``docs/source/api.rst`` documents it at this package path
(``Auto3D.isomers.IsomerEngineFactory``) and that path is the public one -- which
makes this the only ``__init__.py`` in ``Auto3D`` allowed to re-export anything.
Every other name in this package -- ``create_tautomer_engine`` and the
``IsomerEngine``/``TautomerEngine`` protocols -- is imported from the module that
defines it (``Auto3D.isomers.factory``, ``Auto3D.isomers.base``), so that no name
here has two supported spellings. A ``create_isomer_engine`` free function was
the exception until 3.0.0 and is deleted; see ``factory.py``'s docstring for why
``create_tautomer_engine`` beside it was kept.

Example:
    >>> from Auto3D.isomers import IsomerEngineFactory
    >>> engine = IsomerEngineFactory.create("rdkit", input_path="input.smi", ...)
    >>> engine.run()
"""
from __future__ import annotations

from Auto3D.isomers.factory import IsomerEngineFactory

__all__ = ["IsomerEngineFactory"]
