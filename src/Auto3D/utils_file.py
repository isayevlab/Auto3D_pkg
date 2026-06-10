#!/usr/bin/env python
"""
Providing general utilities for working with different formats of molecular files

.. deprecated:: 3.0
    This module is deprecated. Use :mod:`Auto3D.utils.file_ops` instead.
    Functions will be removed in Auto3D v4.0.
"""
from __future__ import annotations

import warnings

_DEPRECATION_MESSAGE = (
    "Auto3D.utils_file is deprecated and will be removed in Auto3D v4.0. "
    "Import from Auto3D.utils.file_ops instead."
)


def _warn() -> None:
    """Emit the module-level deprecation warning."""
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


def guess_file_type(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.guess_file_type`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.guess_file_type(*args, **kwargs)


def smiles2smi(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.smiles2smi`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.smiles2smi(*args, **kwargs)


def combine_smi(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.combine_smi`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.combine_smi(*args, **kwargs)


def countSDF(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.count_sdf`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.count_sdf(*args, **kwargs)


def SDF2chunks(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.SDF2chunks`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.SDF2chunks(*args, **kwargs)


def find_smiles_not_in_sdf(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.find_smiles_not_in_sdf`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.find_smiles_not_in_sdf(*args, **kwargs)


def encode_ids(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.encode_ids`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.encode_ids(*args, **kwargs)


def decode_ids(*args, **kwargs):
    """Deprecated. Delegates to :func:`Auto3D.utils.file_ops.decode_ids`."""
    _warn()
    from Auto3D.utils import file_ops

    return file_ops.decode_ids(*args, **kwargs)
