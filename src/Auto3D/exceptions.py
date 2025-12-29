"""
Exception hierarchy for Auto3D package.

This module defines a structured exception hierarchy for better error handling
and more informative error messages throughout the package.
"""


class Auto3DError(Exception):
    """Base exception for all Auto3D errors.

    All Auto3D-specific exceptions inherit from this class, allowing users
    to catch all Auto3D errors with a single except clause if desired.
    """
    pass


class ConfigurationError(Auto3DError):
    """Raised when there's an invalid configuration.

    This includes missing required parameters, invalid parameter values,
    or incompatible parameter combinations.
    """
    pass


class InputValidationError(Auto3DError):
    """Raised when input file validation fails.

    This includes invalid SMILES strings, malformed SDF files,
    or files with missing required fields.
    """
    pass


class ModelError(Auto3DError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model cannot be found.

    This includes missing model files or invalid model names.
    """
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load.

    This includes corrupted model files or incompatible model formats.
    """
    pass


class OptimizationError(Auto3DError):
    """Base exception for optimization-related errors."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when geometry optimization fails to converge.

    This may indicate that the optimization parameters need adjustment
    or that the molecular structure is problematic.
    """
    pass


class IsomerEnumerationError(Auto3DError):
    """Raised when stereoisomer enumeration fails.

    This includes failures in the RDKit or OpenEye isomer engines.
    """
    pass


class TautomerEnumerationError(Auto3DError):
    """Raised when tautomer enumeration fails."""
    pass


class FileFormatError(Auto3DError):
    """Raised for unsupported or invalid file formats.

    Auto3D supports .smi (SMILES) and .sdf (SDF) file formats.
    """
    pass


class DependencyError(Auto3DError):
    """Raised when a required dependency is not available.

    Some features require optional dependencies like OpenEye toolkits
    or TorchANI. This error is raised when these are needed but not installed.
    """
    pass


class GPUError(Auto3DError):
    """Raised for GPU-related errors.

    This includes missing CUDA devices when GPU computation is requested,
    or GPU memory issues.
    """
    pass
