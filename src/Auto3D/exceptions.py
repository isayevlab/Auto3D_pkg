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


class ModelLoadError(ModelError):
    """Raised when a model fails to load.

    This includes corrupted model files or incompatible model formats.
    """
    pass


class NumericalError(ModelError):
    """Raised when numerical instability is detected.

    This includes NaN or Inf values in energies or forces during
    neural network potential calculations. Usually indicates
    problematic molecular geometries or model limitations.
    """
    pass


class OptimizationError(Auto3DError):
    """Base exception for optimization-related errors.

    Raised directly (not through a subclass) when no 3D structure converges
    for a molecule; see WorkflowOrchestrator._run_pipeline and
    batch_opt/model_wrapper.py.
    """
    pass


class FileFormatError(Auto3DError):
    """Raised for unsupported or invalid file formats.

    Auto3D supports .smi (SMILES) and .sdf (SDF) file formats.
    """
    pass


class DependencyError(Auto3DError):
    """Raised when a required dependency is not available.

    Some features require optional dependencies like OpenEye toolkits,
    TorchANI, or ASE. This error is raised when these are needed but not
    installed.

    Attributes:
        dependency_name: A short key identifying the missing dependency
            (e.g. ``"openeye"``, ``"torchani"``, ``"ase"``). ``cli.errors.
            get_error_hint`` looks this up in its own hints map to show an
            install command; before this attribute existed it was read via
            ``getattr(error, "dependency_name", "unknown")`` on every raise
            site, so every dependency failure showed the same
            "Install the missing dependency: unknown" hint (M26). Defaults to
            ``"unknown"`` so a caller that raises ``DependencyError(msg)``
            without naming a dependency still gets a (generic) hint rather
            than a crash in the hint lookup.
    """

    def __init__(self, message: str, dependency_name: str | None = None) -> None:
        super().__init__(message)
        self.dependency_name = dependency_name or "unknown"


class GPUError(Auto3DError):
    """Raised for GPU-related errors.

    This includes missing CUDA devices when GPU computation is requested,
    or GPU memory issues.
    """
    pass
