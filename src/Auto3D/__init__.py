"""
Auto3D - Automatic generation of low-energy 3D molecular conformers.

This package provides tools for generating 3D conformers from SMILES/SDF files
using neural network potentials (AIMNet2, ANI2x, ANI2xt).
"""

import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

# Optional dependency imports with proper exception handling
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    try:
        from openeye import oechem, oeomega, oequacpac  # noqa: F401  (optional dependency probe)
    except ImportError:
        pass  # OpenEye is optional

    try:
        import torchani  # noqa: F401  (optional dependency probe)
    except ImportError:
        pass  # TorchANI is optional

    try:
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt  # noqa: F401  (optional dependency probe)
    except ImportError:
        pass  # ANI2xt model is optional

__all__ = [
    "__version__",
    # Core API
    "main",
    "generate_conformers",  # canonical alias for main()
    "smiles2mols",
    # Configuration
    "Auto3DOptions",
    "OptimizationConfig",
    "NNPModel",
    # Model creation
    "create_model",
    "ModelFactory",
    # Property calculators
    "calc_spe",
    "opt_geometry",
    "calc_thermo",
    # Tautomers
    "get_stable_tautomers",
    "select_tautomers",
]

# name -> (module, attribute) for lazy public-API imports. generate_conformers is
# the canonical, self-describing alias for main(); main stays for back-compat.
_LAZY_API: dict[str, tuple[str, str]] = {
    "main": ("Auto3D.auto3D", "main"),
    "generate_conformers": ("Auto3D.auto3D", "main"),
    "smiles2mols": ("Auto3D.auto3D", "smiles2mols"),
    "Auto3DOptions": ("Auto3D.config", "Auto3DOptions"),
    "OptimizationConfig": ("Auto3D.config", "OptimizationConfig"),
    "NNPModel": ("Auto3D.config", "NNPModel"),
    "create_model": ("Auto3D.model_factory", "create_model"),
    "ModelFactory": ("Auto3D.model_factory", "ModelFactory"),
    "calc_spe": ("Auto3D.SPE", "calc_spe"),
    "opt_geometry": ("Auto3D.ASE.geometry", "opt_geometry"),
    "calc_thermo": ("Auto3D.ASE.thermo", "calc_thermo"),
    "get_stable_tautomers": ("Auto3D.tautomer", "get_stable_tautomers"),
    "select_tautomers": ("Auto3D.tautomer", "select_tautomers"),
}


# Lazy imports for public API
def __getattr__(name: str):
    """Lazy import for public API functions (see _LAZY_API)."""
    if name in _LAZY_API:
        import importlib
        module_name, attr = _LAZY_API[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
