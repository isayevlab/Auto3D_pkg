"""
Auto3D - Automatic generation of low-energy 3D molecular conformers.

This package provides tools for generating 3D conformers from SMILES/SDF files
using neural network potentials (AIMNet2, ANI2x, ANI2xt).
"""

import warnings
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

# Optional dependency imports with proper exception handling
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    try:
        from openeye import oechem
        from openeye import oequacpac
        from openeye import oeomega
    except ImportError:
        pass  # OpenEye is optional

    try:
        import torchani
    except ImportError:
        pass  # TorchANI is optional

    try:
        from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
    except ImportError:
        pass  # ANI2xt model is optional

__all__ = [
    "__version__",
    # Core API
    "main",
    "options",
    "smiles2mols",
    # Configuration
    "Auto3DOptions",
    "NNPModel",
    # Model creation
    "create_model",
    "ModelFactory",
    # Utilities
    "calc_spe",
    "opt_geometry",
    "calc_thermo",
]

# Lazy imports for public API
def __getattr__(name: str):
    """Lazy import for public API functions."""
    if name == "main":
        from Auto3D.auto3D import main
        return main
    elif name == "options":
        from Auto3D.auto3D import options
        return options
    elif name == "smiles2mols":
        from Auto3D.auto3D import smiles2mols
        return smiles2mols
    elif name == "Auto3DOptions":
        from Auto3D.config import Auto3DOptions
        return Auto3DOptions
    elif name == "NNPModel":
        from Auto3D.config import NNPModel
        return NNPModel
    elif name == "create_model":
        from Auto3D.model_factory import create_model
        return create_model
    elif name == "ModelFactory":
        from Auto3D.model_factory import ModelFactory
        return ModelFactory
    elif name == "calc_spe":
        from Auto3D.SPE import calc_spe
        return calc_spe
    elif name == "opt_geometry":
        from Auto3D.ASE.geometry import opt_geometry
        return opt_geometry
    elif name == "calc_thermo":
        from Auto3D.ASE.thermo import calc_thermo
        return calc_thermo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
