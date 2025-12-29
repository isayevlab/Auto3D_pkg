"""Physical and computational constants for Auto3D."""
from __future__ import annotations

# Energy conversion factors (CODATA 2018)
HARTREE_TO_EV = 27.211386245988  # 1 Hartree in eV
EV_TO_KCAL_PER_MOL = 23.060547830619026  # 1 eV in kcal/mol (original Auto3D value)
HARTREE_TO_KCAL_PER_MOL = 627.50947337481  # 1 Hartree in kcal/mol

# Geometry thresholds
BOND_STRETCH_TOLERANCE = 1.25  # Maximum bond stretch factor
COLLISION_THRESHOLD = 1.1  # Å, minimum distance for clash detection
MIN_ATOM_DISTANCE = 0.9  # Å, minimum allowed interatomic distance

# Conformer generation limits
MAX_CONFORMERS_CAP = 1000  # Maximum conformers per molecule

# Thermodynamics
STANDARD_PRESSURE = 101325  # Pa

# Model names
MODEL_AIMNET = "AIMNET"
MODEL_ANI2X = "ANI2x"
MODEL_ANI2XT = "ANI2xt"

# Supported model names (for validation)
SUPPORTED_MODELS = frozenset({MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT})
