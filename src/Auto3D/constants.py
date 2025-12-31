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

# Conformer generation formula coefficients
# Based on: https://doi.org/10.1021/acs.jctc.0c01213
CONFORMER_ROTATABLE_COEFF = 8.481  # Coefficient for rotatable bond count
CONFORMER_ROTATABLE_EXP = 1.642    # Exponent for rotatable bond count
CONFORMER_MULTIPLIER = 2           # Multiplier for the formula
CONFORMER_RANDOM_SEED = 42         # Random seed for reproducible embedding

# Optimization sentinel values
INITIAL_FMAX_SENTINEL = 999.0  # Initial value for max force (unconverged)
INITIAL_ENERGY_SENTINEL = 999.0  # Initial value for energy (unconverged)

# Thermodynamics
STANDARD_PRESSURE = 101325  # Pa

# Model names
MODEL_AIMNET = "AIMNET"
MODEL_ANI2X = "ANI2x"
MODEL_ANI2XT = "ANI2xt"

# Supported model names (for validation)
SUPPORTED_MODELS = frozenset({MODEL_AIMNET, MODEL_ANI2X, MODEL_ANI2XT})
