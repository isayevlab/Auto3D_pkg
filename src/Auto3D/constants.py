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

# Chunk sizing
DEFAULT_CAPACITY = 42  # molecules per GB of GPU/CPU memory for chunk sizing

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

# Default optimization parameters
DEFAULT_RMSD_THRESHOLD = 0.3  # Angstrom, for duplicate conformer removal
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # eV/Angstrom, force convergence
DEFAULT_OPT_STEPS = 2000  # Maximum optimization steps
DEFAULT_PATIENCE = 250  # Steps before dropping oscillating conformer
DEFAULT_BATCHSIZE_ATOMS = 1024  # Atoms per batch for GPU optimization
DEFAULT_ENERGY_CLUSTER_WINDOW = 0.1  # eV, for RMSD clustering
# eV, energy convergence threshold (~0.02 kcal/mol). Must exceed the fp32 ULP at
# typical molecular total energies (~thousands of eV => ULP ~1e-3 eV); a smaller
# value sits below float32 noise and the energy criterion would never fire
# (review finding #23).
DEFAULT_ENERGY_TOL = 1e-3
DEFAULT_ENERGY_PATIENCE = 3  # Steps energy must be stable before converging
DEFAULT_RANDOM_SEED = 42  # Default random seed for reproducibility
