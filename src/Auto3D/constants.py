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

# Stereoisomer enumeration limit. RDKit's StereoEnumerationOptions defaults
# maxIsomers to 1024, which silently truncates molecules with >=11 unspecified
# stereocenters. Raise the cap high so realistic inputs are not silently lost,
# and warn when the enumerator returns exactly the cap (likely truncated).
MAX_STEREOISOMERS = 2 ** 16  # 65536

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

# Backward-compatible alias: "AIMNET" now maps to the aimnet registry default.
DEFAULT_AIMNET_MODEL = "aimnet2"
# Built-in (non-aimnet) engines kept for back-compat.
BUILTIN_ANI_MODELS = frozenset({MODEL_ANI2X.upper(), MODEL_ANI2XT.upper()})

# Default optimization parameters
DEFAULT_RMSD_THRESHOLD = 0.3  # Angstrom, for duplicate conformer removal
# eV, energy tolerance for the duplicate-conformer test. RMSD dedup compares
# heavy-atom skeletons only, so conformers differing solely in an O-H / N-H rotor
# orientation collapse to RMSD~=0 even though they are distinct minima with
# different energies. Two structures are treated as duplicates only when their
# heavy-atom RMSD is below threshold AND their energies agree within this
# tolerance, so genuine rotamers (which the H-rich conformer budget deliberately
# samples) survive. ~0.23 kcal/mol -- above post-optimization fp32 energy noise
# (~1e-3 eV) so truly identical conformers still dedup.
DEFAULT_DUPLICATE_ENERGY_TOL = 0.01
DEFAULT_CONVERGENCE_THRESHOLD = 0.01  # eV/Angstrom, force convergence
# eV/Angstrom, the deliberately tighter pre-optimization force tolerance used by
# the thermochemistry path (calc_thermo): a true minimum is needed before a
# Hessian/frequency calculation, unlike conformer generation where 0.01 suffices.
DEFAULT_THERMO_CONVERGENCE_THRESHOLD = 2e-4
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

# A linear molecule has one vanishing principal moment of inertia. This is the
# largest smallest-to-largest moment ratio still treated as linear; it is
# dimensionless, so unlike an absolute coordinate tolerance it behaves the same
# for a diatomic and for a long chain. 1e-3 keeps a CO2 bent by 0.01 A linear
# while classifying a 0.3 A bend as nonlinear.
LINEARITY_MOMENT_RATIO = 1e-3
